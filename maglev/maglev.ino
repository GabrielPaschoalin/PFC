#include <Arduino.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ensaios_identificacao.h"

// ============================================================
// MAGLEV - V35 - PULSO HALL LOGGER + SYNC LED POR PWM
// ------------------------------------------------------------
// Versao limpa para aquisicao rapida:
// - Sem ToF
// - Sem Kalman atual
// - Sem controle PD por posicao
// - Sem Hall filtrado
// - Ensaio por pulsos parametrizaveis
// - PWM manual numerico mantido
// - Log CSV minimo a cada 2 ms
// - Sync LED:
//   * 200 ms no inicio do ensaio
//   * 200 ms no fim do ensaio
//   * a cada borda de subida do PWM do ensaio por max(10% do pulso, 10 ms)
// ============================================================
//
// Comandos seriais:
//   PULSO 200 30 1000 10        -> PWM=200, pulso=30 ms, intervalo=1000 ms, 10 repeticoes
//   PULSO 200 30 1000 10 500    -> igual acima, com pre-inicio de 500 ms em PWM=0
//   120                         -> PWM manual direto = 120
//   OFF                         -> para ensaio e zera PWM
//   RESET / R                   -> para ensaio, zera PWM e apaga LED de sync
//
// CSV continuo:
//   T_US,HALL_INF_MV,HALL_SUP_MV,PWM,SYNC_PULSE
// ============================================================

// ------------------- SERIAL -------------------
// Alto o suficiente para folga em log de 2 ms com linha CSV curta.
// Se o conversor USB-serial nao aceitar, reduza para 2000000 ou 921600.
const uint32_t SERIAL_BAUD = 2000000UL;

// ------------------- PINOS -------------------
const int pinPWM = 27;             // PWM no IN1 do L298N
const int pinENA = 14;             // ENA fixo em HIGH
const int pinIN2 = 26;             // IN2 fixo em LOW
const int pinHallCimaAnalog = 36;  // Hall superior
const int pinHallBaixoAnalog = 34; // Hall inferior
const int pinSyncLed = 12;         // LED para sincronismo com video/log

// ------------------- PWM -------------------
const uint32_t freqPWM = 10000;
const uint8_t resolucaoPWM = 8;
const int pwmManualMin = 0;
const int pwmManualMax = 255;
bool pwmAttached = false;
int pwmAtual = 0;

// ------------------- SCHEDULER -------------------
const unsigned long BASE_TICK_US = 2000UL;  // 2 ms => 500 Hz
const uint32_t MAX_CATCHUP_TICKS = 8U;
unsigned long lastBaseTickUs = 0UL;
uint32_t tickCounter = 0;

// ------------------- HALLS -------------------
uint32_t hall_inf_mv = 0;
uint32_t hall_sup_mv = 0;

// ------------------- SYNC LED -------------------
bool syncPulseActive = false;
unsigned long syncPulseEndUs = 0UL;
const unsigned long SYNC_PULSE_FIXED_US = 200000UL;  // 200 ms para inicio/fim do ensaio
const unsigned long SYNC_PULSE_MIN_PWM_US = 10000UL; // minimo de 10 ms para pulsos vinculados ao PWM
unsigned long syncPulsePwmUs = SYNC_PULSE_MIN_PWM_US;
bool syncEndPulseEmitido = false;

// ------------------- ENSAIO -------------------
SaidasIdentificacao saidaIdent;
DadosIdentificacao dadosIdent;

// ------------------- SERIAL RX -------------------
const size_t CMD_BUF_LEN = 96;
char cmdBuffer[CMD_BUF_LEN];
size_t cmdLen = 0;
unsigned long ultimoRxMs = 0UL;
const unsigned long serialFlushIdleMs = 8UL;

int clampInt(int x, int lo, int hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

unsigned long calcularPulsoSyncPwmUs(uint16_t tempoPulsoMs) {
  unsigned long dezPorCentoUs = ((unsigned long)tempoPulsoMs * 1000UL) / 10UL;
  if (dezPorCentoUs < SYNC_PULSE_MIN_PWM_US) return SYNC_PULSE_MIN_PWM_US;
  return dezPorCentoUs;
}

void iniciarPulsoSyncDuracao(unsigned long nowUs, unsigned long duracaoUs) {
  unsigned long novoFimUs = nowUs + duracaoUs;

  // Se ja existe um pulso de sync mais longo em andamento, nao encurta.
  if (syncPulseActive && (long)(syncPulseEndUs - novoFimUs) > 0L) {
    digitalWrite(pinSyncLed, HIGH);
    return;
  }

  syncPulseActive = true;
  syncPulseEndUs = novoFimUs;
  digitalWrite(pinSyncLed, HIGH);
}

void iniciarPulsoSyncFixo(unsigned long nowUs) {
  iniciarPulsoSyncDuracao(nowUs, SYNC_PULSE_FIXED_US);
}

void iniciarPulsoSyncPwm(unsigned long nowUs) {
  iniciarPulsoSyncDuracao(nowUs, syncPulsePwmUs);
}

void pararPulsoSync() {
  syncPulseActive = false;
  syncPulseEndUs = 0UL;
  digitalWrite(pinSyncLed, LOW);
}

void atualizarPulsoSync(unsigned long nowUs) {
  if (syncPulseActive && (long)(nowUs - syncPulseEndUs) >= 0L) {
    pararPulsoSync();
  }
}

void aplicarPWM(int valor, bool gerarSyncNaBorda = false, unsigned long nowUs = 0UL) {
  int pwmAnterior = pwmAtual;
  pwmAtual = clampInt(valor, pwmManualMin, pwmManualMax);

  if (pwmAttached) {
    ledcWrite(pinPWM, (uint32_t)pwmAtual);
  }

  if (gerarSyncNaBorda && pwmAnterior == 0 && pwmAtual > 0) {
    iniciarPulsoSyncPwm(nowUs == 0UL ? micros() : nowUs);
  }
}

void atualizarLeituraHall() {
  hall_inf_mv = analogReadMilliVolts(pinHallBaixoAnalog);
  hall_sup_mv = analogReadMilliVolts(pinHallCimaAnalog);
}

EntradasIdentificacao montarEntradasIdent(unsigned long nowUs) {
  EntradasIdentificacao in;
  in.now_us = nowUs;
  in.pwm_atual = pwmAtual;
  in.hall_inf_mv = hall_inf_mv;
  in.hall_sup_mv = hall_sup_mv;
  return in;
}

void iniciarPulsoParametrizado(unsigned long nowUs,
                               int pwmPulso,
                               uint16_t tempoPulsoMs,
                               uint16_t tempoEntrePulsosMs,
                               uint8_t repeticoes,
                               uint16_t tempoPreInicioMs) {
  ensaiosIdentificacaoStop();
  aplicarPWM(0);
  pararPulsoSync();

  syncPulsePwmUs = calcularPulsoSyncPwmUs(tempoPulsoMs);
  syncEndPulseEmitido = false;

  PerfilIdentificacao perfil = ensaiosIdentificacaoGetPerfilAtivo();
  perfil.pwm_pulso = (uint8_t)clampInt(pwmPulso, pwmManualMin, pwmManualMax);
  perfil.tempo_pulso_ms = tempoPulsoMs;
  perfil.tempo_entre_pulsos_ms = tempoEntrePulsosMs;
  perfil.repeticoes = repeticoes;
  perfil.tempo_pre_inicio_ms = tempoPreInicioMs;

  EntradasIdentificacao in = montarEntradasIdent(nowUs);
  ensaiosIdentificacaoStartComPerfil(in, perfil);

  // Mantem o marcador fixo antigo de inicio do ensaio.
  iniciarPulsoSyncFixo(nowUs);
}

void pararTudo() {
  ensaiosIdentificacaoStop();
  aplicarPWM(0);
  pararPulsoSync();
  syncEndPulseEmitido = false;
}

// ============================================================
// SERIAL - PARSER
// ============================================================
void trimInPlace(char* s) {
  if (!s) return;

  size_t len = strlen(s);
  size_t start = 0;
  while (start < len && isspace((unsigned char)s[start])) start++;

  size_t end = len;
  while (end > start && isspace((unsigned char)s[end - 1])) end--;

  if (start > 0) {
    memmove(s, s + start, end - start);
  }
  s[end - start] = '\0';
}

bool equalsIgnoreCase(const char* a, const char* b) {
  if (!a || !b) return false;
  while (*a && *b) {
    if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return false;
    a++;
    b++;
  }
  return (*a == '\0' && *b == '\0');
}

bool isUnsignedIntString(const char* s) {
  if (!s || *s == '\0') return false;
  while (*s) {
    if (!isdigit((unsigned char)*s)) return false;
    s++;
  }
  return true;
}

void processarComandoC(const char* comandoBruto) {
  if (!comandoBruto) return;

  char comando[CMD_BUF_LEN];
  strncpy(comando, comandoBruto, CMD_BUF_LEN - 1);
  comando[CMD_BUF_LEN - 1] = '\0';
  trimInPlace(comando);
  if (comando[0] == '\0') return;

  // Comando parametrizado:
  //   PULSO <pwm> <tempo_pulso_ms> <tempo_entre_pulsos_ms> <repeticoes> [tempo_pre_inicio_ms]
  {
    char cmdWord[12] = {0};
    int pwmPulso = 0;
    int tempoPulsoMs = 0;
    int tempoEntrePulsosMs = 0;
    int repeticoes = 0;
    int tempoPreInicioMs = 0;

    int n = sscanf(comando,
                   "%11s %d %d %d %d %d",
                   cmdWord,
                   &pwmPulso,
                   &tempoPulsoMs,
                   &tempoEntrePulsosMs,
                   &repeticoes,
                   &tempoPreInicioMs);

    if ((n == 5 || n == 6) && equalsIgnoreCase(cmdWord, "PULSO")) {
      bool ok = true;
      ok = ok && (pwmPulso >= pwmManualMin && pwmPulso <= pwmManualMax);
      ok = ok && (tempoPulsoMs >= 1 && tempoPulsoMs <= 60000);
      ok = ok && (tempoEntrePulsosMs >= 0 && tempoEntrePulsosMs <= 60000);
      ok = ok && (repeticoes >= 1 && repeticoes <= 255);
      ok = ok && (tempoPreInicioMs >= 0 && tempoPreInicioMs <= 60000);

      if (ok) {
        iniciarPulsoParametrizado(micros(),
                                  pwmPulso,
                                  (uint16_t)tempoPulsoMs,
                                  (uint16_t)tempoEntrePulsosMs,
                                  (uint8_t)repeticoes,
                                  (uint16_t)tempoPreInicioMs);
      }
      return;
    }
  }

  if (equalsIgnoreCase(comando, "OFF") || equalsIgnoreCase(comando, "RESET") || equalsIgnoreCase(comando, "R")) {
    pararTudo();
    return;
  }

  // PWM manual numerico direto: 0..255
  if (isUnsignedIntString(comando)) {
    long valor = strtol(comando, nullptr, 10);
    if (valor >= pwmManualMin && valor <= pwmManualMax) {
      ensaiosIdentificacaoStop();
      pararPulsoSync();
      aplicarPWM((int)valor);
    }
    return;
  }
}

void flushSerialCommandBuffer() {
  if (cmdLen == 0) return;
  cmdBuffer[cmdLen] = '\0';
  processarComandoC(cmdBuffer);
  cmdLen = 0;
}

void pollSerialCommands() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    ultimoRxMs = millis();

    if (ch == '\r' || ch == '\n') {
      flushSerialCommandBuffer();
      continue;
    }

    if (cmdLen < (CMD_BUF_LEN - 1)) {
      cmdBuffer[cmdLen++] = ch;
    } else {
      cmdLen = 0;
    }
  }

  // Permite comando sem line ending, como nas versoes anteriores.
  if (cmdLen > 0 && (millis() - ultimoRxMs) >= serialFlushIdleMs) {
    flushSerialCommandBuffer();
  }
}

// ============================================================
// TELEMETRIA RAPIDA
// ============================================================
void printCsvHeader() {
  Serial.println("T_US,HALL_INF_MV,HALL_SUP_MV,PWM,SYNC_PULSE");
}

void printFastTelemetry(unsigned long nowUs) {
  char line[96];
  int n = snprintf(line,
                   sizeof(line),
                   "%lu,%lu,%lu,%d,%d\n",
                   nowUs,
                   (unsigned long)hall_inf_mv,
                   (unsigned long)hall_sup_mv,
                   pwmAtual,
                   syncPulseActive ? 1 : 0);

  if (n > 0) {
    Serial.write((const uint8_t*)line, (size_t)n);
  }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
#if defined(ARDUINO_ARCH_ESP32)
  Serial.setRxBufferSize(512);
  Serial.setTxBufferSize(4096);
#endif
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(2);

  analogReadResolution(12);

  pinMode(pinENA, OUTPUT);
  digitalWrite(pinENA, HIGH);

  pinMode(pinIN2, OUTPUT);
  digitalWrite(pinIN2, LOW);

  pinMode(pinHallBaixoAnalog, INPUT);
  pinMode(pinHallCimaAnalog, INPUT);

  pinMode(pinSyncLed, OUTPUT);
  digitalWrite(pinSyncLed, LOW);

#if defined(ARDUINO_ARCH_ESP32)
  analogSetPinAttenuation(pinHallBaixoAnalog, ADC_11db);
  analogSetPinAttenuation(pinHallCimaAnalog, ADC_11db);
#endif

  pwmAttached = ledcAttach(pinPWM, freqPWM, resolucaoPWM);
  if (!pwmAttached) {
    while (true) {
      delay(1000);
    }
  }

  aplicarPWM(0);
  ensaiosIdentificacaoBegin();
  atualizarLeituraHall();

  printCsvHeader();
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  pollSerialCommands();

  unsigned long nowUs = micros();
  atualizarPulsoSync(nowUs);

  if (lastBaseTickUs == 0UL) {
    lastBaseTickUs = nowUs;
    return;
  }

  uint32_t ticksProcessed = 0;
  while ((unsigned long)(nowUs - lastBaseTickUs) >= BASE_TICK_US && ticksProcessed < MAX_CATCHUP_TICKS) {
    lastBaseTickUs += BASE_TICK_US;
    tickCounter++;

    atualizarLeituraHall();

    EntradasIdentificacao in = montarEntradasIdent(lastBaseTickUs);
    ensaiosIdentificacaoUpdate(in, saidaIdent);

    if (saidaIdent.ativo || saidaIdent.finalizado || saidaIdent.abortado) {
      aplicarPWM(saidaIdent.pwm_cmd, true, lastBaseTickUs);
    }

    if (saidaIdent.finalizado && !syncEndPulseEmitido) {
      iniciarPulsoSyncFixo(lastBaseTickUs);
      syncEndPulseEmitido = true;
    }

    printFastTelemetry(lastBaseTickUs);

    ticksProcessed++;
    nowUs = micros();
    atualizarPulsoSync(nowUs);
  }

  if (ticksProcessed >= MAX_CATCHUP_TICKS) {
    lastBaseTickUs = nowUs;
  }
}
