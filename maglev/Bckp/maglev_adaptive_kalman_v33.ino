#include <Arduino.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "leitura_sensor.h"
#include "ensaios_identificacao.h"

// ============================================================
// MAGLEV - KALMAN ADAPTATIVO + IDENTIFICACAO POR PULSO FIXO - V33
// ------------------------------------------------------------
// Protocolo desta versao:
// 1) preserva a base V32 de PWM, pinos, scheduler, dois Halls e telemetria
// 2) remove a dependencia do ToF na logica de identificacao
// 3) aplica pulsos fixos parametrizados: PWM, tempo, intervalo e repeticoes
// 4) o ToF continua sendo lido e logado apenas como informacao auxiliar
// 5) o LED de sincronismo acende no inicio do teste para alinhar video x log
//
// Comandos principais:
//   PULSO 200 30 1000 10      -> PWM=200, pulso=30 ms, intervalo=1000 ms, 10 repeticoes
//   INICIAR_IDENT             -> usa o perfil default do ensaio
//   OFF                       -> aborta/zera PWM
// ============================================================

static const char TEST_ID_FALLBACK[] = "IDENT_PULSO_FIXO";
static const uint8_t SENSOR_TYPE = 2;  // 1=agressivo, 2=meio-termo, 3=conservador

// ------------------- PINOS -------------------
const int pinPWM = 27;
const int pinENA = 14;
const int pinIN2 = 26;
const int pinHallCimaAnalog = 36;
const int pinHallBaixoAnalog = 34;
const int pinHallAnalog = pinHallBaixoAnalog;  // compatibilidade: logica atual/ensaios seguem usando o Hall de baixo
const int pinSyncLed = 12;

// ------------------- PWM -------------------
const uint32_t freqPWM = 10000;
const uint8_t resolucaoPWM = 8;
const int pwmManualMin = 0;
const int pwmManualMax = 255;
bool pwmAttached = false;

// ------------------- SENSOR -------------------
const uint8_t sensorSdaPin = 21;
const uint8_t sensorSclPin = 22;

// ------------------- SCHEDULER -------------------
const unsigned long BASE_TICK_US = 2000UL;
const uint32_t HALL_DIV = 1U;
const uint32_t EST_DIV = 1U;
const uint32_t CTRL_DIV = 1U;
const uint32_t LOG_DIV = 5U;
const uint32_t MAX_CATCHUP_TICKS = 8U;

unsigned long lastBaseTickUs = 0;
uint32_t tickCounter = 0;

// ------------------- ESTADO -------------------
int pwmAtual = 0;
float raw_mm = NAN;
float actual_mm = NAN;
float ultimoRawValido = 0.0f;
float ultimoActualValido = 0.0f;
bool temUltimoRawValido = false;
bool temUltimoActualValido = false;
uint32_t hall_cima = 0;
uint32_t hall_cima_filtrado = 0;
uint32_t hall_baixo = 0;
uint32_t hall_baixo_filtrado = 0;
uint32_t hallMilliVolts = 0;          // compatibilidade: equivalente a hall_baixo
uint32_t hallMilliVoltsFiltrado = 0; // compatibilidade: equivalente a hall_baixo_filtrado

bool syncPulseActive = false;
unsigned long syncPulseEndUs = 0UL;
const unsigned long SYNC_PULSE_US = 200000UL;

float dtEstSeconds = 0.002f;
float dtCtrlSeconds = 0.002f;
unsigned long dtHallUs = 2000UL;
unsigned long dtEstUs = 2000UL;
unsigned long dtCtrlUs = 2000UL;

unsigned long ultimoHallTaskUs = 0;
unsigned long ultimoEstTaskUs = 0;
unsigned long ultimoCtrlTaskUs = 0;

LeituraSensorVL53L0XData sensorData;
SaidasIdentificacao saidaIdent;
DadosIdentificacao dadosIdent;

// ------------------- CONTROLE PD OPCIONAL -------------------
bool controleLigado = false;
bool controlePrimed = false;
float gainCtrl = 0.0f;
float zcCtrl = 0.0f;
float refCtrlMm = 0.0f;
float baseCtrl = 0.0f;
float floorCtrl = 0.0f;
float ceilCtrl = 255.0f;
float erroAnteriorDb = 0.0f;
const float deadbandMm = 0.5f;
const float CONTROL_SIGN = +1.0f;

const float SCALE_LOW = -5.0f;
const float SCALE_HIGH = 30.0f;

// ------------------- SERIAL -------------------
static const size_t CMD_BUF_LEN = 96;
char cmdBuffer[CMD_BUF_LEN];
size_t cmdLen = 0;
unsigned long ultimoRxMs = 0;
const unsigned long serialFlushIdleMs = 35UL;

// ============================================================
// HELPERS
// ============================================================
float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

bool finitef_local(float x) {
  return isfinite(x);
}

const char* textoOuNull(const char* s) {
  return (s && s[0] != '\0') ? s : "NULL";
}

float aplicarCalibracaoMm(float leituraFiltradaMm) {
  static const float GANHO_CAL = 2.11f;
  static const float POLARIZACAO_CAL = -103.0f;
  return GANHO_CAL * leituraFiltradaMm + POLARIZACAO_CAL;
}

float normalizarCurva(float leituraFiltrada) {
  static const float referencia = 72.0f;
  return referencia - leituraFiltrada;
}

float aplicarBandaMorta(float erro) {
  if (fabsf(erro) <= deadbandMm) {
    return 0.0f;
  }
  return erro;
}

float valorSeguro(float valor, float ultimoValido, bool temUltimoValido) {
  if (finitef_local(valor)) return valor;
  if (temUltimoValido) return ultimoValido;
  return 0.0f;
}

unsigned long atualizarDeltaUs(unsigned long &ultimoUs, unsigned long agoraUs, unsigned long fallbackUs) {
  unsigned long deltaUs = fallbackUs;
  if (ultimoUs != 0UL) {
    deltaUs = agoraUs - ultimoUs;
  }
  ultimoUs = agoraUs;
  if (deltaUs == 0UL) deltaUs = fallbackUs;
  return deltaUs;
}

void aplicarPWM(int valor) {
  if (valor < pwmManualMin) valor = pwmManualMin;
  if (valor > pwmManualMax) valor = pwmManualMax;
  pwmAtual = valor;

  if (pwmAttached) {
    ledcWrite(pinPWM, (uint32_t)pwmAtual);
  }
}

void iniciarPulsoSync(unsigned long nowUs) {
  syncPulseActive = true;
  syncPulseEndUs = nowUs + SYNC_PULSE_US;
  digitalWrite(pinSyncLed, HIGH);
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

void desligarControle() {
  controleLigado = false;
  controlePrimed = false;
  erroAnteriorDb = 0.0f;
  baseCtrl = 0.0f;
  floorCtrl = 0.0f;
  ceilCtrl = 255.0f;
  aplicarPWM(0);
}

void atualizarLeituraSensor(float dtSeconds) {
  leituraSensorVL53L0XUpdate(dtSeconds);
  sensorData = leituraSensorVL53L0XGetData();
  raw_mm = sensorData.raw_mm;

  if (finitef_local(sensorData.filtered_mm)) {
    float calibrada = aplicarCalibracaoMm(sensorData.filtered_mm);
    actual_mm = normalizarCurva(calibrada);
  } else {
    actual_mm = NAN;
  }

  if (finitef_local(raw_mm)) {
    ultimoRawValido = raw_mm;
    temUltimoRawValido = true;
  }
  if (finitef_local(actual_mm)) {
    ultimoActualValido = actual_mm;
    temUltimoActualValido = true;
  }
}

void atualizarLeituraHall() {
  uint32_t hallBaixoRaw = analogReadMilliVolts(pinHallBaixoAnalog);
  hall_baixo = hallBaixoRaw;

  static bool hallBaixoFilterInit = false;
  static float hallBaixoFilt = 0.0f;
  const float alphaHall = 0.25f;

  if (!hallBaixoFilterInit) {
    hallBaixoFilt = (float)hallBaixoRaw;
    hallBaixoFilterInit = true;
  } else {
    hallBaixoFilt = alphaHall * (float)hallBaixoRaw + (1.0f - alphaHall) * hallBaixoFilt;
  }

  hall_baixo_filtrado = (uint32_t)lroundf(hallBaixoFilt);

  uint32_t hallCimaRaw = analogReadMilliVolts(pinHallCimaAnalog);
  hall_cima = hallCimaRaw;

  static bool hallCimaFilterInit = false;
  static float hallCimaFilt = 0.0f;

  if (!hallCimaFilterInit) {
    hallCimaFilt = (float)hallCimaRaw;
    hallCimaFilterInit = true;
  } else {
    hallCimaFilt = alphaHall * (float)hallCimaRaw + (1.0f - alphaHall) * hallCimaFilt;
  }

  hall_cima_filtrado = (uint32_t)lroundf(hallCimaFilt);

  hallMilliVolts = hall_baixo;
  hallMilliVoltsFiltrado = hall_baixo_filtrado;
}

EntradasIdentificacao montarEntradasIdent(unsigned long nowUs) {
  EntradasIdentificacao in;
  in.now_us = nowUs;
  in.pwm_atual = pwmAtual;
  in.hall_raw_mv = hallMilliVolts;
  in.hall_filt_mv = hallMilliVoltsFiltrado;
  in.hall_cima_raw_mv = hall_cima;
  in.hall_cima_filt_mv = hall_cima_filtrado;
  in.tof_raw_mm = raw_mm;
  in.tof_actual_mm = actual_mm;
  in.tof_valido = sensorData.measurement_valid;
  in.novo_tof = sensorData.new_measurement;
  in.range_status = sensorData.range_status;
  return in;
}

void iniciarIdentificacao(unsigned long nowUs) {
  desligarControle();
  EntradasIdentificacao in = montarEntradasIdent(nowUs);
  ensaiosIdentificacaoStart(in);
  iniciarPulsoSync(nowUs);
}

void iniciarIdentificacaoParametrizada(unsigned long nowUs, int pwmPulso, uint16_t tempoPulsoMs, uint16_t tempoEntrePulsosMs, uint8_t repeticoes) {
  desligarControle();

  PerfilIdentificacao perfil = ensaiosIdentificacaoGetPerfilAtivo();
  strncpy(perfil.nome, "PULSO_CMD", sizeof(perfil.nome) - 1);
  perfil.nome[sizeof(perfil.nome) - 1] = '\0';
  strncpy(perfil.test_id, "IDENT_PULSO_CMD", sizeof(perfil.test_id) - 1);
  perfil.test_id[sizeof(perfil.test_id) - 1] = '\0';
  perfil.pwm_pulso = (uint8_t)clampf((float)pwmPulso, 0.0f, 255.0f);
  perfil.tempo_pulso_ms = tempoPulsoMs;
  perfil.tempo_entre_pulsos_ms = tempoEntrePulsosMs;
  perfil.repeticoes = repeticoes;
  perfil.tempo_pre_inicio_ms = 0;

  EntradasIdentificacao in = montarEntradasIdent(nowUs);
  ensaiosIdentificacaoStartComPerfil(in, perfil);
  iniciarPulsoSync(nowUs);
}

// ============================================================
// SERIAL
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

void ligarControle(float gain, float zc, float ref, float base, float floorPwm, float ceilPwm) {
  float floorLocal = clampf(floorPwm, (float)pwmManualMin, (float)pwmManualMax);
  float ceilLocal = clampf(ceilPwm, (float)pwmManualMin, (float)pwmManualMax);

  if (floorLocal > ceilLocal) {
    float tmp = floorLocal;
    floorLocal = ceilLocal;
    ceilLocal = tmp;
  }

  gainCtrl = gain;
  zcCtrl = zc;
  refCtrlMm = ref;
  floorCtrl = floorLocal;
  ceilCtrl = ceilLocal;
  baseCtrl = clampf(base, floorCtrl, ceilCtrl);
  controleLigado = true;
  controlePrimed = false;
  erroAnteriorDb = 0.0f;
  aplicarPWM((int)lroundf(baseCtrl));
}

void processarComandoC(const char* comandoBruto) {
  if (!comandoBruto) return;

  char comando[CMD_BUF_LEN];
  strncpy(comando, comandoBruto, CMD_BUF_LEN - 1);
  comando[CMD_BUF_LEN - 1] = '\0';
  trimInPlace(comando);
  if (comando[0] == '\0') return;

  if (equalsIgnoreCase(comando, "INICIAR_IDENT") || equalsIgnoreCase(comando, "INICIAR IDENT")) {
    iniciarIdentificacao(micros());
    return;
  }

  // Comando parametrizado:
  //   PULSO <pwm> <tempo_pulso_ms> <tempo_entre_pulsos_ms> <repeticoes>
  // Exemplo:
  //   PULSO 200 30 1000 10
  {
    char cmdWordPulso[12] = {0};
    int pwmPulso = 0;
    int tempoPulsoMs = 0;
    int tempoEntrePulsosMs = 0;
    int repeticoes = 0;
    int nPulso = sscanf(comando, "%11s %d %d %d %d", cmdWordPulso, &pwmPulso, &tempoPulsoMs, &tempoEntrePulsosMs, &repeticoes);
    if (nPulso == 5 && (equalsIgnoreCase(cmdWordPulso, "PULSO") || equalsIgnoreCase(cmdWordPulso, "PULSE") || equalsIgnoreCase(cmdWordPulso, "IDENT"))) {
      bool ok = true;
      ok = ok && (pwmPulso >= pwmManualMin && pwmPulso <= pwmManualMax);
      ok = ok && (tempoPulsoMs >= 1 && tempoPulsoMs <= 60000);
      ok = ok && (tempoEntrePulsosMs >= 0 && tempoEntrePulsosMs <= 60000);
      ok = ok && (repeticoes >= 1 && repeticoes <= 255);
      if (ok) {
        iniciarIdentificacaoParametrizada(micros(), pwmPulso, (uint16_t)tempoPulsoMs, (uint16_t)tempoEntrePulsosMs, (uint8_t)repeticoes);
      }
      return;
    }
  }

  if (equalsIgnoreCase(comando, "OFF")) {
    ensaiosIdentificacaoStop();
    desligarControle();
    pararPulsoSync();
    return;
  }

  if (equalsIgnoreCase(comando, "R") || equalsIgnoreCase(comando, "RESET")) {
    ensaiosIdentificacaoReset();
    leituraSensorVL53L0XResetEstimator();
    desligarControle();
    pararPulsoSync();
    return;
  }

  float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f, e = 0.0f, f = 0.0f;
  char cmdWord[8] = {0};
  int n = sscanf(comando, "%7s %f %f %f %f %f %f", cmdWord, &a, &b, &c, &d, &e, &f);
  if (n == 7 && equalsIgnoreCase(cmdWord, "ON")) {
    ensaiosIdentificacaoStop();
    bool baseOk = (d >= pwmManualMin && d <= pwmManualMax);
    bool floorOk = (e >= pwmManualMin && e <= pwmManualMax);
    bool ceilOk = (f >= pwmManualMin && f <= pwmManualMax);
    if (baseOk && floorOk && ceilOk) {
      ligarControle(a, b, c, d, e, f);
    }
    return;
  }

  if (isUnsignedIntString(comando)) {
    long valor = strtol(comando, nullptr, 10);
    if (valor >= pwmManualMin && valor <= pwmManualMax) {
      ensaiosIdentificacaoStop();
      controleLigado = false;
      controlePrimed = false;
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

  if (cmdLen > 0 && (millis() - ultimoRxMs) >= serialFlushIdleMs) {
    flushSerialCommandBuffer();
  }
}

// ============================================================
// TELEMETRIA
// ============================================================
void printCsvHeader() {
  Serial.println(
    "LOG_FORMAT:TEST_ID:<texto>,TEST_PARAM:<texto>,T_US:<us>,"
    "HALL_INF_MV:<mV>,HALL_INF_FILT_MV:<mV>,HALL_SUP_MV:<mV>,HALL_SUP_FILT_MV:<mV>,"
    "PWM:<0-255>,RAW:<mm>,ACTUAL:<mm>,REF:<mm>,ERRO:<mm>,"
    "TOF_VALID:<0|1>,NEW_TOF:<0|1>,RANGE_STATUS:<int>,SIGNAL_MCPS:<float>,AMBIENT_MCPS:<float>,SPAD:<float>,"
    "VEL_MM_S:<float>,MEAS_RATE_HZ:<float>,MEAS_AGE_US:<us>,DT_HALL_US:<us>,DT_TOF_US:<us>,DT_EST_US:<us>,DT_CTRL_US:<us>,"
    "IDENT_ACTIVE:<0|1>,IDENT_STATE:<texto>,IDENT_TYPE:<texto>,IDENT_REP:<n>,IDENT_REP_MAX:<n>,IDENT_FASE:<n>,"
    "LIM_PWM:<int>,LIM_HALL_MV:<mV>,LIM_ACTUAL_MM:<mm>,LIM_RAW_MM:<mm>,DELTA_LIM_MM:<mm>,"
    "TEST_PWM:<int>,TOF_BASE_MM:<mm>,PULSO_HALL_PICO_MV:<mV>,PULSO_ACTUAL_PICO_MM:<mm>,DELTA_PULSO_MM:<mm>,IDENT_MOTIVO:<texto>,"
    "SYNC_PULSE:<0|1>,LIM_HALL_CIMA_MV:<mV>,PULSO_HALL_CIMA_PICO_MV:<mV>,"
    "PULSO_PWM:<int>,PULSO_T_MS:<ms>,PULSO_ENTRE_MS:<ms>,PULSO_PRE_MS:<ms>,"
    "HALL_BASE_MV:<mV>,HALL_DELTA_PICO_MV:<mV>,HALL_CIMA_BASE_MV:<mV>,HALL_CIMA_DELTA_PICO_MV:<mV>,"
    "SCALE_LOW:<float>,SCALE_HIGH:<float>"
  );
}

void printUnifiedTelemetry(unsigned long nowUs) {
  dadosIdent = ensaiosIdentificacaoGetDados();

  float rawLog = valorSeguro(raw_mm, ultimoRawValido, temUltimoRawValido);
  float actualLog = valorSeguro(actual_mm, ultimoActualValido, temUltimoActualValido);
  float refLog = refCtrlMm;
  float erroLog = actualLog - refLog;

  unsigned long dtTofUs = (unsigned long)lroundf(sensorData.measurement_dt_s * 1.0e6f);
  unsigned long measAgeUs = (unsigned long)lroundf(sensorData.measurement_age_s * 1.0e6f);
  const char* testIdLog = (dadosIdent.test_id[0] != '\0') ? dadosIdent.test_id : TEST_ID_FALLBACK;

  // Serial.print("TEST_ID:");
  // Serial.print(textoOuNull(testIdLog));
  // Serial.print(",TEST_PARAM:");
  // Serial.print(textoOuNull(dadosIdent.test_param));
  Serial.print(",T_US:");
  Serial.print(nowUs);
  Serial.print(",HALL_INF_MV:");
  Serial.print((int)hallMilliVolts);
  // Serial.print(",HALL_INF_FILT_MV:");
  // Serial.print((int)hallMilliVoltsFiltrado);
  Serial.print(",HALL_SUP_MV:");
  Serial.print((int)hall_cima);
  // Serial.print(",HALL_SUP_FILT_MV:");
  // Serial.print((int)hall_cima_filtrado);
  Serial.print(",PWM:");
  Serial.print(pwmAtual);
  // Serial.print(",RAW:");
  // Serial.print(rawLog, 3);
  // Serial.print(",ACTUAL:");
  // Serial.print(actualLog, 3);
  // Serial.print(",REF:");
  // Serial.print(refLog, 3);
  // Serial.print(",ERRO:");
  // Serial.print(erroLog, 3);
  // Serial.print(",TOF_VALID:");
  // Serial.print(sensorData.measurement_valid ? 1 : 0);
  // Serial.print(",NEW_TOF:");
  // Serial.print(sensorData.new_measurement ? 1 : 0);
  // Serial.print(",RANGE_STATUS:");
  // Serial.print(sensorData.range_status);
  // Serial.print(",SIGNAL_MCPS:");
  // Serial.print(sensorData.signal_mcps, 4);
  // Serial.print(",AMBIENT_MCPS:");
  // Serial.print(sensorData.ambient_mcps, 4);
  // Serial.print(",SPAD:");
  // Serial.print(sensorData.spad_eff, 4);
  // Serial.print(",VEL_MM_S:");
  // Serial.print(sensorData.velocity_mm_s, 4);
  // Serial.print(",MEAS_RATE_HZ:");
  // Serial.print(sensorData.measurement_rate_hz, 4);
  // Serial.print(",MEAS_AGE_US:");
  // Serial.print(measAgeUs);
  // Serial.print(",DT_HALL_US:");
  // Serial.print(dtHallUs);
  // Serial.print(",DT_TOF_US:");
  // Serial.print(dtTofUs);
  // Serial.print(",DT_EST_US:");
  // Serial.print(dtEstUs);
  // Serial.print(",DT_CTRL_US:");
  // Serial.print(dtCtrlUs);
  // Serial.print(",IDENT_ACTIVE:");
  // Serial.print(dadosIdent.ativo ? 1 : 0);
  // Serial.print(",IDENT_STATE:");
  // Serial.print(textoOuNull(ensaiosIdentificacaoNomeEstado(dadosIdent.estado)));
  // Serial.print(",IDENT_TYPE:");
  // Serial.print(textoOuNull(ensaiosIdentificacaoNomeTipo(dadosIdent.tipo_sinal)));
  // Serial.print(",IDENT_REP:");
  // Serial.print(dadosIdent.repeticao_atual);
  // Serial.print(",IDENT_REP_MAX:");
  // Serial.print(dadosIdent.repeticoes_totais);
  // Serial.print(",IDENT_FASE:");
  // Serial.print(dadosIdent.fase_execucao);
  // Serial.print(",LIM_PWM:");
  // Serial.print(dadosIdent.pwm_limiar);
  // Serial.print(",LIM_HALL_MV:");
  // Serial.print(dadosIdent.hall_limiar_mv);
  // Serial.print(",LIM_ACTUAL_MM:");
  // if (finitef_local(dadosIdent.lim_actual_mm)) Serial.print(dadosIdent.lim_actual_mm, 3); else Serial.print("NULL");
  // Serial.print(",LIM_RAW_MM:");
  // if (finitef_local(dadosIdent.lim_raw_mm)) Serial.print(dadosIdent.lim_raw_mm, 3); else Serial.print("NULL");
  // Serial.print(",DELTA_LIM_MM:");
  // if (finitef_local(dadosIdent.delta_lim_mm)) Serial.print(dadosIdent.delta_lim_mm, 3); else Serial.print("NULL");
  // Serial.print(",TEST_PWM:");
  // Serial.print(dadosIdent.pwm_teste);
  // Serial.print(",TOF_BASE_MM:");
  // if (finitef_local(dadosIdent.tof_base_mm)) Serial.print(dadosIdent.tof_base_mm, 3); else Serial.print("NULL");
  // Serial.print(",PULSO_HALL_PICO_MV:");
  // Serial.print(dadosIdent.hall_pulso_pico_mv);
  // Serial.print(",PULSO_ACTUAL_PICO_MM:");
  // if (finitef_local(dadosIdent.pulso_pico_actual_mm)) Serial.print(dadosIdent.pulso_pico_actual_mm, 3); else Serial.print("NULL");
  // Serial.print(",DELTA_PULSO_MM:");
  // if (finitef_local(dadosIdent.delta_pulso_mm)) Serial.print(dadosIdent.delta_pulso_mm, 3); else Serial.print("NULL");
  // Serial.print(",IDENT_MOTIVO:");
  // Serial.print(textoOuNull(dadosIdent.motivo));
  Serial.print(",SYNC_PULSE:");
  Serial.print(syncPulseActive ? 1 : 0);
  // Serial.print(",LIM_HALL_CIMA_MV:");
  // Serial.print(dadosIdent.hall_cima_limiar_mv);
  // Serial.print(",PULSO_HALL_CIMA_PICO_MV:");
  // Serial.print(dadosIdent.hall_cima_pulso_pico_mv);
  // Serial.print(",PULSO_PWM:");
  // Serial.print(dadosIdent.pwm_pulso);
  // Serial.print(",PULSO_T_MS:");
  // Serial.print(dadosIdent.tempo_pulso_ms);
  // Serial.print(",PULSO_ENTRE_MS:");
  // Serial.print(dadosIdent.tempo_entre_pulsos_ms);
  // Serial.print(",PULSO_PRE_MS:");
  // Serial.print(dadosIdent.tempo_pre_inicio_ms);
  // Serial.print(",HALL_BASE_MV:");
  // Serial.print((int)dadosIdent.hall_base_mv);
  // Serial.print(",HALL_DELTA_PICO_MV:");
  // Serial.print((int)dadosIdent.hall_delta_pico_mv);
  // Serial.print(",HALL_CIMA_BASE_MV:");
  // Serial.print((int)dadosIdent.hall_cima_base_mv);
  // Serial.print(",HALL_CIMA_DELTA_PICO_MV:");
  // Serial.print((int)dadosIdent.hall_cima_delta_pico_mv);
  // Serial.print(",SCALE_LOW:");
  // Serial.print(SCALE_LOW, 2);
  // Serial.print(",SCALE_HIGH:");
  // Serial.println(SCALE_HIGH, 2);
}

// ============================================================
// CONTROLE
// ============================================================
void atualizarControlePD(float dtSeconds) {
  if (!controleLigado) {
    return;
  }

  if (!finitef_local(actual_mm)) {
    aplicarPWM(0);
    return;
  }

  float erro = actual_mm - refCtrlMm;
  float erroDb = aplicarBandaMorta(erro);

  if (!controlePrimed) {
    erroAnteriorDb = erroDb;
    controlePrimed = true;
  }

  float derivErro = 0.0f;
  if (dtSeconds > 1e-6f) {
    derivErro = (erroDb - erroAnteriorDb) / dtSeconds;
  }

  float kp = gainCtrl * zcCtrl;
  float kd = gainCtrl;

  float deltaPWM = CONTROL_SIGN * (kp * erroDb + kd * derivErro);

  float pwmControle = baseCtrl + deltaPWM;
  pwmControle = clampf(pwmControle, floorCtrl, ceilCtrl);
  pwmControle = clampf(pwmControle, (float)pwmManualMin, (float)pwmManualMax);

  aplicarPWM((int)lroundf(pwmControle));
  erroAnteriorDb = erroDb;
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  Serial.setTimeout(5);
  delay(300);

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

  leituraSensorVL53L0XSetSensorType(SENSOR_TYPE);
  bool sensorOk = leituraSensorVL53L0XBegin(sensorSdaPin, sensorSclPin);
  if (!sensorOk) {
    while (true) {
      delay(1000);
    }
  }

  ensaiosIdentificacaoBegin();

  sensorData = leituraSensorVL53L0XGetData();
  raw_mm = sensorData.raw_mm;
  actual_mm = sensorData.filtered_mm;

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

    if ((tickCounter % HALL_DIV) == 0U) {
      unsigned long taskUs = micros();
      dtHallUs = atualizarDeltaUs(ultimoHallTaskUs, taskUs, BASE_TICK_US * HALL_DIV);
      atualizarLeituraHall();
    }

    if ((tickCounter % EST_DIV) == 0U) {
      unsigned long taskUs = micros();
      dtEstUs = atualizarDeltaUs(ultimoEstTaskUs, taskUs, BASE_TICK_US * EST_DIV);
      dtEstSeconds = clampf((float)dtEstUs * 1.0e-6f, 0.001f, 0.050f);
      atualizarLeituraSensor(dtEstSeconds);
    }

    if ((tickCounter % CTRL_DIV) == 0U) {
      unsigned long taskUs = micros();
      dtCtrlUs = atualizarDeltaUs(ultimoCtrlTaskUs, taskUs, BASE_TICK_US * CTRL_DIV);
      dtCtrlSeconds = clampf((float)dtCtrlUs * 1.0e-6f, 0.001f, 0.050f);

      EntradasIdentificacao in = montarEntradasIdent(taskUs);
      ensaiosIdentificacaoUpdate(in, saidaIdent);

      if (saidaIdent.ativo || saidaIdent.finalizado || saidaIdent.abortado) {
        aplicarPWM(saidaIdent.pwm_cmd);
        controleLigado = false;
        controlePrimed = false;
      } else {
        atualizarControlePD(dtCtrlSeconds);
      }
    }

    if ((tickCounter % LOG_DIV) == 0U) {
      printUnifiedTelemetry(lastBaseTickUs);
    }

    ticksProcessed++;
    nowUs = micros();
  }

  if (ticksProcessed >= MAX_CATCHUP_TICKS) {
    lastBaseTickUs = nowUs;
  }
}
