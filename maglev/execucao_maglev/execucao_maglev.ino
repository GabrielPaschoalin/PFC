#include <Arduino.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// MAGLEV - V1 - ESTIMADOR DE POSICAO POR HALL + PWM MANUAL
// ------------------------------------------------------------
// Modelo usado:
//
// x_mm = X_REF_MM
//      + K_HINF * (HALL_INF_MV - HINF_REF_MV)
//      + K_HSUP * (HALL_SUP_MV - HSUP_REF_MV)
//      + K_PWM  * (PWM - PWM_REF)
//
// Comandos no Monitor Serial:
//   0..255   -> aplica PWM manual na bobina
//   OFF      -> zera PWM
//   RESET/R  -> zera PWM
//
// Saida CSV:
//   T_MS,HALL_INF_MV,HALL_SUP_MV,PWM,X_MM
// ============================================================


// ------------------- SERIAL -------------------
const uint32_t SERIAL_BAUD = 2000000;


// ------------------- PINOS -------------------
// Mantidos conforme V35
const int pinPWM = 27;             // PWM no IN1 do L298N
const int pinENA = 14;             // ENA fixo em HIGH
const int pinIN2 = 26;             // IN2 fixo em LOW

const int pinHallCimaAnalog = 36;  // Hall superior
const int pinHallBaixoAnalog = 34; // Hall inferior


// ------------------- PWM -------------------
const uint32_t freqPWM = 10000;
const uint8_t resolucaoPWM = 8;

const int pwmManualMin = 0;
const int pwmManualMax = 255;

bool pwmAttached = false;
int pwmAtual = 0;


// ------------------- LEITURA HALL -------------------
const int N_AMOSTRAS_HALL = 8;


// ------------------- MODELO DE CALIBRACAO -------------------
// Modelo 1 global, usando dados com PWM=0 e PWM>0.
// Depois podemos trocar esses coeficientes pela regressao final.
const float X_REF_MM    = 11.6949f;

const float HINF_REF_MV = 1479.6636f;
const float HSUP_REF_MV = 1630.8482f;
const float PWM_REF     = 29.1268f;

const float K_HINF      = 0.00522737f;  // mm/mV
const float K_HSUP      = 0.00297976f;  // mm/mV
const float K_PWM       = 0.00921397f;  // mm/unidade de PWM


// ------------------- TELEMETRIA -------------------
const unsigned long PRINT_PERIOD_MS = 50;
unsigned long lastPrintMs = 0;


// ------------------- SERIAL RX -------------------
const size_t CMD_BUF_LEN = 64;
char cmdBuffer[CMD_BUF_LEN];
size_t cmdLen = 0;
unsigned long ultimoRxMs = 0;
const unsigned long serialFlushIdleMs = 8;


// ============================================================
// FUNCOES AUXILIARES
// ============================================================

int clampInt(int x, int lo, int hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}


void trimInPlace(char* s) {
  if (!s) return;

  size_t len = strlen(s);
  size_t start = 0;

  while (start < len && isspace((unsigned char)s[start])) {
    start++;
  }

  size_t end = len;
  while (end > start && isspace((unsigned char)s[end - 1])) {
    end--;
  }

  if (start > 0) {
    memmove(s, s + start, end - start);
  }

  s[end - start] = '\0';
}


bool equalsIgnoreCase(const char* a, const char* b) {
  if (!a || !b) return false;

  while (*a && *b) {
    if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
      return false;
    }
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


// ============================================================
// PWM
// ============================================================

void aplicarPWM(int valor) {
  pwmAtual = clampInt(valor, pwmManualMin, pwmManualMax);

  if (pwmAttached) {
    ledcWrite(pinPWM, (uint32_t)pwmAtual);
  }
}


void pararTudo() {
  aplicarPWM(0);
}


// ============================================================
// LEITURA DOS SENSORES
// ============================================================

uint32_t lerMediaMilliVolts(int pinAnalogico, int nAmostras) {
  uint32_t soma = 0;

  for (int i = 0; i < nAmostras; i++) {
    soma += analogReadMilliVolts(pinAnalogico);
    delayMicroseconds(250);
  }

  return soma / (uint32_t)nAmostras;
}


void lerHalls(uint32_t &hallInfMv, uint32_t &hallSupMv) {
  hallInfMv = lerMediaMilliVolts(pinHallBaixoAnalog, N_AMOSTRAS_HALL);
  hallSupMv = lerMediaMilliVolts(pinHallCimaAnalog, N_AMOSTRAS_HALL);
}


// ============================================================
// ESTIMADOR DE POSICAO
// ============================================================

float estimarPosicaoMm(float hallInfMv, float hallSupMv, float pwm) {
  float xMm = X_REF_MM
            + K_HINF * (hallInfMv - HINF_REF_MV)
            + K_HSUP * (hallSupMv - HSUP_REF_MV)
            + K_PWM  * (pwm       - PWM_REF);

  return xMm;
}


// ============================================================
// SERIAL - COMANDOS
// ============================================================

void processarComandoC(const char* comandoBruto) {
  if (!comandoBruto) return;

  char comando[CMD_BUF_LEN];
  strncpy(comando, comandoBruto, CMD_BUF_LEN - 1);
  comando[CMD_BUF_LEN - 1] = '\0';

  trimInPlace(comando);

  if (comando[0] == '\0') return;

  if (equalsIgnoreCase(comando, "OFF") ||
      equalsIgnoreCase(comando, "RESET") ||
      equalsIgnoreCase(comando, "R")) {
    pararTudo();
    Serial.println("# PWM zerado.");
    return;
  }

  if (isUnsignedIntString(comando)) {
    long valor = strtol(comando, nullptr, 10);

    if (valor >= pwmManualMin && valor <= pwmManualMax) {
      aplicarPWM((int)valor);

      Serial.print("# PWM aplicado: ");
      Serial.println(pwmAtual);
    } else {
      Serial.println("# Erro: digite um PWM entre 0 e 255.");
    }

    return;
  }

  Serial.println("# Comando invalido. Use 0..255, OFF, RESET ou R.");
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

  // Permite comando mesmo sem line ending no Monitor Serial.
  if (cmdLen > 0 && (millis() - ultimoRxMs) >= serialFlushIdleMs) {
    flushSerialCommandBuffer();
  }
}


// ============================================================
// TELEMETRIA
// ============================================================

void printCsvHeader() {
  Serial.println("T_MS,HALL_INF_MV,HALL_SUP_MV,PWM,X_MM");
}


void printTelemetry(unsigned long nowMs) {
  uint32_t hallInfMv = 0;
  uint32_t hallSupMv = 0;

  lerHalls(hallInfMv, hallSupMv);

  float xMm = estimarPosicaoMm((float)hallInfMv, (float)hallSupMv, (float)pwmAtual);

  Serial.print(nowMs);
  Serial.print(",");
  Serial.print(hallInfMv);
  Serial.print(",");
  Serial.print(hallSupMv);
  Serial.print(",");
  Serial.print(pwmAtual);
  Serial.print(",");
  Serial.println(xMm, 4);
}


// ============================================================
// SETUP
// ============================================================

void setup() {
#if defined(ARDUINO_ARCH_ESP32)
  Serial.setRxBufferSize(512);
  Serial.setTxBufferSize(2048);
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

#if defined(ARDUINO_ARCH_ESP32)
  analogSetPinAttenuation(pinHallBaixoAnalog, ADC_11db);
  analogSetPinAttenuation(pinHallCimaAnalog, ADC_11db);
#endif

  pwmAttached = ledcAttach(pinPWM, freqPWM, resolucaoPWM);

  if (!pwmAttached) {
    Serial.println("# ERRO: falha ao configurar PWM.");
    while (true) {
      delay(1000);
    }
  }

  aplicarPWM(0);

  delay(500);

  Serial.println("# MAGLEV V1 - Estimador de posicao por Hall + PWM manual");
  Serial.println("# Digite um valor de PWM entre 0 e 255.");
  Serial.println("# Digite OFF, RESET ou R para zerar.");
  printCsvHeader();
}


// ============================================================
// LOOP
// ============================================================

void loop() {
  pollSerialCommands();

  unsigned long nowMs = millis();

  if ((nowMs - lastPrintMs) >= PRINT_PERIOD_MS) {
    lastPrintMs = nowMs;
    printTelemetry(nowMs);
  }
}