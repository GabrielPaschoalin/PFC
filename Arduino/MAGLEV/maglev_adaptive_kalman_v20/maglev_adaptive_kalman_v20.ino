#include <Arduino.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "src/leitura_sensor_vl53l0x.h"

// ============================================================
// MAGLEV - KALMAN ADAPTATIVO V20
// ------------------------------------------------------------
// Baseada na V17, com alteracoes pedidas:
// - comando de controle agora: ON GAIN ZC REF BASE FLOOR CEIL
// - BASE define o PWM nominal fixo
// - FLOOR e CEIL sao obrigatorios e limitam o comando final
// - o controle gera um trim ao redor da base:
//       PWM = BASE + deltaPWM
// - o sinal usado no controle nao e arredondado
//   (mantem variacao continua para suavizar a resposta)
//
// Log unico:
// PWM, RAW, ACTUAL, REF, ERRO, HALL_MV, SCALE_LOW, SCALE_HIGH
// ============================================================

// ------------------- CONFIGURACAO DE PINOS -------------------
const int pinPWM = 27;  // PWM no IN1 do L298N
const int pinENA = 14;  // ENA fixo em HIGH
const int pinIN2 = 26;  // IN2 fixo em LOW
const int pinHallAnalog = 34;  // AO do KY-024 em ADC1 do ESP32

// ------------------- PWM -------------------
const uint32_t freqPWM = 5000;
const uint8_t resolucaoPWM = 8;
const int pwmManualMin = 0;
const int pwmManualMax = 255;
const int pwmControlMin = 0;
const int pwmControlMax = 255;
bool pwmAttached = false;

// ------------------- SENSOR / AMOSTRAGEM -------------------
const uint8_t sensorSdaPin = 21;
const uint8_t sensorSclPin = 22;
const unsigned long intervaloLeituraMs = 60UL;

unsigned long ultimoLoopMs = 0;
unsigned long ultimoLoopUs = 0;
float dtLoopSeconds = 0.060f;

// ------------------- ESTADO GERAL -------------------
int pwmAtual = 0;
float raw_mm = NAN;
float actual_mm = NAN;
float ultimoRawValido = 0.0f;
float ultimoActualValido = 0.0f;
bool temUltimoRawValido = false;
bool temUltimoActualValido = false;
uint32_t hallMilliVolts = 0;

// ------------------- CONTROLE PD COM TRIM EM TORNO DA BASE -------------------
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

// Se a acao ficar invertida na bancada, troque para -1.0f.
const float CONTROL_SIGN = +1.0f;

// Guias fixos para o plotter
const float SCALE_LOW = -5.0f;
const float SCALE_HIGH = 30.0f;

// ------------------- SERIAL NAO BLOQUEANTE -------------------
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

bool finitef_local(float x) {
  return isfinite(x);
}

float valorSeguro(float valor, float ultimoValido, bool temUltimoValido) {
  if (finitef_local(valor)) return valor;
  if (temUltimoValido) return ultimoValido;
  return 0.0f;
}

void aplicarPWM(int valor) {
  if (valor < pwmManualMin) valor = pwmManualMin;
  if (valor > pwmManualMax) valor = pwmManualMax;
  pwmAtual = valor;

  if (pwmAttached) {
    ledcWrite(pinPWM, (uint32_t)pwmAtual);
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

void atualizarLeituraSensor() {
  leituraSensorVL53L0XUpdate(dtLoopSeconds);
  raw_mm = leituraSensorVL53L0XGetRawMm();

  float filtrada = leituraSensorVL53L0XGetFilteredMm();
  if (finitef_local(filtrada)) {
    actual_mm = aplicarCalibracaoMm(filtrada);
    actual_mm = normalizarCurva(actual_mm);
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
  hallMilliVolts = analogReadMilliVolts(pinHallAnalog);
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
  float floorLocal = clampf(floorPwm, (float)pwmControlMin, (float)pwmControlMax);
  float ceilLocal = clampf(ceilPwm, (float)pwmControlMin, (float)pwmControlMax);

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

  if (equalsIgnoreCase(comando, "OFF")) {
    desligarControle();
    return;
  }

  if (equalsIgnoreCase(comando, "R") || equalsIgnoreCase(comando, "RESET")) {
    leituraSensorVL53L0XResetEstimator();
    desligarControle();
    return;
  }

  float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f, e = 0.0f, f = 0.0f;
  char cmdWord[8] = {0};
  int n = sscanf(comando, "%7s %f %f %f %f %f %f", cmdWord, &a, &b, &c, &d, &e, &f);
  if (n == 7 && equalsIgnoreCase(cmdWord, "ON")) {
    bool baseOk = (d >= pwmControlMin && d <= pwmControlMax);
    bool floorOk = (e >= pwmControlMin && e <= pwmControlMax);
    bool ceilOk = (f >= pwmControlMin && f <= pwmControlMax);
    if (baseOk && floorOk && ceilOk) {
      ligarControle(a, b, c, d, e, f);
    }
    return;
  }

  if (isUnsignedIntString(comando)) {
    long valor = strtol(comando, nullptr, 10);
    if (valor >= pwmManualMin && valor <= pwmManualMax) {
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
// TELEMETRIA UNICA
// ============================================================
void printUnifiedTelemetry() {
  float rawLog = valorSeguro(raw_mm, ultimoRawValido, temUltimoRawValido);
  float actualLog = valorSeguro(actual_mm, ultimoActualValido, temUltimoActualValido);
  float refLog = refCtrlMm;
  float erroLog = actualLog - refLog;

  Serial.print("PWM:");
  Serial.print(pwmAtual);
  Serial.print(",RAW:");
  Serial.print(rawLog, 2);
  Serial.print(",ACTUAL:");
  Serial.print(actualLog, 2);
  Serial.print(",REF:");
  Serial.print(refLog, 2);
  Serial.print(",ERRO:");
  Serial.print(erroLog, 2);
  Serial.print(",HALL_MV:");
  Serial.print(hallMilliVolts);
  Serial.print(",SCALE_LOW:");
  Serial.print(SCALE_LOW, 2);
  Serial.print(",SCALE_HIGH:");
  Serial.println(SCALE_HIGH, 2);
}

// ============================================================
// CONTROLE
// ============================================================
void atualizarControlePD() {
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
  if (dtLoopSeconds > 1e-6f) {
    derivErro = (erroDb - erroAnteriorDb) / dtLoopSeconds;
  }

  float kp = gainCtrl * zcCtrl;
  float kd = gainCtrl;

  float deltaPWM = CONTROL_SIGN * (kp * erroDb + kd * derivErro);

  float pwmControle = baseCtrl + deltaPWM;
  pwmControle = clampf(pwmControle, floorCtrl, ceilCtrl);
  pwmControle = clampf(pwmControle, (float)pwmControlMin, (float)pwmControlMax);

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

  pinMode(pinHallAnalog, INPUT);
#if defined(ARDUINO_ARCH_ESP32)
  analogSetPinAttenuation(pinHallAnalog, ADC_11db);
#endif

  pwmAttached = ledcAttach(pinPWM, freqPWM, resolucaoPWM);
  if (!pwmAttached) {
    while (true) {
      delay(1000);
    }
  }

  aplicarPWM(0);

  bool sensorOk = leituraSensorVL53L0XBegin(sensorSdaPin, sensorSclPin);
  if (!sensorOk) {
    while (true) {
      delay(1000);
    }
  }

  raw_mm = leituraSensorVL53L0XGetRawMm();
  actual_mm = leituraSensorVL53L0XGetFilteredMm();
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  pollSerialCommands();

  unsigned long agoraMs = millis();
  if (agoraMs - ultimoLoopMs < intervaloLeituraMs) {
    return;
  }
  ultimoLoopMs = agoraMs;

  unsigned long agoraUs = micros();
  if (ultimoLoopUs == 0) {
    ultimoLoopUs = agoraUs;
  }

  float dtMedido = (agoraUs - ultimoLoopUs) * 1e-6f;
  ultimoLoopUs = agoraUs;
  dtLoopSeconds = clampf(dtMedido, 0.02f, 0.20f);

  atualizarLeituraSensor();
  atualizarLeituraHall();
  atualizarControlePD();
  printUnifiedTelemetry();
}
