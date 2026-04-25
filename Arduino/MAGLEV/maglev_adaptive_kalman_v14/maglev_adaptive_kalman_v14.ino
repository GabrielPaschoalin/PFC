#include <Arduino.h>
#include <math.h>
#include "src/leitura_sensor_vl53l0x.h"

// ============================================================
// MAGLEV - KALMAN ADAPTATIVO V13
// ------------------------------------------------------------
// Baseada na V11, preservando:
// - estrutura modular (.ino + modulo do sensor)
// - calibracao / normalizacao da posicao
// - PWM manual, serial e telemetria
//
// Adicionado nesta V13:
// - comando ON <GAIN> <ZC> <REF>
// - comando OFF
// - controle PD discreto classico sobre o erro
// - banda morta aplicada no controle inteiro
// - saturacao do PWM de controle em 0..150
// - saida nomeada: pwm, atual, ref, erro quando o controle estiver ON
//
// Lei implementada:
//   e[k]    = actual_mm - REF
//   e_db[k] = 0               se |e[k]| <= deadband
//             e[k]            caso contrario
//   de[k]   = (e_db[k] - e_db[k-1]) / dt
//   u[k]    = GAIN * (ZC * e_db[k] + de[k])
//
// onde:
//   Kd = GAIN
//   Kp = GAIN * ZC
// ============================================================

// ------------------- CONFIGURACAO DE PINOS -------------------
const int pinPWM = 27;  // PWM no IN1 do L298N
const int pinENA = 14;  // ENA fixo em HIGH
const int pinIN2 = 26;  // IN2 fixo em LOW

// ------------------- PWM -------------------
const uint32_t freqPWM = 5000;
const uint8_t resolucaoPWM = 8;
const int pwmManualMin = 0;
const int pwmManualMax = 255;
const int pwmControlMin = 0;
const int pwmControlMax = 150;
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
bool detailedOutput = false;
float raw_mm = NAN;
float actual_mm = NAN;

// ------------------- CONTROLE PD -------------------
bool controleLigado = false;
bool controlePrimed = false;
float gainCtrl = 0.0f;
float zcCtrl = 0.0f;
float refCtrlMm = 0.0f;
float erroAnteriorDb = 0.0f;
const float deadbandMm = 0.5f;

bool controlCsvHeaderPrinted = false;

// Se a acao ficar invertida na bancada, troque para -1.0f.
const float CONTROL_SIGN = +1.0f;

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
  return roundf(referencia - leituraFiltrada);
}

float aplicarBandaMorta(float erro) {
  if (fabsf(erro) <= deadbandMm) {
    return 0.0f;
  }
  return erro;
}

void aplicarPWM(int valor) {
  valor = constrain(valor, pwmManualMin, pwmManualMax);
  pwmAtual = valor;

  if (pwmAttached) {
    ledcWrite(pinPWM, (uint32_t)pwmAtual);
  }
}

void desligarControle() {
  controleLigado = false;
  controlePrimed = false;
  erroAnteriorDb = 0.0f;
  controlCsvHeaderPrinted = false;
  aplicarPWM(0);
}

void atualizarLeituraSensor() {
  leituraSensorVL53L0XUpdate(dtLoopSeconds);
  raw_mm = leituraSensorVL53L0XGetRawMm();
  actual_mm = aplicarCalibracaoMm(leituraSensorVL53L0XGetFilteredMm());
  actual_mm = normalizarCurva(actual_mm);
}

// ============================================================
// SERIAL
// ============================================================


void printHelp() {
  Serial.println("Comandos:");
  Serial.println("  ON <GAIN> <ZC> <REF>  -> liga o controle PD");
  Serial.println("  OFF                   -> desliga o controle");
  Serial.println("  reset                 -> reseta o estimador e desliga o controle");
  Serial.println("  status                -> imprime o estado atual");
  Serial.println("  d=0 / d=1             -> telemetria resumida / detalhada");
  Serial.println("  0..255                -> PWM manual (desliga o controle)");
}

void printStatus() {
  Serial.print("STATUS,pwm:");
  Serial.print(pwmAtual);
  Serial.print(",dt:");
  Serial.print(dtLoopSeconds, 4);
  Serial.print(",sensor_init:");
  Serial.print(leituraSensorVL53L0XIsInitialized() ? 1 : 0);
  Serial.print(",raw_mm:");
  Serial.print(raw_mm, 3);
  Serial.print(",actual_mm:");
  Serial.print(actual_mm, 3);
  Serial.print(",control_on:");
  Serial.print(controleLigado ? 1 : 0);
  Serial.print(",gain:");
  Serial.print(gainCtrl, 4);
  Serial.print(",zc:");
  Serial.print(zcCtrl, 4);
  Serial.print(",ref:");
  Serial.print(refCtrlMm, 3);
  Serial.print(",deadband:");
  Serial.print(deadbandMm, 3);
  Serial.print(",output_mode:");
  Serial.println(detailedOutput ? 1 : 0);
}

void ligarControle(float gain, float zc, float ref) {
  gainCtrl = gain;
  zcCtrl = zc;
  refCtrlMm = ref;
  controleLigado = true;
  controlePrimed = false;
  erroAnteriorDb = 0.0f;
  controlCsvHeaderPrinted = false;

  Serial.print("CTRL,ON,gain:");
  Serial.print(gainCtrl, 4);
  Serial.print(",zc:");
  Serial.print(zcCtrl, 4);
  Serial.print(",ref:");
  Serial.println(refCtrlMm, 3);
}

void processarComando(String comando) {
  String original = comando;
  original.trim();
  if (original.length() == 0) return;

  String lower = original;
  lower.toLowerCase();

  if (lower == "off") {
    desligarControle();
    Serial.println("CTRL,OFF");
    return;
  }

  if (lower == "r" || lower == "reset") {
    leituraSensorVL53L0XResetEstimator();
    desligarControle();
    Serial.println("CTRL,RESET");
    return;
  }

  if (lower == "status") {
    printStatus();
    return;
  }

  if (lower == "?" || lower == "help") {
    printHelp();
    return;
  }

  if (lower.startsWith("d=")) {
    detailedOutput = (lower.substring(2).toInt() != 0);
    return;
  }

  float a = 0.0f, b = 0.0f, c = 0.0f;
  if (sscanf(original.c_str(), "ON %f %f %f", &a, &b, &c) == 3 ||
      sscanf(original.c_str(), "on %f %f %f", &a, &b, &c) == 3) {
    ligarControle(a, b, c);
    return;
  }

  bool ehNumero = true;
  for (unsigned int i = 0; i < original.length(); i++) {
    if (!isDigit(original[i])) {
      ehNumero = false;
      break;
    }
  }

  if (ehNumero) {
    int valor = original.toInt();
    if (valor < pwmManualMin || valor > pwmManualMax) {
      Serial.println("Digite um valor entre 0 e 255.");
      return;
    }

    controleLigado = false;
    controlePrimed = false;
    controlCsvHeaderPrinted = false;
    aplicarPWM(valor);
    return;
  }

  Serial.println("Comando invalido. Use: ON <GAIN> <ZC> <REF>, OFF, reset, status, d=0, d=1 ou um numero entre 0 e 255.");
}

// ============================================================
// TELEMETRIA
// ============================================================
void printControlCsvHeaderOnce() {
  if (!controlCsvHeaderPrinted) {
    Serial.println("pwm,atual,ref,erro");
    controlCsvHeaderPrinted = true;
  }
}

void printControlCsvTelemetry() {
  float erro = actual_mm - refCtrlMm;
  printControlCsvHeaderOnce();
  Serial.print("pwm:");
  Serial.print(pwmAtual);
  Serial.print(",atual:");
  Serial.print(actual_mm, 2);
  Serial.print(",ref:");
  Serial.print(refCtrlMm, 2);
  Serial.print(",erro:");
  Serial.println(erro, 2);
}

void printGlobalPlotTelemetry() {
  Serial.print("pwm:");
  Serial.print(pwmAtual);
  Serial.print(",raw:");
  Serial.print(raw_mm, 2);
  Serial.print(",actual:");
  Serial.print(actual_mm, 2);
  Serial.print(",scale70:");
  Serial.print(-30.0f, 2);
  Serial.print(",scale80:");
  Serial.println(30.0f, 2);
}

void printDetailedTelemetry() {
  float erro = actual_mm - refCtrlMm;
  float erroDb = aplicarBandaMorta(erro);
  Serial.print("pwm:");
  Serial.print(pwmAtual);
  Serial.print(",dt:");
  Serial.print(dtLoopSeconds, 4);
  Serial.print(",raw_mm:");
  Serial.print(raw_mm, 3);
  Serial.print(",actual_mm:");
  Serial.print(actual_mm, 3);
  Serial.print(",ref_mm:");
  Serial.print(refCtrlMm, 3);
  Serial.print(",erro_mm:");
  Serial.print(erro, 3);
  Serial.print(",erro_db_mm:");
  Serial.print(erroDb, 3);
  Serial.print(",control_on:");
  Serial.print(controleLigado ? 1 : 0);
  Serial.print(",gain:");
  Serial.print(gainCtrl, 4);
  Serial.print(",zc:");
  Serial.print(zcCtrl, 4);
  Serial.println();
}

// ============================================================
// CONTROLE
// ============================================================
void atualizarControlePD() {
  if (!controleLigado) {
    return;
  }

  if (!isfinite(actual_mm)) {
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

  float pwmControle = CONTROL_SIGN * (kp * erroDb + kd * derivErro);
  pwmControle = clampf(pwmControle, (float)pwmControlMin, (float)pwmControlMax);

  aplicarPWM((int)lroundf(pwmControle));
  erroAnteriorDb = erroDb;
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(pinENA, OUTPUT);
  digitalWrite(pinENA, HIGH);

  pinMode(pinIN2, OUTPUT);
  digitalWrite(pinIN2, LOW);

  pwmAttached = ledcAttach(pinPWM, freqPWM, resolucaoPWM);
  if (!pwmAttached) {
    Serial.println("Erro ao configurar o PWM no ESP32.");
    while (true) {
      delay(1000);
    }
  }

  aplicarPWM(0);

  bool sensorOk = leituraSensorVL53L0XBegin(sensorSdaPin, sensorSclPin);
  if (!sensorOk) {
    Serial.println("Erro ao inicializar o VL53L0X.");
    while (true) {
      delay(1000);
    }
  }

  raw_mm = leituraSensorVL53L0XGetRawMm();
  actual_mm = leituraSensorVL53L0XGetFilteredMm();
  printHelp();
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  if (Serial.available()) {
    String comando = Serial.readStringUntil('\n');
    processarComando(comando);
  }

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
  atualizarControlePD();

  if (detailedOutput) {
    printDetailedTelemetry();
  } else if (controleLigado) {
    printControlCsvTelemetry();
  } else {
    printGlobalPlotTelemetry();
  }
}
