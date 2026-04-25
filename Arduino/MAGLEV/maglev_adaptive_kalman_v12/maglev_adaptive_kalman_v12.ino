#include <Arduino.h>
#include <math.h>
#include "leitura_sensor_vl53l0x.h"

// ============================================================
// MAGLEV - V12
// Controle PD discreto simples sobre o erro
// ------------------------------------------------------------
// Comandos seriais:
//   ON <GAIN> <ZERO_POS> <REF_MM>
//   OFF
//   BASE <PWM>     (opcional, ajusta a pré-carga empírica)
//   R              (reset do estimador e desliga o controle)
//   H              (ajuda)
//
// Saída serial (CSV):
//   PWM,atual,ref
//
// Observações:
// - usa a distância filtrada do módulo VL53L0X
// - derivada sobre o erro de 1 passo
// - banda morta só no termo P
// - saturação do termo D e do PWM final
// ============================================================

// ------------------- HARDWARE -------------------
// Revise estes pinos caso a sua V11 use outros valores.
static const int PIN_PWM      = 27;
static const int PIN_ENABLE   = 14;
static const int PIN_DIRECAO  = 26;

// Se a bancada reagir com sinal invertido, troque para -1.0f
static const float CONTROL_SIGN = +1.0f;

// PWM ESP32 (LEDC)
static const int PWM_CHANNEL        = 0;
static const int PWM_FREQUENCY_HZ   = 1000;
static const int PWM_RESOLUTION_BIT = 8;
static const int PWM_MAX_CODE       = 255;

// ------------------- TEMPOS -------------------
static unsigned long loopPeriodMs   = 60;
static unsigned long logPeriodMs    = 100;
static unsigned long lastStepMs     = 0;
static unsigned long lastStepUs     = 0;
static unsigned long lastLogMs      = 0;

// ------------------- CONTROLE -------------------
static bool  controlEnabled    = false;
static bool  controlPrimed     = false;

static float gainInput         = 240.0f;  // argumento GAIN
static float zeroPos           = 35.0f;   // argumento ZERO_POS
static float refMm             = 17.0f;   // argumento REF_MM

static float basePWM_guess     = 130.0f;  // pré-carga empírica
static float deadbandMm        = 0.5f;    // banda morta só no P
static float dTermLimitPwm     = 15.0f;   // limite do termo D

static int pwmMin              = 0;
static int pwmMax              = 150;
static int currentPwmSent      = 0;

static float rawErrPrev        = NAN;
static bool  csvHeaderPrinted  = false;

// ------------------- HELPERS -------------------
static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static inline bool finitef(float x) {
  return isfinite(x);
}

void applyPWM(int pwm) {
  pwm = constrain(pwm, pwmMin, pwmMax);
  pwm = constrain(pwm, 0, PWM_MAX_CODE);
  currentPwmSent = pwm;
  ledcWrite(PWM_CHANNEL, pwm);
}

void stopControlAndOutput() {
  controlEnabled = false;
  controlPrimed = false;
  rawErrPrev = NAN;
  applyPWM(0);
}

void printHelp() {
  Serial.println();
  Serial.println("Comandos:");
  Serial.println("  ON <GAIN> <ZERO_POS> <REF_MM>  -> liga o controle");
  Serial.println("  OFF                            -> desliga o controle");
  Serial.println("  BASE <PWM>                     -> ajusta basePWM_guess");
  Serial.println("  R                              -> reset do estimador + OFF");
  Serial.println("  H                              -> ajuda");
  Serial.println();
  Serial.println("CSV:");
  Serial.println("  PWM,atual,ref");
  Serial.println();
}

void printCsvHeaderOnce() {
  if (!csvHeaderPrinted) {
    Serial.println("PWM,atual,ref");
    csvHeaderPrinted = true;
  }
}

void logCsv(float atualMm) {
  printCsvHeaderOnce();
  Serial.print(currentPwmSent);
  Serial.print(",");
  if (finitef(atualMm)) Serial.print(atualMm, 2);
  else Serial.print("nan");
  Serial.print(",");
  if (finitef(refMm)) Serial.println(refMm, 2);
  else Serial.println("nan");
}

void armControl(float gain, float zero, float ref) {
  gainInput = gain;
  zeroPos = zero;
  refMm = ref;

  controlEnabled = true;
  controlPrimed = false;
  rawErrPrev = NAN;
  csvHeaderPrinted = false;

  Serial.print("CTRL,ON,gain:");
  Serial.print(gainInput, 3);
  Serial.print(",zero:");
  Serial.print(zeroPos, 3);
  Serial.print(",ref:");
  Serial.print(refMm, 3);
  Serial.print(",base:");
  Serial.println(basePWM_guess, 1);
}

void processSerialCommand() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  String upper = line;
  upper.toUpperCase();

  if (upper == "OFF") {
    stopControlAndOutput();
    Serial.println("CTRL,OFF");
    return;
  }

  if (upper == "R") {
    leituraSensorVL53L0XResetEstimator();
    stopControlAndOutput();
    Serial.println("CTRL,RESET");
    return;
  }

  if (upper == "H" || upper == "HELP") {
    printHelp();
    return;
  }

  float a = 0.0f, b = 0.0f, c = 0.0f;

  if (sscanf(line.c_str(), "ON %f %f %f", &a, &b, &c) == 3 ||
      sscanf(line.c_str(), "on %f %f %f", &a, &b, &c) == 3) {
    armControl(a, b, c);
    return;
  }

  if (sscanf(line.c_str(), "BASE %f", &a) == 1 ||
      sscanf(line.c_str(), "base %f", &a) == 1) {
    basePWM_guess = clampf(a, 0.0f, (float)pwmMax);
    Serial.print("BASE,");
    Serial.println(basePWM_guess, 1);
    return;
  }

  Serial.print("Comando invalido: ");
  Serial.println(line);
  printHelp();
}

void configurePwm() {
  pinMode(PIN_ENABLE, OUTPUT);
  pinMode(PIN_DIRECAO, OUTPUT);

  digitalWrite(PIN_ENABLE, HIGH);
  digitalWrite(PIN_DIRECAO, HIGH);

  ledcSetup(PWM_CHANNEL, PWM_FREQUENCY_HZ, PWM_RESOLUTION_BIT);
  ledcAttachPin(PIN_PWM, PWM_CHANNEL);
  applyPWM(0);
}

// ------------------- SETUP -------------------
void setup() {
  Serial.begin(115200);
  delay(300);

  configurePwm();

  bool ok = leituraSensorVL53L0XBegin(21, 22);
  if (!ok) {
    Serial.println("ERRO: falha ao iniciar VL53L0X");
  } else {
    Serial.println("VL53L0X,OK");
  }

  lastStepMs = millis();
  lastStepUs = micros();
  lastLogMs = millis();

  printHelp();
}

// ------------------- LOOP -------------------
void loop() {
  processSerialCommand();

  unsigned long nowMs = millis();
  if ((nowMs - lastStepMs) < loopPeriodMs) {
    return;
  }
  lastStepMs = nowMs;

  unsigned long nowUs = micros();
  if (lastStepUs == 0) lastStepUs = nowUs;
  float dt = (nowUs - lastStepUs) * 1e-6f;
  lastStepUs = nowUs;
  dt = clampf(dt, 0.02f, 0.20f);

  leituraSensorVL53L0XUpdate(dt);
  float atualMm = leituraSensorVL53L0XGetFilteredMm();

  if (!controlEnabled) {
    applyPWM(0);

    if ((millis() - lastLogMs) >= logPeriodMs) {
      logCsv(atualMm);
      lastLogMs = millis();
    }
    return;
  }

  if (!finitef(atualMm)) {
    applyPWM(0);

    if ((millis() - lastLogMs) >= logPeriodMs) {
      logCsv(atualMm);
      lastLogMs = millis();
    }
    return;
  }

  // Primeira iteração após ON:
  // aplica só a pré-carga e arma a memória do D.
  if (!controlPrimed) {
    float rawErr = atualMm - refMm;
    rawErrPrev = rawErr;
    applyPWM((int)lroundf(basePWM_guess));
    controlPrimed = true;

    if ((millis() - lastLogMs) >= logPeriodMs) {
      logCsv(atualMm);
      lastLogMs = millis();
    }
    return;
  }

  // Erro bruto para o D
  float rawErr = atualMm - refMm;

  // Banda morta apenas no P
  float pErr = rawErr;
  if (fabsf(pErr) <= deadbandMm) {
    pErr = 0.0f;
  }

  // Derivada de 1 passo sobre o erro
  float dErr = 0.0f;
  if (finitef(rawErrPrev) && dt > 1e-5f) {
    dErr = (rawErr - rawErrPrev) / dt;
  }

  // Mapeamento discutido:
  // Kd = GAIN / 1000
  // Kp = Kd * ZERO_POS
  float Kd = gainInput / 1000.0f;
  float Kp = Kd * zeroPos;

  float termP = CONTROL_SIGN * Kp * pErr;
  float termD = CONTROL_SIGN * Kd * dErr;
  termD = clampf(termD, -dTermLimitPwm, dTermLimitPwm);

  float pwmCmd = basePWM_guess + termP + termD;
  pwmCmd = clampf(pwmCmd, (float)pwmMin, (float)pwmMax);

  applyPWM((int)lroundf(pwmCmd));
  rawErrPrev = rawErr;

  if ((millis() - lastLogMs) >= logPeriodMs) {
    logCsv(atualMm);
    lastLogMs = millis();
  }
}
