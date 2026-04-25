#include <Arduino.h>
#include <math.h>
#include "src/leitura_sensor_vl53l0x.h"

// ============================================================
// MAGLEV - KALMAN ADAPTATIVO V11
// ------------------------------------------------------------
// Estrutura modular:
// - maglev_adaptive_kalman_v11.ino -> PWM manual, serial e telemetria
// - src/leitura_sensor_vl53l0x.h   -> interface do módulo do sensor
// - src/leitura_sensor_vl53l0x.cpp -> leitura + Kalman adaptativo
//
// Base desta versão:
// - módulo do sensor preservado conforme código validado em bancada
// - pinagem do PWM adaptada conforme PWM_extraiValores
// - ENA sempre em HIGH
// - IN2 sempre em LOW
// - PWM aplicado em IN1
// ============================================================

// ------------------- CONFIGURAÇÃO DE PINOS -------------------
const int pinPWM = 27;  // PWM no IN1 do L298N
const int pinENA = 14;  // ENA fixo em HIGH
const int pinIN2 = 26;  // IN2 fixo em LOW

// ------------------- PWM -------------------
const uint32_t freqPWM = 5000;
const uint8_t resolucaoPWM = 8;
const int pwmMin = 0;
const int pwmMax = 255;
bool pwmAttached = false;

// ------------------- SENSOR / AMOSTRAGEM -------------------
const uint8_t sensorSdaPin = 21;
const uint8_t sensorSclPin = 22;
const unsigned long intervaloLeituraMs = 60UL;

unsigned long ultimoLoopMs = 0;
unsigned long ultimoLoopUs = 0;
float dtLoopSeconds = 0.060f;

// ------------------- ESTADO -------------------
int pwmAtual = 0;
bool detailedOutput = false;
float raw_mm = NAN;
float actual_mm = NAN;

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
  // static const float POLARIZACAO_CAL = -18.5098f;
  static const float POLARIZACAO_CAL = -103.0f;
  return GANHO_CAL * leituraFiltradaMm + POLARIZACAO_CAL;
}

float normalizarCurva(float leituraFiltrada){ 
  static const float referencia = 72;
  static const int direcao = -1;
  return roundf(referencia - leituraFiltrada);
}

void aplicarPWM(int valor) {
  valor = constrain(valor, pwmMin, pwmMax);
  pwmAtual = valor;

  if (pwmAttached) {
    ledcWrite(pinPWM, (uint32_t)pwmAtual);
  }
}

void atualizarLeituraSensor() {
  leituraSensorVL53L0XUpdate(dtLoopSeconds);
  raw_mm = leituraSensorVL53L0XGetRawMm();
  actual_mm = aplicarCalibracaoMm(leituraSensorVL53L0XGetFilteredMm());
  actual_mm = normalizarCurva(actual_mm);
  // actual_mm = -(117 - actual_mm); // zero é quando a bolinha esá colada na bobina
}

// ============================================================
// SERIAL
// ============================================================
void printHelp() {
  Serial.println();
  Serial.println("Comandos disponiveis:");
  Serial.println("  reset       -> reseta apenas o estimador do sensor");
  Serial.println("  status      -> imprime o estado atual");
  Serial.println("  d=0         -> telemetria global (plotter)");
  Serial.println("  d=1         -> telemetria detalhada");
  Serial.println("  <0..255>    -> aplica PWM manualmente");
  Serial.println("  ?           -> ajuda");
  Serial.println();
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
  Serial.print(",output_mode:");
  Serial.println(detailedOutput ? 1 : 0);
}

void processarComando(String comando) {
  comando.trim();
  comando.toLowerCase();

  if (comando.length() == 0) return;

  if (comando == "r" || comando == "reset") {
    leituraSensorVL53L0XResetEstimator();
    return;
  }

  if (comando == "status") {
    printStatus();
    return;
  }

  if (comando == "?" || comando == "help") {
    printHelp();
    return;
  }

  if (comando.startsWith("d=")) {
    detailedOutput = (comando.substring(2).toInt() != 0);
    return;
  }

  bool ehNumero = true;
  for (unsigned int i = 0; i < comando.length(); i++) {
    if (!isDigit(comando[i])) {
      ehNumero = false;
      break;
    }
  }

  if (ehNumero) {
    int valor = comando.toInt();
    if (valor < pwmMin || valor > pwmMax) {
      Serial.println("Digite um valor entre 0 e 255.");
      return;
    }

    aplicarPWM(valor);
    return;
  }

  Serial.println("Comando invalido. Use: reset, status, d=0, d=1 ou um numero entre 0 e 255.");
}

// ============================================================
// TELEMETRIA
// ============================================================
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
  Serial.print("pwm:");
  Serial.print(pwmAtual);
  Serial.print(",dt:");
  Serial.print(dtLoopSeconds, 4);
  Serial.print(",raw_mm:");
  Serial.print(raw_mm, 3);
  Serial.print(",actual_mm:");
  Serial.print(roundf(actual_mm), 3);
  Serial.print(",sensor_init:");
  Serial.println(leituraSensorVL53L0XIsInitialized() ? 1 : 0);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // ENA fixo em HIGH
  pinMode(pinENA, OUTPUT);
  digitalWrite(pinENA, HIGH);

  // IN2 fixo em LOW
  pinMode(pinIN2, OUTPUT);
  digitalWrite(pinIN2, LOW);

  // PWM em IN1
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

  Serial.println("Sistema pronto.");
  Serial.println("Estrutura da v11:");
  Serial.println("- .ino principal: PWM manual / serial / telemetria");
  Serial.println("- modulo sensor: leitura + Kalman adaptativo");
  Serial.println("Ligacao esperada:");
  Serial.println("- pinPWM -> IN1");
  Serial.println("- pinENA -> ENA (sempre HIGH)");
  Serial.println("- pinIN2 -> IN2 (sempre LOW)");
  Serial.println("- SDA -> GPIO 21");
  Serial.println("- SCL -> GPIO 22");
  Serial.println("pwm:-1,raw:-1,actual:-1,scale70:70,scale80:80");

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

  if (detailedOutput) {
    printDetailedTelemetry();
  } else {
    printGlobalPlotTelemetry();
  }
}
