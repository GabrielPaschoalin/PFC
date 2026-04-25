#include "Adafruit_VL53L0X.h"

// Cria o objeto do sensor VL53L0X
Adafruit_VL53L0X lox = Adafruit_VL53L0X();

unsigned long tempoAnterior = 0;
const long intervalo = 1000;  // 1 segundo

void setup() {
  Serial.begin(115200);
  // Wire.begin(18, 19); // SDA, SCL
  delay(1000);  // tempo para o monitor serial abrir

  Serial.println("Teste com VL53L0X e Wemos D1 R32");

  if (!lox.begin()) {
    Serial.println(F("Falha ao iniciar o VL53L0X"));
    while (1)
      ;
  }

  Serial.println(F("Exemplo de medição simples da API VL53L0X"));
}

void loop() {
  VL53L0X_RangingMeasurementData_t measure;

  unsigned long tempoAtual = millis();

  if (tempoAtual - tempoAnterior >= intervalo) {
    tempoAnterior = tempoAtual;

    // Obtém uma medição do sensor
    lox.rangingTest(&measure, false);

    if (measure.RangeStatus != 4) {  // 4 significa fora de alcance
      Serial.print(measure.RangeMilliMeter);
      Serial.print(" ");
      Serial.println(0);
    } else {
    }
  }
}
