#include "Modulo_sensor.h"

static const unsigned long INTERVALO_AMOSTRAGEM_US = 100000UL;  // 100 ms
static unsigned long ultimoInstanteUs = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  bool ok = leituraSensorVL53L0XBegin();
  Serial.print("sensor_inicializado:");
  Serial.println(ok ? 1 : 0);

  ultimoInstanteUs = micros();
}

void loop() {
  if (!leituraSensorVL53L0XIsInitialized()) {
    delay(500);
    return;
  }

  unsigned long agoraUs = micros();
  unsigned long decorridoUs = agoraUs - ultimoInstanteUs;

  if (decorridoUs >= INTERVALO_AMOSTRAGEM_US) {
    float dt = decorridoUs * 1e-6f;
    ultimoInstanteUs = agoraUs;

    if (leituraSensorVL53L0XUpdate(dt)) {
      float raw = leituraSensorVL53L0XGetRawMm();
      float filtrado = leituraSensorVL53L0XGetFilteredMm();

      Serial.print("raw_mm:");
      if (isfinite(raw)) Serial.print(raw, 2);
      else Serial.print("nan");

      Serial.print(",filtrado_mm:");
      if (isfinite(filtrado)) Serial.println(filtrado, 2);
      else Serial.println("nan");
    }
  }
}
