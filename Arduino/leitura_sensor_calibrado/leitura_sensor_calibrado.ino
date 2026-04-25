#include "leitura_sensor_vl53l0x.h"

static const unsigned long INTERVALO_AMOSTRAGEM_MS = 100;
static const float DT_AMOSTRAGEM_S = 0.100f;

// Coeficientes da reta de calibração
// distancia_real_mm = GANHO_CAL * leitura_filtrada_mm + POLARIZACAO_CAL
static const float GANHO_CAL = 0.9470f;
static const float POLARIZACAO_CAL = -18.5098f;

unsigned long ultimoPassoMs = 0;
unsigned long ultimoAvisoErroMs = 0;

float aplicarCalibracaoMm(float leituraFiltradaMm) {
  return GANHO_CAL * leituraFiltradaMm + POLARIZACAO_CAL;
}

void imprimirCabecalho() {
  Serial.println("raw_mm,filtrado_mm,calibrado_mm");
}

void imprimirValorOuNan(float valor, uint8_t casas = 2) {
  if (isfinite(valor)) Serial.print(valor, casas);
  else Serial.print("nan");
}

void setup() {
  Serial.begin(115200);
  delay(500);

  bool ok = leituraSensorVL53L0XBegin();

  Serial.print("sensor_inicializado=");
  Serial.println(ok ? 1 : 0);
  Serial.print("intervalo_ms=");
  Serial.println(INTERVALO_AMOSTRAGEM_MS);
  Serial.print("ganho_cal=");
  Serial.println(GANHO_CAL, 6);
  Serial.print("polarizacao_cal=");
  Serial.println(POLARIZACAO_CAL, 6);

  imprimirCabecalho();
  ultimoPassoMs = millis();
}

void loop() {
  if (!leituraSensorVL53L0XIsInitialized()) {
    unsigned long agora = millis();
    if (agora - ultimoAvisoErroMs >= 1000UL) {
      ultimoAvisoErroMs = agora;
      Serial.println("erro: sensor nao inicializado");
    }
    return;
  }

  unsigned long agora = millis();
  if (agora - ultimoPassoMs < INTERVALO_AMOSTRAGEM_MS) {
    return;
  }
  ultimoPassoMs = agora;

  if (!leituraSensorVL53L0XUpdate(DT_AMOSTRAGEM_S)) {
    return;
  }

  float rawMm = leituraSensorVL53L0XGetRawMm();
  float filtradoMm = leituraSensorVL53L0XGetFilteredMm();
  float calibradoMm = NAN;

  if (isfinite(filtradoMm)) {
    calibradoMm = aplicarCalibracaoMm(filtradoMm);
  }

  imprimirValorOuNan(rawMm);
  Serial.print(",");
  imprimirValorOuNan(filtradoMm);
  Serial.print(",");
  imprimirValorOuNan(calibradoMm);
  Serial.println();
}
