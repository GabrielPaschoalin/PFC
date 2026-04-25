#include "leitura_sensor_vl53l0x.h"

static const unsigned long INTERVALO_AMOSTRAGEM_MS = 100;
static const float DT_AMOSTRAGEM_S = 0.100f;

static const int N_AMOSTRAS_MEDIA = 1;
static const int N_DESCARTE_INICIAL = 0;

unsigned long ultimoPassoMs = 0;
unsigned long ultimoAvisoErroMs = 0;

static const float GANHO_CAL = 0.9470f;
// static const float POLARIZACAO_CAL = -18.5098f;
static const float POLARIZACAO_CAL = -10.0f;

int descarteAtual = 0;
int amostrasColetadas = 0;
float somaFiltrado = 0.0f;
float mediaFinalMm = NAN;
bool calibracaoConcluida = false;

float aplicarCalibracaoMm(float leituraFiltradaMm) {
  return GANHO_CAL * leituraFiltradaMm + POLARIZACAO_CAL;
}

void imprimirCabecalho() {
  Serial.println("raw_mm,filtrado_mm,media_mm");
}

void imprimirLinha(float raw, float filtrado, float media) {
  if (isfinite(raw)) Serial.print(raw, 2);
  else Serial.print("nan");

  Serial.print(",");

  if (isfinite(filtrado)) Serial.print(filtrado, 2);
  else Serial.print("nan");

  Serial.print(",");

  if (isfinite(media)) Serial.println(media, 2);
  else Serial.println("nan");
}

void setup() {
  Serial.begin(115200);
  delay(500);

  bool ok = leituraSensorVL53L0XBegin();
  Serial.print("sensor_inicializado=");
  Serial.println(ok ? 1 : 0);

  Serial.print("intervalo_ms=");
  Serial.println(INTERVALO_AMOSTRAGEM_MS);
  Serial.print("n_media=");
  Serial.println(N_AMOSTRAS_MEDIA);
  Serial.print("n_descarte=");
  Serial.println(N_DESCARTE_INICIAL);

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

  float raw = leituraSensorVL53L0XGetRawMm();
  float filtrado = leituraSensorVL53L0XGetFilteredMm();

  if (!isfinite(filtrado)) {
    imprimirLinha(raw, filtrado, NAN);
    return;
  }

  if (descarteAtual < N_DESCARTE_INICIAL) {
    descarteAtual++;
    // imprimirLinha(raw, filtrado, NAN);
    return;
  }

  if (!calibracaoConcluida) {
    somaFiltrado += filtrado;
    amostrasColetadas++;

    float mediaParcial = somaFiltrado / (float)amostrasColetadas;
    // imprimirLinha(raw, filtrado, mediaParcial);

    if (amostrasColetadas >= N_AMOSTRAS_MEDIA) {
      mediaFinalMm = mediaParcial;
      // calibracaoConcluida = true;
      somaFiltrado = 0.0f;
      amostrasColetadas = 0;

      float calibrado = aplicarCalibracaoMm(mediaFinalMm);
      // Serial.println("calibracao_concluida=1");
      // Serial.print("media_final_mm= ");
      Serial.print(mediaFinalMm, 2);
      Serial.print(" ");
      // Serial.print("calibrado=");
      Serial.println(calibrado, 2);

    }
    return;
  }

  // imprimirLinha(raw, filtrado, mediaFinalMm);
}
