#include <Wire.h>
#include <math.h>
#include <Adafruit_VL53L0X.h>

// =====================================================
// CONFIG GERAL
// =====================================================
const int pinPWM = 27;
const int pinEnable = 14;
const int pinDirecao = 26;

int pwm_valor = 0;

// Amostragem
const unsigned long intervalo = 60;   // ms
unsigned long tempoAnterior = 0;

// Frequência de amostragem derivada do intervalo
const float fs = 1000.0f / (float)intervalo;

// Corte do Butterworth
// Pode ajustar depois. 1.0 a 1.5 Hz costuma ser um bom começo
const float fc = 1.2f;

// Sensor
Adafruit_VL53L0X lox = Adafruit_VL53L0X();
VL53L0X_RangingMeasurementData_t measure;

// =====================================================
// DIAGNÓSTICO / SAÍDAS
// =====================================================
float distancia_raw = -1.0f;
float distancia_filt = -1.0f;
float ultima_distancia_valida = -1.0f;

uint8_t range_status = 255;
float signal_mcps = 0.0f;
float ambient_mcps = 0.0f;
float spad_eff = 0.0f;

int accepted = 0;   // 1 = leitura entrou no pipeline do filtro, 0 = rejeitada

// =====================================================
// GATE DE QUALIDADE
// Ajustado para testes com bolinha PARADA
// =====================================================
const float MIN_VALID_MM = 10.0f;
const float MAX_VALID_MM = 120.0f;

// thresholds suaves; depois vocês podem apertar
const float MIN_SIGNAL_MCPS = 5.0f;
const float MIN_SPAD_EFF = 1.5f;

// como a bolinha ficará parada, salto muito grande tende a ser leitura ruim
const float MAX_STEP_MM_STATIONARY = 8.0f;

bool usarStepGate = true;
bool temUltimoRawAceito = false;
float ultimoRawAceito = -1.0f;

// =====================================================
// BUTTERWORTH 2a ORDEM - GANHO DC = 1
// =====================================================
float a1, a2, b0, b1, b2;

// estados
float x_1 = 0.0f, x_2 = 0.0f;
float y_1 = 0.0f, y_2 = 0.0f;

// flag de inicialização
bool filtroInicializado = false;

void calcularCoeficientesButterworth() {
  // Forma correta com ganho DC unitário
  // K = tan(pi*fc/fs)
  // norm = 1 / (1 + sqrt(2)K + K^2)
  // b0 = K^2 * norm
  // b1 = 2*b0
  // b2 = b0
  // a1 = 2*(K^2 - 1)*norm
  // a2 = (1 - sqrt(2)K + K^2)*norm
  float K = tanf(PI * fc / fs);
  float K2 = K * K;
  float q = sqrtf(2.0f);
  float norm = 1.0f / (1.0f + q * K + K2);

  b0 = K2 * norm;
  b1 = 2.0f * b0;
  b2 = b0;

  a1 = 2.0f * (K2 - 1.0f) * norm;
  a2 = (1.0f - q * K + K2) * norm;
}

float butterworth(float x) {
  float y = b0 * x + b1 * x_1 + b2 * x_2 - a1 * y_1 - a2 * y_2;

  x_2 = x_1;
  x_1 = x;
  y_2 = y_1;
  y_1 = y;

  return y;
}

// =====================================================
// FILTRO DE OUTLIER
// =====================================================
const int N_OUT = 20;
float outBuffer[N_OUT];
int outIdx = 0;
bool outCheio = false;

float k_sigma = 2.0f;
float media = 0.0f;
float desvio = 0.0f;

void atualizarEstatisticasOutlier() {
  int tamanho = outCheio ? N_OUT : outIdx;
  if (tamanho < 2) return;

  float soma = 0.0f;
  for (int i = 0; i < tamanho; i++) soma += outBuffer[i];
  media = soma / tamanho;

  float var = 0.0f;
  for (int i = 0; i < tamanho; i++) {
    float d = outBuffer[i] - media;
    var += d * d;
  }
  desvio = sqrtf(var / (tamanho - 1));
}

float filtroOutlier(float valor) {
  if (!outCheio && outIdx < 5) {
    outBuffer[outIdx++] = valor;
    return valor;
  }

  atualizarEstatisticasOutlier();

  if (desvio > 0.0f && fabsf(valor - media) > k_sigma * desvio) {
    return media;
  }

  outBuffer[outIdx] = valor;
  outIdx = (outIdx + 1) % N_OUT;
  if (outIdx == 0) outCheio = true;

  return valor;
}

// =====================================================
// MEDIANA CURTA
// =====================================================
const int MED_N = 5;
float medBuffer[MED_N];
int medIdx = 0;
bool medCheio = false;

void inserirMediana(float v) {
  medBuffer[medIdx] = v;
  medIdx = (medIdx + 1) % MED_N;
  if (medIdx == 0) medCheio = true;
}

float medianaAtual() {
  int n = medCheio ? MED_N : medIdx;
  if (n <= 0) return ultima_distancia_valida;

  float temp[MED_N];
  for (int i = 0; i < n; i++) temp[i] = medBuffer[i];

  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (temp[j] < temp[i]) {
        float aux = temp[i];
        temp[i] = temp[j];
        temp[j] = aux;
      }
    }
  }

  if (n % 2 == 1) return temp[n / 2];
  return 0.5f * (temp[n / 2 - 1] + temp[n / 2]);
}

// =====================================================
// HELPERS
// =====================================================
float fix1616ToFloat(FixPoint1616_t x) {
  return ((float)((int32_t)x)) / 65536.0f;
}

void inicializarFiltros(float valorInicial) {
  x_1 = valorInicial;
  x_2 = valorInicial;
  y_1 = valorInicial;
  y_2 = valorInicial;

  for (int i = 0; i < N_OUT; i++) outBuffer[i] = valorInicial;
  outIdx = 5;
  outCheio = false;
  media = valorInicial;
  desvio = 0.0f;

  for (int i = 0; i < MED_N; i++) medBuffer[i] = valorInicial;
  medIdx = 0;
  medCheio = false;

  distancia_filt = valorInicial;
  ultima_distancia_valida = valorInicial;
  filtroInicializado = true;
}

void resetarEstadoFiltro() {
  filtroInicializado = false;
  temUltimoRawAceito = false;
  ultimoRawAceito = -1.0f;
  distancia_filt = -1.0f;
  ultima_distancia_valida = -1.0f;
}

bool leituraConfiavel(float raw) {
  if (range_status != 0) return false;
  if (raw < MIN_VALID_MM || raw > MAX_VALID_MM) return false;
  if (signal_mcps < MIN_SIGNAL_MCPS) return false;
  if (spad_eff < MIN_SPAD_EFF) return false;

  if (usarStepGate && temUltimoRawAceito) {
    if (fabsf(raw - ultimoRawAceito) > MAX_STEP_MM_STATIONARY) {
      return false;
    }
  }

  return true;
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(pinEnable, OUTPUT);
  pinMode(pinDirecao, OUTPUT);

  digitalWrite(pinEnable, HIGH);
  digitalWrite(pinDirecao, LOW);

  analogWrite(pinPWM, pwm_valor);

  Wire.begin(21, 22);
  Wire.setClock(100000);

  bool ok = lox.begin(0x29, false, &Wire, Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_ACCURACY);
  if (!ok) {
    Serial.println("erro_sensor:1");
    while (1);
  }

  // Mais estável sem ficar lento demais
  lox.setMeasurementTimingBudgetMicroSeconds(50000);

  calcularCoeficientesButterworth();
  resetarEstadoFiltro();

  Serial.println("raw_mm:-1,filt_mm:-1,status:-1,signal_mcps:-1,ambient_mcps:-1,spad_eff:-1,pwm:-1,accepted:-1");
}

// =====================================================
// LOOP
// =====================================================
void loop() {
  unsigned long tempoAtual = millis();

  if (tempoAtual - tempoAnterior >= intervalo) {
    tempoAnterior = tempoAtual;

    tratarComandoSerial();
    analogWrite(pinPWM, pwm_valor);

    leituraDistancia();

    accepted = 0;

    bool ok = leituraConfiavel(distancia_raw);

    if (ok) {
      accepted = 1;

      if (!filtroInicializado) {
        inicializarFiltros(distancia_raw);
      } else {
        float d = distancia_raw;

        d = filtroOutlier(d);
        inserirMediana(d);
        d = medianaAtual();
        d = butterworth(d);

        distancia_filt = d;
        ultima_distancia_valida = d;
      }

      ultimoRawAceito = distancia_raw;
      temUltimoRawAceito = true;
    } else {
      // mantém última boa
      distancia_filt = ultima_distancia_valida;
    }

    imprimirPlotter();
  }
}

// =====================================================
// SERIAL
// Comandos:
// - número inteiro: muda PWM
// - r : reset/rearma o filtro após reposicionar a bolinha
// =====================================================
void tratarComandoSerial() {
  if (!Serial.available()) return;

  String comando = Serial.readStringUntil('\n');
  comando.trim();

  if (comando.length() == 0) return;

  if (comando.equalsIgnoreCase("r")) {
    resetarEstadoFiltro();
    return;
  }

  pwm_valor = comando.toInt();
  pwm_valor = constrain(pwm_valor, 0, 255);
}

// =====================================================
// LEITURA DO SENSOR
// =====================================================
void leituraDistancia() {
  VL53L0X_Error sensorStatus = lox.rangingTest(&measure, false);

  if (sensorStatus != VL53L0X_ERROR_NONE) {
    distancia_raw = -1.0f;
    range_status = 254;
    signal_mcps = -1.0f;
    ambient_mcps = -1.0f;
    spad_eff = -1.0f;
    return;
  }

  distancia_raw = (float)measure.RangeMilliMeter;
  range_status = measure.RangeStatus;
  signal_mcps = fix1616ToFloat(measure.SignalRateRtnMegaCps);
  ambient_mcps = fix1616ToFloat(measure.AmbientRateRtnMegaCps);
  spad_eff = ((float)measure.EffectiveSpadRtnCount) / 256.0f;
}

// =====================================================
// SERIAL PLOTTER
// =====================================================
void imprimirPlotter() {
  Serial.print("raw_mm:");
  Serial.print(distancia_raw, 2);
  Serial.print(",");

  Serial.print("filt_mm:");
  Serial.print(distancia_filt, 2);
  Serial.print(",");

  Serial.print("status:");
  Serial.print((int)range_status);
  Serial.print(",");

  Serial.print("signal_mcps:");
  Serial.print(signal_mcps, 4);
  Serial.print(",");

  Serial.print("ambient_mcps:");
  Serial.print(ambient_mcps, 4);
  Serial.print(",");

  Serial.print("spad_eff:");
  Serial.print(spad_eff, 2);
  Serial.print(",");

  Serial.print("pwm:");
  Serial.print(pwm_valor);
  Serial.print(",");

  Serial.print("accepted:");
  Serial.println(accepted);
}
