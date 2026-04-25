#include <Wire.h>
#include <math.h>
#include <Adafruit_VL53L0X.h>

// ============================================================
// LEITURA DO VL53L0X + KALMAN ADAPTATIVO ROBUSTO
// ------------------------------------------------------------
// Recorte do projeto original contendo apenas:
// - leitura do sensor VL53L0X
// - buffers da janela recente
// - cálculo de features, trendScore, jitterScore e opticalRisk
// - detecção de repouso (rest mode)
// - filtro de Kalman adaptativo e robusto
// - saída serial para Serial Monitor e Serial Plotter
//
// Saída em cada ciclo:
//   medido_mm:<valor>,filtrado_mm:<valor>
// ============================================================

Adafruit_VL53L0X lox = Adafruit_VL53L0X();
VL53L0X_RangingMeasurementData_t measure;

// ============================================================
// AMOSTRAGEM
// ============================================================
unsigned long tempoAnteriorMs = 0;
unsigned long intervalo_ms = 60;   // agenda nominal
unsigned long lastStepMicros = 0;
float dt_used = 0.060f;            // dt real usado no ciclo

// ============================================================
// PARÂMETROS IDENTIFICADOS / AJUSTES
// ============================================================
const float R0_BASE = 2.19810276f;
const float QPOS_MIN_BASE = 0.109905138f;
const float QPOS_MAX_IDENTIFIED = 10.9905138f;

const float SIGMA_STOP = 1.4826f;
const float DELTA_P90_STOP = 3.0f;
const float SLOPE_SOFT_MM_PER_SAMPLE = 0.2857142857f;
const float SLOPE_HARD_MM_PER_SAMPLE = 0.5714285714f;

const float SIGNAL_REF_P10 = 19.7266f;
const float AMBIENT_REF_P90 = 0.1641f;

const float RAW_MIN_MM = 1.0f;
const float RAW_MAX_MM = 300.0f;

float qMaxScale = 0.25f;
float velLeakBase = 0.92f;
float velLeakRest = 0.65f;
float restRFloor = 12.0f;
float qPosRest = 0.010f;
float qVelRest = 0.001f;
float innovationDeadbandRest = 0.50f;

int restCounter = 0;
bool restMode = false;

// ============================================================
// JANELA DINÂMICA
// ============================================================
const int MAX_W = 21;
int currentW = 15;

float rawBuf[MAX_W];
float signalBuf[MAX_W];
float ambientBuf[MAX_W];
float spadBuf[MAX_W];
float innovationBuf[MAX_W];
int statusBuf[MAX_W];

int bufHead = 0;
int bufCount = 0;

// ============================================================
// ESTIMADOR KALMAN
// x = [posição; velocidade]
// ============================================================
bool kalmanInitialized = false;

float x_pos = 0.0f;
float x_vel = 0.0f;

float P11 = 2.0f, P12 = 0.0f;
float P21 = 0.0f, P22 = 1.0f;

float raw_mm = NAN;
float signal_mcps = NAN;
float ambient_mcps = NAN;
float spad_eff = NAN;
int range_status = -1;

float pred_mm = NAN;
float actual_mm = NAN;
float actual_v = NAN;

float meanW = NAN;
float stdW = NAN;
float madW = NAN;
float medianAbsDeltaW = NAN;
float slopeW = NAN;
float residStdW = NAN;
float coherenceW = NAN;

float innovationMeanW = NAN;
float innovationStdW = NAN;
float innovationMadW = NAN;
float innovationMedianAbsDeltaW = NAN;
float innovationSlopeW = NAN;
float innovationCoherenceW = NAN;
int innovationRunLenW = 0;
float innovationBlendUsed = 0.0f;

float signalMedW = NAN;
float signalStdW = NAN;
float ambientMedW = NAN;
float ambientStdW = NAN;
float spadMedW = NAN;
float spadStdW = NAN;
float badStatusRatioW = NAN;

float trendScore = 0.0f;
float jitterScore = 0.0f;
float opticalRisk = 0.0f;

float qPosUsed = QPOS_MIN_BASE;
float qVelUsed = 0.20f;
float rEffUsed = R0_BASE;
float innovationUsed = 0.0f;
float innovationNormUsed = 0.0f;
int measurementUsed = 0;

// ============================================================
// HELPERS MATEMÁTICOS
// ============================================================
float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

float lerpf(float a, float b, float t) {
  return a + (b - a) * clampf(t, 0.0f, 1.0f);
}

float sqrf(float x) {
  return x * x;
}

float fix1616ToFloat(FixPoint1616_t x) {
  return ((float)((int32_t)x)) / 65536.0f;
}

int circularIndexFromEnd(int back) {
  int idx = bufHead - 1 - back;
  while (idx < 0) idx += MAX_W;
  return idx % MAX_W;
}

void copyRecentSamples(float *dst, const float *src, int n) {
  for (int i = 0; i < n; i++) {
    int idx = circularIndexFromEnd(n - 1 - i);
    dst[i] = src[idx];
  }
}

void copyRecentStatus(int *dst, const int *src, int n) {
  for (int i = 0; i < n; i++) {
    int idx = circularIndexFromEnd(n - 1 - i);
    dst[i] = src[idx];
  }
}

float meanArray(const float *x, int n) {
  if (n <= 0) return NAN;
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i];
  return s / (float)n;
}

float stdArray(const float *x, int n, float meanVal) {
  if (n < 2) return 0.0f;
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += sqrf(x[i] - meanVal);
  return sqrtf(s / (float)(n - 1));
}

void sortArray(float *x, int n) {
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (x[j] < x[i]) {
        float tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
      }
    }
  }
}

float medianArray(const float *x, int n) {
  if (n <= 0) return NAN;
  float tmp[MAX_W];
  for (int i = 0; i < n; i++) tmp[i] = x[i];
  sortArray(tmp, n);
  if (n % 2 == 1) return tmp[n / 2];
  return 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
}

float madStdFromArray(const float *x, int n) {
  if (n <= 0) return NAN;
  float med = medianArray(x, n);
  float dev[MAX_W];
  for (int i = 0; i < n; i++) dev[i] = fabsf(x[i] - med);
  float mad = medianArray(dev, n);
  return 1.4826f * mad;
}

float medianAbsDelta(const float *x, int n) {
  if (n < 2) return 0.0f;
  float d[MAX_W];
  for (int i = 1; i < n; i++) d[i - 1] = fabsf(x[i] - x[i - 1]);
  return medianArray(d, n - 1);
}

float directionCoherence(const float *x, int n) {
  if (n < 2) return 0.0f;
  float sumDx = 0.0f;
  float sumAbsDx = 0.0f;
  for (int i = 1; i < n; i++) {
    float dx = x[i] - x[i - 1];
    sumDx += dx;
    sumAbsDx += fabsf(dx);
  }
  if (sumAbsDx <= 1e-6f) return 0.0f;
  return fabsf(sumDx) / sumAbsDx;
}

int trailingSignRunLength(const float *x, int n, float deadband) {
  if (n <= 0) return 0;

  int run = 0;
  int lastSign = 0;
  for (int i = n - 1; i >= 0; i--) {
    int s = 0;
    if (x[i] > deadband) s = 1;
    else if (x[i] < -deadband) s = -1;
    else break;

    if (lastSign == 0) {
      lastSign = s;
      run = 1;
    } else if (s == lastSign) {
      run++;
    } else {
      break;
    }
  }
  return run;
}

void linearFit(const float *y, int n, float &slope, float &intercept, float &residStd) {
  if (n < 2) {
    slope = 0.0f;
    intercept = (n == 1) ? y[0] : 0.0f;
    residStd = 0.0f;
    return;
  }

  float sx = 0.0f, sy = 0.0f, sxx = 0.0f, sxy = 0.0f;
  for (int i = 0; i < n; i++) {
    float x = (float)i;
    sx += x;
    sy += y[i];
    sxx += x * x;
    sxy += x * y[i];
  }

  float denom = (float)n * sxx - sx * sx;
  if (fabsf(denom) < 1e-6f) {
    slope = 0.0f;
    intercept = sy / (float)n;
    residStd = 0.0f;
    return;
  }

  slope = ((float)n * sxy - sx * sy) / denom;
  intercept = (sy - slope * sx) / (float)n;

  if (n < 3) {
    residStd = 0.0f;
    return;
  }

  float sr = 0.0f;
  for (int i = 0; i < n; i++) {
    float fit = intercept + slope * (float)i;
    sr += sqrf(y[i] - fit);
  }
  residStd = sqrtf(sr / (float)(n - 2));
}

// ============================================================
// BUFFERS E RESET
// ============================================================
void clearBuffers() {
  bufHead = 0;
  bufCount = 0;
  for (int i = 0; i < MAX_W; i++) {
    rawBuf[i] = 0.0f;
    signalBuf[i] = 0.0f;
    ambientBuf[i] = 0.0f;
    spadBuf[i] = 0.0f;
    innovationBuf[i] = 0.0f;
    statusBuf[i] = -1;
  }
}

void pushSample(float raw, float signal, float ambient, float spad, float innovation, int status) {
  rawBuf[bufHead] = raw;
  signalBuf[bufHead] = signal;
  ambientBuf[bufHead] = ambient;
  spadBuf[bufHead] = spad;
  innovationBuf[bufHead] = innovation;
  statusBuf[bufHead] = status;
  bufHead = (bufHead + 1) % MAX_W;
  if (bufCount < MAX_W) bufCount++;
}

void resetEstimator(bool clearAllBuffers = true) {
  kalmanInitialized = false;
  x_pos = 0.0f;
  x_vel = 0.0f;

  P11 = 2.0f;
  P12 = 0.0f;
  P21 = 0.0f;
  P22 = 1.0f;

  pred_mm = NAN;
  actual_mm = NAN;
  actual_v = NAN;

  meanW = NAN;
  stdW = NAN;
  madW = NAN;
  medianAbsDeltaW = NAN;
  slopeW = NAN;
  residStdW = NAN;
  coherenceW = NAN;

  innovationMeanW = NAN;
  innovationStdW = NAN;
  innovationMadW = NAN;
  innovationMedianAbsDeltaW = NAN;
  innovationSlopeW = NAN;
  innovationCoherenceW = NAN;
  innovationRunLenW = 0;
  innovationBlendUsed = 0.0f;

  signalMedW = NAN;
  signalStdW = NAN;
  ambientMedW = NAN;
  ambientStdW = NAN;
  spadMedW = NAN;
  spadStdW = NAN;
  badStatusRatioW = NAN;

  trendScore = 0.0f;
  jitterScore = 0.0f;
  opticalRisk = 0.0f;

  qPosUsed = QPOS_MIN_BASE;
  qVelUsed = 0.20f;
  rEffUsed = R0_BASE;
  innovationUsed = 0.0f;
  innovationNormUsed = 0.0f;
  measurementUsed = 0;

  restCounter = 0;
  restMode = false;

  if (clearAllBuffers) clearBuffers();
}

void seedEstimator(float firstMeasurement) {
  kalmanInitialized = true;
  x_pos = firstMeasurement;
  x_vel = 0.0f;

  P11 = 0.80f;
  P12 = 0.0f;
  P21 = 0.0f;
  P22 = 0.20f;

  pred_mm = firstMeasurement;
  actual_mm = firstMeasurement;
  actual_v = 0.0f;
}

// ============================================================
// FEATURES
// ============================================================
bool computeWindowFeatures() {
  int n = min(bufCount, currentW);
  if (n < 4) return false;

  float rawTmp[MAX_W];
  float signalTmp[MAX_W];
  float ambientTmp[MAX_W];
  float spadTmp[MAX_W];
  float innovationTmp[MAX_W];
  int statusTmp[MAX_W];

  copyRecentSamples(rawTmp, rawBuf, n);
  copyRecentSamples(signalTmp, signalBuf, n);
  copyRecentSamples(ambientTmp, ambientBuf, n);
  copyRecentSamples(spadTmp, spadBuf, n);
  copyRecentSamples(innovationTmp, innovationBuf, n);
  copyRecentStatus(statusTmp, statusBuf, n);

  meanW = meanArray(rawTmp, n);
  stdW = stdArray(rawTmp, n, meanW);
  madW = madStdFromArray(rawTmp, n);
  medianAbsDeltaW = medianAbsDelta(rawTmp, n);
  coherenceW = directionCoherence(rawTmp, n);

  float interceptDummy = 0.0f;
  linearFit(rawTmp, n, slopeW, interceptDummy, residStdW);

  innovationMeanW = meanArray(innovationTmp, n);
  innovationStdW = stdArray(innovationTmp, n, innovationMeanW);
  innovationMadW = madStdFromArray(innovationTmp, n);
  innovationMedianAbsDeltaW = medianAbsDelta(innovationTmp, n);
  innovationCoherenceW = directionCoherence(innovationTmp, n);
  innovationRunLenW = trailingSignRunLength(innovationTmp, n, 0.60f);

  float innovationInterceptDummy = 0.0f;
  float innovationResidDummy = 0.0f;
  linearFit(innovationTmp, n, innovationSlopeW, innovationInterceptDummy, innovationResidDummy);

  signalMedW = medianArray(signalTmp, n);
  ambientMedW = medianArray(ambientTmp, n);
  spadMedW = medianArray(spadTmp, n);

  float signalMean = meanArray(signalTmp, n);
  float ambientMean = meanArray(ambientTmp, n);
  float spadMean = meanArray(spadTmp, n);

  signalStdW = stdArray(signalTmp, n, signalMean);
  ambientStdW = stdArray(ambientTmp, n, ambientMean);
  spadStdW = stdArray(spadTmp, n, spadMean);

  int bad = 0;
  for (int i = 0; i < n; i++) {
    if (statusTmp[i] != 0) bad++;
  }
  badStatusRatioW = (float)bad / (float)n;

  return true;
}

// ============================================================
// SCORES
// ============================================================
float computeTrendScore(float predictedPos) {
  float rawSlopeNorm = 0.0f;
  if (SLOPE_HARD_MM_PER_SAMPLE > SLOPE_SOFT_MM_PER_SAMPLE) {
    rawSlopeNorm = (fabsf(slopeW) - SLOPE_SOFT_MM_PER_SAMPLE) /
                   (SLOPE_HARD_MM_PER_SAMPLE - SLOPE_SOFT_MM_PER_SAMPLE);
  }
  rawSlopeNorm = clampf(rawSlopeNorm, 0.0f, 1.0f);

  float rawCoherenceNorm = clampf((coherenceW - 0.55f) / (0.90f - 0.55f), 0.0f, 1.0f);

  float rawScale = max(2.0f * SIGMA_STOP, max(2.0f * madW, 1.0f));
  float rawOffsetNorm = fabsf(meanW - predictedPos) / rawScale;
  rawOffsetNorm = clampf((rawOffsetNorm - 0.75f) / (2.0f - 0.75f), 0.0f, 1.0f);

  float rawScore = 0.55f * rawSlopeNorm + 0.30f * rawCoherenceNorm + 0.15f * rawOffsetNorm;
  rawScore = clampf(rawScore, 0.0f, 1.0f);

  float innovationScale = max(1.2f * SIGMA_STOP, max(1.5f * innovationMadW, 0.75f));

  float innovationBiasNorm = fabsf(innovationMeanW) / innovationScale;
  innovationBiasNorm = clampf((innovationBiasNorm - 0.35f) / (1.50f - 0.35f), 0.0f, 1.0f);

  float innovationSlopeSoft = 0.45f * SLOPE_SOFT_MM_PER_SAMPLE;
  float innovationSlopeHard = 0.85f * SLOPE_HARD_MM_PER_SAMPLE;
  float innovationSlopeNorm = 0.0f;
  if (innovationSlopeHard > innovationSlopeSoft) {
    innovationSlopeNorm = (fabsf(innovationSlopeW) - innovationSlopeSoft) /
                          (innovationSlopeHard - innovationSlopeSoft);
  }
  innovationSlopeNorm = clampf(innovationSlopeNorm, 0.0f, 1.0f);

  float innovationCoherenceNorm = clampf((innovationCoherenceW - 0.40f) / (0.88f - 0.40f), 0.0f, 1.0f);
  float innovationRunNorm = clampf(((float)innovationRunLenW - 2.0f) / (8.0f - 2.0f), 0.0f, 1.0f);

  float innovationScore = 0.40f * innovationRunNorm
                        + 0.25f * innovationCoherenceNorm
                        + 0.20f * innovationBiasNorm
                        + 0.15f * innovationSlopeNorm;
  innovationScore = clampf(innovationScore, 0.0f, 1.0f);

  float windowProgress = clampf(((float)min(bufCount, currentW) - 4.0f) / max((float)(currentW - 4), 1.0f), 0.0f, 1.0f);
  float innovationBlend = 0.20f + 0.35f * innovationRunNorm + 0.10f * innovationCoherenceNorm;
  innovationBlend = clampf(innovationBlend * windowProgress, 0.15f, 0.65f);
  innovationBlendUsed = innovationBlend;

  float score = (1.0f - innovationBlend) * rawScore + innovationBlend * innovationScore;
  return clampf(score, 0.0f, 1.0f);
}

float computeJitterScore(float trend) {
  float stdNorm = stdW / max(1.4f * SIGMA_STOP, 0.7f);
  stdNorm = clampf(stdNorm, 0.0f, 1.0f);

  float deltaNorm = medianAbsDeltaW / max(DELTA_P90_STOP, 1.0f);
  deltaNorm = clampf(deltaNorm, 0.0f, 1.0f);

  float residNorm = residStdW / max(1.5f * SIGMA_STOP, 0.7f);
  residNorm = clampf(residNorm, 0.0f, 1.0f);

  float antiCoherence = 1.0f - clampf(coherenceW, 0.0f, 1.0f);

  float score = 0.35f * stdNorm + 0.25f * deltaNorm + 0.20f * residNorm + 0.20f * antiCoherence;
  score *= (1.0f - 0.55f * trend);
  return clampf(score, 0.0f, 1.0f);
}

float computeOpticalRisk() {
  float ambientHigh = 0.0f;
  if (ambientMedW > AMBIENT_REF_P90) {
    ambientHigh = (ambientMedW - AMBIENT_REF_P90) / max(AMBIENT_REF_P90, 0.05f);
  }
  ambientHigh = clampf(ambientHigh, 0.0f, 1.0f);

  float signalLow = 0.0f;
  if (signalMedW < SIGNAL_REF_P10) {
    signalLow = (SIGNAL_REF_P10 - signalMedW) / max(SIGNAL_REF_P10, 1.0f);
  }
  signalLow = clampf(signalLow, 0.0f, 1.0f);

  float signalCv = signalStdW / max(signalMedW, 1.0f);
  float spadCv = spadStdW / max(spadMedW, 1.0f);
  float instability = clampf(1.8f * signalCv + 1.2f * spadCv, 0.0f, 1.0f);

  float risk = 0.55f * clampf(badStatusRatioW, 0.0f, 1.0f)
             + 0.20f * ambientHigh
             + 0.15f * signalLow
             + 0.10f * instability;

  return clampf(risk, 0.0f, 1.0f);
}

// ============================================================
// MODO ESTACIONÁRIO
// ============================================================
bool restEvidence() {
  bool condTrend = trendScore < 0.08f;
  bool condSlope = fabsf(slopeW) < 0.08f;
  bool condStd = stdW < 1.15f * SIGMA_STOP;
  bool condDelta = medianAbsDeltaW < 1.30f;
  bool condCoh = coherenceW < 0.20f;
  bool condOpt = opticalRisk < 0.05f;
  return condTrend && condSlope && condStd && condDelta && condCoh && condOpt;
}

void updateRestMode() {
  if (restEvidence()) {
    if (restCounter < 20) restCounter++;
  } else {
    if (restCounter > -20) restCounter--;
  }

  if (!restMode && restCounter >= 3) restMode = true;
  if (restMode && restCounter <= -2) restMode = false;
}

// ============================================================
// SENSOR
// ============================================================
bool readSensor() {
  VL53L0X_Error sensorStatus = lox.rangingTest(&measure, false);

  if (sensorStatus != VL53L0X_ERROR_NONE) {
    raw_mm = NAN;
    signal_mcps = NAN;
    ambient_mcps = NAN;
    spad_eff = NAN;
    range_status = 254;
    return false;
  }

  raw_mm = (float)measure.RangeMilliMeter;
  signal_mcps = fix1616ToFloat(measure.SignalRateRtnMegaCps);
  ambient_mcps = fix1616ToFloat(measure.AmbientRateRtnMegaCps);
  spad_eff = ((float)measure.EffectiveSpadRtnCount) / 256.0f;
  range_status = (int)measure.RangeStatus;

  return true;
}

bool measurementPhysicallyValid() {
  if (range_status != 0) return false;
  if (!isfinite(raw_mm)) return false;
  if (raw_mm < RAW_MIN_MM || raw_mm > RAW_MAX_MM) return false;
  return true;
}

// ============================================================
// KALMAN
// ============================================================
void applyRestVelocityDrain(float innovationAbs, bool hasInnovation) {
  if (!restMode) return;

  float drainFactor = 1.0f;
  if (hasInnovation) {
    float quietSpan = max(2.0f * innovationDeadbandRest, 0.10f);
    float quietRatio = 1.0f - clampf(innovationAbs / quietSpan, 0.0f, 1.0f);
    drainFactor = lerpf(1.0f, 0.35f, quietRatio);
  } else if (fabsf(x_vel) < 0.20f) {
    drainFactor = 0.65f;
  }

  x_vel *= drainFactor;

  if (fabsf(x_vel) < 0.08f) {
    if (!hasInnovation || innovationAbs < 0.50f * innovationDeadbandRest) {
      x_vel = 0.0f;
    }
  }
}

void kalmanStep(float z, bool measurementValid) {
  float velLeak = restMode ? velLeakRest : velLeakBase;
  float pred_v = velLeak * x_vel;
  pred_mm = x_pos + dt_used * pred_v;

  if (restMode) {
    qPosUsed = qPosRest;
    qVelUsed = qVelRest;
  } else {
    float qPosMaxUsed = qMaxScale * QPOS_MAX_IDENTIFIED;
    qPosUsed = lerpf(QPOS_MIN_BASE, qPosMaxUsed, trendScore);
    qVelUsed = lerpf(0.005f, 0.50f, trendScore);
  }

  float a12 = dt_used * velLeak;
  float a22 = velLeak;

  float P11Pred = P11 + a12 * (P12 + P21) + a12 * a12 * P22 + qPosUsed;
  float P12Pred = a22 * P12 + a12 * a22 * P22;
  float P21Pred = a22 * P21 + a12 * a22 * P22;
  float P22Pred = a22 * a22 * P22 + qVelUsed;

  innovationUsed = 0.0f;
  innovationNormUsed = 0.0f;
  measurementUsed = 0;

  if (!measurementValid) {
    x_pos = pred_mm;
    x_vel = pred_v;
    applyRestVelocityDrain(0.0f, false);
    P11 = P11Pred;
    P12 = P12Pred;
    P21 = P21Pred;
    P22 = P22Pred;
    actual_mm = x_pos;
    actual_v = x_vel;
    rEffUsed = R0_BASE * 25.0f;
    return;
  }

  float rFactor = 1.35f + 2.10f * jitterScore + 1.10f * opticalRisk - 0.25f * trendScore;
  rFactor = clampf(rFactor, 1.2f, 20.0f);
  float Rpre = R0_BASE * rFactor;

  float innovation = z - pred_mm;
  float innovationAbsPre = fabsf(innovation);

  if (restMode && innovationAbsPre < innovationDeadbandRest) {
    innovation = 0.0f;
  }

  float Spre = P11Pred + Rpre;
  float innovationNorm = innovation / sqrtf(max(Spre, 1e-6f));

  const float HUBER_C = 2.8f;
  float huberW = 1.0f;
  float absNu = fabsf(innovationNorm);
  if (absNu > HUBER_C) {
    huberW = HUBER_C / absNu;
  }
  huberW = clampf(huberW, 0.25f, 1.0f);

  rEffUsed = Rpre / (huberW * huberW);
  if (restMode && rEffUsed < restRFloor) {
    rEffUsed = restRFloor;
  }

  innovationUsed = innovation;
  innovationNormUsed = innovationNorm;

  float S = P11Pred + rEffUsed;
  float K1 = P11Pred / max(S, 1e-6f);
  float K2 = P21Pred / max(S, 1e-6f);

  x_pos = pred_mm + K1 * innovation;
  x_vel = pred_v + K2 * innovation;
  applyRestVelocityDrain(innovationAbsPre, true);

  float P11New = (1.0f - K1) * P11Pred;
  float P12New = (1.0f - K1) * P12Pred;
  float P21New = P21Pred - K2 * P11Pred;
  float P22New = P22Pred - K2 * P12Pred;

  P11 = P11New;
  P12 = P12New;
  P21 = P21New;
  P22 = P22New;

  actual_mm = x_pos;
  actual_v = x_vel;
  measurementUsed = 1;
}

// ============================================================
// TELEMETRIA
// ============================================================
void printTelemetry() {
  Serial.print("medido_mm:");
  if (isfinite(raw_mm)) Serial.print(raw_mm, 2);
  else Serial.print("nan");

  Serial.print(",filtrado_mm:");
  if (isfinite(actual_mm)) Serial.println(actual_mm, 2);
  else Serial.println("nan");
}

// ============================================================
// SETUP / LOOP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(21, 22);
  Wire.setClock(100000);

  bool ok = lox.begin(0x29, false, &Wire, Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_ACCURACY);
  if (!ok) {
    Serial.println("erro_sensor:1");
    while (1) {}
  }

  lox.setMeasurementTimingBudgetMicroSeconds(50000);

  resetEstimator(true);

  Serial.println("medido_mm:0,filtrado_mm:0");
}

void loop() {
  unsigned long nowMs = millis();
  if (nowMs - tempoAnteriorMs < intervalo_ms) return;
  tempoAnteriorMs = nowMs;

  unsigned long nowUs = micros();
  if (lastStepMicros == 0) lastStepMicros = nowUs;
  float dtMeas = (nowUs - lastStepMicros) * 1e-6f;
  lastStepMicros = nowUs;
  dt_used = clampf(dtMeas, 0.02f, 0.20f);

  bool sensorOk = readSensor();
  bool measValid = sensorOk && measurementPhysicallyValid();

  if (measValid) {
    float innovationPreview = 0.0f;
    if (kalmanInitialized) {
      float velLeakPreview = restMode ? velLeakRest : velLeakBase;
      float predPreview = x_pos + dt_used * (velLeakPreview * x_vel);
      innovationPreview = raw_mm - predPreview;
    }
    pushSample(raw_mm, signal_mcps, ambient_mcps, spad_eff, innovationPreview, range_status);
  }

  bool enoughWindow = computeWindowFeatures();

  if (!kalmanInitialized) {
    if (measValid && isfinite(raw_mm)) {
      seedEstimator(raw_mm);
    }
    printTelemetry();
    return;
  }

  if (!enoughWindow) {
    meanW = raw_mm;
    stdW = 0.0f;
    madW = 0.0f;
    medianAbsDeltaW = 0.0f;
    slopeW = 0.0f;
    residStdW = 0.0f;
    coherenceW = 0.0f;
    innovationMeanW = 0.0f;
    innovationStdW = 0.0f;
    innovationMadW = 0.0f;
    innovationMedianAbsDeltaW = 0.0f;
    innovationSlopeW = 0.0f;
    innovationCoherenceW = 0.0f;
    innovationRunLenW = 0;
    innovationBlendUsed = 0.0f;
    signalMedW = signal_mcps;
    ambientMedW = ambient_mcps;
    spadMedW = spad_eff;
    signalStdW = 0.0f;
    ambientStdW = 0.0f;
    spadStdW = 0.0f;
    badStatusRatioW = (range_status == 0) ? 0.0f : 1.0f;
    trendScore = 0.0f;
    jitterScore = 0.0f;
    opticalRisk = (range_status == 0) ? 0.0f : 1.0f;
    restMode = false;
    restCounter = 0;
  } else {
    float velLeakPreview = restMode ? velLeakRest : velLeakBase;
    float predictedPosSimple = x_pos + dt_used * (velLeakPreview * x_vel);
    trendScore = computeTrendScore(predictedPosSimple);
    jitterScore = computeJitterScore(trendScore);
    opticalRisk = computeOpticalRisk();
    updateRestMode();
  }

  kalmanStep(raw_mm, measValid);
  printTelemetry();
}
