#include "leitura_sensor.h"

#include <Wire.h>
#include <math.h>
#include <Adafruit_VL53L0X.h>

namespace {

Adafruit_VL53L0X lox;
VL53L0X_RangingMeasurementData_t measure;

bool sensorInitialized = false;
bool streamStarted = false;
bool kalmanInitialized = false;

constexpr uint8_t SENSOR_I2C_ADDR = 0x29;
constexpr float REF_DT_S = 0.060f;
constexpr float DEFAULT_MEAS_DT_S = 0.020f;
constexpr float MIN_PREDICT_DT_S = 0.001f;
constexpr float MAX_PREDICT_DT_S = 0.050f;
constexpr float MIN_MEAS_DT_S = 0.005f;
constexpr float MAX_MEAS_DT_S = 0.100f;
constexpr float RAW_MIN_MM = 1.0f;
constexpr float RAW_MAX_MM = 300.0f;
constexpr uint32_t STREAM_STALL_RESTART_US = 150000UL;
constexpr int MAX_CONSECUTIVE_STREAM_ERRORS = 5;

constexpr float R0_BASE = 2.19810276f;
constexpr float QPOS_MIN_BASE = 0.109905138f;
constexpr float QPOS_MAX_IDENTIFIED = 10.9905138f;

constexpr float QPOS_MIN_RATE = QPOS_MIN_BASE / REF_DT_S;
constexpr float QPOS_MAX_RATE = QPOS_MAX_IDENTIFIED / REF_DT_S;
constexpr float QPOS_REST_RATE = 0.010f / REF_DT_S;
constexpr float QVEL_LOW_RATE = 0.005f / REF_DT_S;
constexpr float QVEL_HIGH_RATE = 0.50f / REF_DT_S;
constexpr float QVEL_REST_RATE = 0.001f / REF_DT_S;

constexpr float SIGMA_STOP = 1.4826f;
constexpr float DELTA_RATE_P90_STOP = 3.0f / REF_DT_S;
constexpr float SLOPE_SOFT_MM_PER_S = 0.2857142857f / REF_DT_S;
constexpr float SLOPE_HARD_MM_PER_S = 0.5714285714f / REF_DT_S;
constexpr float REST_SLOPE_MM_PER_S = 0.08f / REF_DT_S;

constexpr float SIGNAL_REF_P10 = 19.7266f;
constexpr float AMBIENT_REF_P90 = 0.1641f;

uint8_t currentSensorType = LEITURA_SENSOR_VL53L0X_MEDIUM;
uint32_t currentTimingBudgetUs = 32000UL;
uint32_t currentI2cClockHz = 400000UL;
bool applyAggressiveVcsel = false;
bool applyRelaxedChecks = false;
Adafruit_VL53L0X::VL53L0X_Sense_config_t currentSenseConfig = Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_SPEED;

float qMaxScale = 0.25f;
float velLeakBaseRef = 0.92f;
float velLeakRestRef = 0.65f;
float restRFloor = 12.0f;
float innovationDeadbandRest = 0.50f;

int restCounter = 0;
bool restMode = false;

constexpr int MAX_W = 15;
int currentW = 9;

float rawBuf[MAX_W];
float signalBuf[MAX_W];
float ambientBuf[MAX_W];
float spadBuf[MAX_W];
float innovationBuf[MAX_W];
float sampleDtBuf[MAX_W];
int statusBuf[MAX_W];

int bufHead = 0;
int bufCount = 0;

float x_pos = 0.0f;
float x_vel = 0.0f;
float P11 = 2.0f, P12 = 0.0f;
float P21 = 0.0f, P22 = 1.0f;

float raw_mm = NAN;
float signal_mcps = NAN;
float ambient_mcps = NAN;
float spad_eff = NAN;
int range_status = -1;
bool measurementValid = false;
bool newMeasurement = false;

float pred_mm = NAN;
float actual_mm = NAN;
float actual_v = NAN;

float predictDtUsed = DEFAULT_MEAS_DT_S;
float measurementDtUsed = DEFAULT_MEAS_DT_S;
float measurementRateHz = 0.0f;
float measurementAgeSeconds = 0.0f;

float meanW = NAN;
float stdW = NAN;
float madW = NAN;
float medianAbsDeltaRateW = NAN;
float slopeW = NAN;
float residStdW = NAN;
float coherenceW = NAN;
float dtMedW = DEFAULT_MEAS_DT_S;

float innovationMeanW = NAN;
float innovationStdW = NAN;
float innovationMadW = NAN;
float innovationMedianAbsDeltaRateW = NAN;
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

unsigned long lastPredictMicros = 0;
unsigned long lastMeasurementMicros = 0;
int consecutiveStreamErrors = 0;

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

float computeDtSeconds(unsigned long nowUs, unsigned long &lastUs, float fallback, float minDt, float maxDt) {
  float dt = fallback;
  if (lastUs != 0UL) {
    dt = (float)(nowUs - lastUs) * 1.0e-6f;
  }
  lastUs = nowUs;
  if (!isfinite(dt) || dt <= 0.0f) dt = fallback;
  return clampf(dt, minDt, maxDt);
}

float leakForDt(float leakAtRef, float dtSeconds) {
  if (!isfinite(dtSeconds) || dtSeconds <= 0.0f) return leakAtRef;
  return powf(clampf(leakAtRef, 0.01f, 0.9999f), dtSeconds / REF_DT_S);
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

float medianAbsDeltaRate(const float *x, const float *dt, int n) {
  if (n < 2) return 0.0f;
  float d[MAX_W];
  for (int i = 1; i < n; i++) {
    float dtLocal = clampf(dt[i], MIN_MEAS_DT_S, MAX_MEAS_DT_S);
    d[i - 1] = fabsf(x[i] - x[i - 1]) / dtLocal;
  }
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

void linearFitTimed(const float *y, const float *dt, int n, float &slope, float &intercept, float &residStd) {
  if (n < 2) {
    slope = 0.0f;
    intercept = (n == 1) ? y[0] : 0.0f;
    residStd = 0.0f;
    return;
  }

  float t[MAX_W];
  t[0] = 0.0f;
  for (int i = 1; i < n; i++) {
    t[i] = t[i - 1] + clampf(dt[i], MIN_MEAS_DT_S, MAX_MEAS_DT_S);
  }

  float sx = 0.0f, sy = 0.0f, sxx = 0.0f, sxy = 0.0f;
  for (int i = 0; i < n; i++) {
    sx += t[i];
    sy += y[i];
    sxx += t[i] * t[i];
    sxy += t[i] * y[i];
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
    float fit = intercept + slope * t[i];
    sr += sqrf(y[i] - fit);
  }
  residStd = sqrtf(sr / (float)(n - 2));
}

void clearBuffers() {
  bufHead = 0;
  bufCount = 0;
  for (int i = 0; i < MAX_W; i++) {
    rawBuf[i] = 0.0f;
    signalBuf[i] = 0.0f;
    ambientBuf[i] = 0.0f;
    spadBuf[i] = 0.0f;
    innovationBuf[i] = 0.0f;
    sampleDtBuf[i] = DEFAULT_MEAS_DT_S;
    statusBuf[i] = -1;
  }
}

void pushSample(float raw, float signal, float ambient, float spad, float innovation, int status, float sampleDt) {
  rawBuf[bufHead] = raw;
  signalBuf[bufHead] = signal;
  ambientBuf[bufHead] = ambient;
  spadBuf[bufHead] = spad;
  innovationBuf[bufHead] = innovation;
  sampleDtBuf[bufHead] = clampf(sampleDt, MIN_MEAS_DT_S, MAX_MEAS_DT_S);
  statusBuf[bufHead] = status;
  bufHead = (bufHead + 1) % MAX_W;
  if (bufCount < MAX_W) bufCount++;
}

void resetEstimatorInternal(bool clearAllBuffers = true) {
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
  medianAbsDeltaRateW = NAN;
  slopeW = NAN;
  residStdW = NAN;
  coherenceW = NAN;
  dtMedW = DEFAULT_MEAS_DT_S;

  innovationMeanW = NAN;
  innovationStdW = NAN;
  innovationMadW = NAN;
  innovationMedianAbsDeltaRateW = NAN;
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

  raw_mm = NAN;
  signal_mcps = NAN;
  ambient_mcps = NAN;
  spad_eff = NAN;
  range_status = -1;
  measurementValid = false;
  newMeasurement = false;

  predictDtUsed = DEFAULT_MEAS_DT_S;
  measurementDtUsed = DEFAULT_MEAS_DT_S;
  measurementRateHz = 0.0f;
  measurementAgeSeconds = 0.0f;

  consecutiveStreamErrors = 0;

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

bool computeWindowFeatures() {
  int n = min(bufCount, currentW);
  if (n < 4) return false;

  float rawTmp[MAX_W];
  float signalTmp[MAX_W];
  float ambientTmp[MAX_W];
  float spadTmp[MAX_W];
  float innovationTmp[MAX_W];
  float dtTmp[MAX_W];
  int statusTmp[MAX_W];

  copyRecentSamples(rawTmp, rawBuf, n);
  copyRecentSamples(signalTmp, signalBuf, n);
  copyRecentSamples(ambientTmp, ambientBuf, n);
  copyRecentSamples(spadTmp, spadBuf, n);
  copyRecentSamples(innovationTmp, innovationBuf, n);
  copyRecentSamples(dtTmp, sampleDtBuf, n);
  copyRecentStatus(statusTmp, statusBuf, n);

  meanW = meanArray(rawTmp, n);
  stdW = stdArray(rawTmp, n, meanW);
  madW = madStdFromArray(rawTmp, n);
  medianAbsDeltaRateW = medianAbsDeltaRate(rawTmp, dtTmp, n);
  coherenceW = directionCoherence(rawTmp, n);
  dtMedW = medianArray(dtTmp, n);

  float interceptDummy = 0.0f;
  linearFitTimed(rawTmp, dtTmp, n, slopeW, interceptDummy, residStdW);

  innovationMeanW = meanArray(innovationTmp, n);
  innovationStdW = stdArray(innovationTmp, n, innovationMeanW);
  innovationMadW = madStdFromArray(innovationTmp, n);
  innovationMedianAbsDeltaRateW = medianAbsDeltaRate(innovationTmp, dtTmp, n);
  innovationCoherenceW = directionCoherence(innovationTmp, n);
  innovationRunLenW = trailingSignRunLength(innovationTmp, n, 0.60f);

  float innovationInterceptDummy = 0.0f;
  float innovationResidDummy = 0.0f;
  linearFitTimed(innovationTmp, dtTmp, n, innovationSlopeW, innovationInterceptDummy, innovationResidDummy);

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

float computeTrendScore(float predictedPos) {
  float rawSlopeNorm = 0.0f;
  if (SLOPE_HARD_MM_PER_S > SLOPE_SOFT_MM_PER_S) {
    rawSlopeNorm = (fabsf(slopeW) - SLOPE_SOFT_MM_PER_S) /
                   (SLOPE_HARD_MM_PER_S - SLOPE_SOFT_MM_PER_S);
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

  float innovationSlopeSoft = 0.45f * SLOPE_SOFT_MM_PER_S;
  float innovationSlopeHard = 0.85f * SLOPE_HARD_MM_PER_S;
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

  float deltaNorm = medianAbsDeltaRateW / max(DELTA_RATE_P90_STOP, 1.0f);
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

bool restEvidence() {
  bool condTrend = trendScore < 0.08f;
  bool condSlope = fabsf(slopeW) < REST_SLOPE_MM_PER_S;
  bool condStd = stdW < 1.15f * SIGMA_STOP;
  bool condDelta = medianAbsDeltaRateW < (1.30f / REF_DT_S);
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

void loadMeasurementFieldsFromStruct() {
  raw_mm = (float)measure.RangeMilliMeter;
  signal_mcps = fix1616ToFloat(measure.SignalRateRtnMegaCps);
  ambient_mcps = fix1616ToFloat(measure.AmbientRateRtnMegaCps);
  spad_eff = ((float)measure.EffectiveSpadRtnCount) / 256.0f;
  range_status = (int)measure.RangeStatus;
}

bool measurementPhysicallyValid() {
  if (range_status != 0) return false;
  if (!isfinite(raw_mm)) return false;
  if (raw_mm < RAW_MIN_MM || raw_mm > RAW_MAX_MM) return false;
  return true;
}

void applySensorProfile() {
  switch (currentSensorType) {
    case LEITURA_SENSOR_VL53L0X_AGGRESSIVE:
      currentTimingBudgetUs = 20000UL;
      currentI2cClockHz = 400000UL;
      currentSenseConfig = Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_SPEED;
      applyAggressiveVcsel = true;
      applyRelaxedChecks = true;
      currentW = 7;
      qMaxScale = 0.35f;
      break;

    case LEITURA_SENSOR_VL53L0X_CONSERVATIVE:
      currentTimingBudgetUs = 50000UL;
      currentI2cClockHz = 400000UL;
      currentSenseConfig = Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_ACCURACY;
      applyAggressiveVcsel = false;
      applyRelaxedChecks = false;
      currentW = 11;
      qMaxScale = 0.25f;
      break;

    case LEITURA_SENSOR_VL53L0X_MEDIUM:
    default:
      currentSensorType = LEITURA_SENSOR_VL53L0X_MEDIUM;
      currentTimingBudgetUs = 32000UL;
      currentI2cClockHz = 400000UL;
      currentSenseConfig = Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_SPEED;
      applyAggressiveVcsel = false;
      applyRelaxedChecks = true;
      currentW = 9;
      qMaxScale = 0.30f;
      break;
  }
}

bool restartContinuousStream() {
  lox.stopMeasurement(false);
  if (lox.setDeviceMode(VL53L0X_DEVICEMODE_CONTINUOUS_RANGING, false) != VL53L0X_ERROR_NONE) {
    streamStarted = false;
    return false;
  }
  if (lox.startMeasurement(false) != VL53L0X_ERROR_NONE) {
    streamStarted = false;
    return false;
  }
  lox.clearInterruptMask(false);
  streamStarted = true;
  consecutiveStreamErrors = 0;
  return true;
}

bool pullMeasurementNonBlocking(unsigned long nowUs) {
  newMeasurement = false;

  if (!streamStarted) {
    return restartContinuousStream();
  }

  if ((lastMeasurementMicros != 0UL) && ((nowUs - lastMeasurementMicros) > STREAM_STALL_RESTART_US)) {
    restartContinuousStream();
  }

  if (!lox.isRangeComplete()) {
    return true;
  }

  if (lox.getRangingMeasurement(&measure, false) != VL53L0X_ERROR_NONE) {
    consecutiveStreamErrors++;
    if (consecutiveStreamErrors >= MAX_CONSECUTIVE_STREAM_ERRORS) {
      restartContinuousStream();
    }
    return false;
  }

  lox.clearInterruptMask(false);
  loadMeasurementFieldsFromStruct();

  measurementDtUsed = computeDtSeconds(nowUs, lastMeasurementMicros, DEFAULT_MEAS_DT_S, MIN_MEAS_DT_S, MAX_MEAS_DT_S);
  measurementRateHz = 1.0f / max(measurementDtUsed, 1e-6f);
  measurementAgeSeconds = 0.0f;
  newMeasurement = true;
  measurementValid = measurementPhysicallyValid();
  consecutiveStreamErrors = 0;
  return true;
}

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

void updateFeaturesFromMeasurement() {
  if (!newMeasurement || !measurementValid) return;

  float innovationPreview = 0.0f;
  if (kalmanInitialized) {
    innovationPreview = raw_mm - x_pos;
  }

  pushSample(raw_mm, signal_mcps, ambient_mcps, spad_eff, innovationPreview, range_status, measurementDtUsed);

  bool enoughWindow = computeWindowFeatures();
  if (!enoughWindow) {
    meanW = raw_mm;
    stdW = 0.0f;
    madW = 0.0f;
    medianAbsDeltaRateW = 0.0f;
    slopeW = 0.0f;
    residStdW = 0.0f;
    coherenceW = 0.0f;
    innovationMeanW = 0.0f;
    innovationStdW = 0.0f;
    innovationMadW = 0.0f;
    innovationMedianAbsDeltaRateW = 0.0f;
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
    return;
  }

  trendScore = computeTrendScore(x_pos);
  jitterScore = computeJitterScore(trendScore);
  opticalRisk = computeOpticalRisk();
  updateRestMode();
}

void kalmanPredict(float dtSeconds) {
  if (!kalmanInitialized) return;

  float velLeak = restMode ? leakForDt(velLeakRestRef, dtSeconds)
                           : leakForDt(velLeakBaseRef, dtSeconds);
  float pred_v = velLeak * x_vel;
  pred_mm = x_pos + dtSeconds * pred_v;

  if (restMode) {
    qPosUsed = QPOS_REST_RATE * dtSeconds;
    qVelUsed = QVEL_REST_RATE * dtSeconds;
  } else {
    float qPosMaxUsed = qMaxScale * QPOS_MAX_RATE * dtSeconds;
    qPosUsed = lerpf(QPOS_MIN_RATE * dtSeconds, qPosMaxUsed, trendScore);
    qVelUsed = lerpf(QVEL_LOW_RATE * dtSeconds, QVEL_HIGH_RATE * dtSeconds, trendScore);
  }

  float a12 = dtSeconds * velLeak;
  float a22 = velLeak;

  float P11Pred = P11 + a12 * (P12 + P21) + a12 * a12 * P22 + qPosUsed;
  float P12Pred = a22 * P12 + a12 * a22 * P22;
  float P21Pred = a22 * P21 + a12 * a22 * P22;
  float P22Pred = a22 * a22 * P22 + qVelUsed;

  P11 = P11Pred;
  P12 = P12Pred;
  P21 = P21Pred;
  P22 = P22Pred;

  x_pos = pred_mm;
  x_vel = pred_v;
  applyRestVelocityDrain(0.0f, false);

  actual_mm = x_pos;
  actual_v = x_vel;
  measurementUsed = 0;
  innovationUsed = 0.0f;
  innovationNormUsed = 0.0f;
  rEffUsed = R0_BASE * 25.0f;
}

void kalmanCorrect(float z) {
  if (!kalmanInitialized) {
    seedEstimator(z);
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

  float Spre = P11 + Rpre;
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

  float S = P11 + rEffUsed;
  float K1 = P11 / max(S, 1e-6f);
  float K2 = P21 / max(S, 1e-6f);

  x_pos = pred_mm + K1 * innovation;
  x_vel = actual_v + K2 * innovation;
  applyRestVelocityDrain(innovationAbsPre, true);

  float oldP11 = P11;
  float oldP12 = P12;
  float oldP21 = P21;
  float oldP22 = P22;

  P11 = (1.0f - K1) * oldP11;
  P12 = (1.0f - K1) * oldP12;
  P21 = oldP21 - K2 * oldP11;
  P22 = oldP22 - K2 * oldP12;

  actual_mm = x_pos;
  actual_v = x_vel;
  measurementUsed = 1;
}

}  // namespace

void leituraSensorVL53L0XSetSensorType(uint8_t sensorType) {
  if (sensorType < LEITURA_SENSOR_VL53L0X_AGGRESSIVE || sensorType > LEITURA_SENSOR_VL53L0X_CONSERVATIVE) {
    currentSensorType = LEITURA_SENSOR_VL53L0X_MEDIUM;
  } else {
    currentSensorType = sensorType;
  }
  applySensorProfile();
}

uint8_t leituraSensorVL53L0XGetSensorType() {
  return currentSensorType;
}

uint32_t leituraSensorVL53L0XGetTimingBudgetUs() {
  return currentTimingBudgetUs;
}

uint32_t leituraSensorVL53L0XGetI2CClockHz() {
  return currentI2cClockHz;
}

bool leituraSensorVL53L0XBegin(uint8_t sdaPin, uint8_t sclPin) {
  sensorInitialized = false;
  streamStarted = false;

  applySensorProfile();

  Wire.begin(sdaPin, sclPin);
  Wire.setClock(currentI2cClockHz);

  bool ok = lox.begin(SENSOR_I2C_ADDR, false, &Wire, currentSenseConfig);
  if (!ok) {
    resetEstimatorInternal(true);
    return false;
  }

  lox.setMeasurementTimingBudgetMicroSeconds(currentTimingBudgetUs);

  if (applyAggressiveVcsel) {
    lox.setVcselPulsePeriod(VL53L0X_VCSEL_PERIOD_PRE_RANGE, 12);
    lox.setVcselPulsePeriod(VL53L0X_VCSEL_PERIOD_FINAL_RANGE, 8);
  }

  if (applyRelaxedChecks) {
    lox.setLimitCheckValue(VL53L0X_CHECKENABLE_SIGNAL_RATE_FINAL_RANGE, (FixPoint1616_t)(0.15f * 65536.0f));
    lox.setLimitCheckValue(VL53L0X_CHECKENABLE_SIGMA_FINAL_RANGE, (FixPoint1616_t)(60.0f * 65536.0f));
    lox.setLimitCheckEnable(VL53L0X_CHECKENABLE_RANGE_IGNORE_THRESHOLD, 1);
  }

  resetEstimatorInternal(true);

  if (!restartContinuousStream()) {
    resetEstimatorInternal(true);
    return false;
  }

  lastPredictMicros = micros();
  lastMeasurementMicros = 0UL;
  sensorInitialized = true;
  return true;
}

bool leituraSensorVL53L0XUpdate(float dtSeconds) {
  if (!sensorInitialized) return false;

  unsigned long nowUs = micros();

  predictDtUsed = computeDtSeconds(nowUs, lastPredictMicros,
                                   isfinite(dtSeconds) ? dtSeconds : DEFAULT_MEAS_DT_S,
                                   MIN_PREDICT_DT_S, MAX_PREDICT_DT_S);

  bool streamOk = pullMeasurementNonBlocking(nowUs);
  measurementAgeSeconds = (lastMeasurementMicros == 0UL)
                            ? 0.0f
                            : (float)(nowUs - lastMeasurementMicros) * 1.0e-6f;

  if (newMeasurement) {
    updateFeaturesFromMeasurement();
  }

  if (!kalmanInitialized) {
    if (newMeasurement && measurementValid && isfinite(raw_mm)) {
      seedEstimator(raw_mm);
    }
    return streamOk;
  }

  kalmanPredict(predictDtUsed);

  if (newMeasurement && measurementValid) {
    kalmanCorrect(raw_mm);
  }

  return streamOk;
}

float leituraSensorVL53L0XGetRawMm() {
  return raw_mm;
}

float leituraSensorVL53L0XGetFilteredMm() {
  return actual_mm;
}

float leituraSensorVL53L0XGetVelocityMmS() {
  return actual_v;
}

float leituraSensorVL53L0XGetSignalMcps() {
  return signal_mcps;
}

float leituraSensorVL53L0XGetAmbientMcps() {
  return ambient_mcps;
}

float leituraSensorVL53L0XGetSpadEff() {
  return spad_eff;
}

float leituraSensorVL53L0XGetPredictDtSeconds() {
  return predictDtUsed;
}

float leituraSensorVL53L0XGetMeasurementDtSeconds() {
  return measurementDtUsed;
}

float leituraSensorVL53L0XGetMeasurementRateHz() {
  return measurementRateHz;
}

float leituraSensorVL53L0XGetMeasurementAgeSeconds() {
  return measurementAgeSeconds;
}

int leituraSensorVL53L0XGetRangeStatus() {
  return range_status;
}

bool leituraSensorVL53L0XHasNewMeasurement() {
  return newMeasurement;
}

bool leituraSensorVL53L0XMeasurementValid() {
  return measurementValid;
}

bool leituraSensorVL53L0XIsInitialized() {
  return sensorInitialized;
}

LeituraSensorVL53L0XData leituraSensorVL53L0XGetData() {
  LeituraSensorVL53L0XData data;
  data.raw_mm = raw_mm;
  data.filtered_mm = actual_mm;
  data.velocity_mm_s = actual_v;
  data.signal_mcps = signal_mcps;
  data.ambient_mcps = ambient_mcps;
  data.spad_eff = spad_eff;
  data.predict_dt_s = predictDtUsed;
  data.measurement_dt_s = measurementDtUsed;
  data.measurement_rate_hz = measurementRateHz;
  data.measurement_age_s = measurementAgeSeconds;
  data.timing_budget_us = currentTimingBudgetUs;
  data.i2c_clock_hz = currentI2cClockHz;
  data.sensor_type = currentSensorType;
  data.range_status = range_status;
  data.measurement_valid = measurementValid;
  data.new_measurement = newMeasurement;
  data.rest_mode = restMode;
  data.initialized = sensorInitialized;
  return data;
}

void leituraSensorVL53L0XResetEstimator() {
  resetEstimatorInternal(true);
}
