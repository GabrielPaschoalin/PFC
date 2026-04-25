#include <Wire.h>
#include <math.h>
#include <Adafruit_VL53L0X.h>

// ============================================================
// MAGLEV - KALMAN ADAPTATIVO V7 PWM FIX + AUTO SWEEP + DIRAC
// ------------------------------------------------------------
// Baseado na V3, preservando:
// - leitura contínua do VL53L0X
// - dt real por micros()
// - janela dinâmica / features / risk scores
// - Kalman adaptativo rápido
//
// Adições da V4/V5:
// - máquina de estados não bloqueante para coarse/fine/hold
// - baseline local por ensaio
// - reset com assentamento mecânico / histerese
// - classificação de ensaios e logging serial
// - proteção imediata em CAPTURE_OR_SATURATION
// - invalidação por opticalRisk / jitterScore
//
// Adição da V6:
// - modo DIRAC(BASE_PWM,PEAK_PWM,PULSE_MS,INTERVAL_MS,SETTLE_MS)
// - base contínua + pico curto não bloqueante
// - telemetria enxuta para Serial Plotter durante DIRAC
//
// Correção desta versão:
// - PWM migrado para LEDC explícito
// - frequência fixada em 1000 Hz
// - resolução fixada em 8 bits
// ============================================================


// ============================================================
// HARDWARE
// ============================================================
const int pinPWM = 27;
const int pinEnable = 14;
const int pinDirecao = 26;
int pwm_valor = 0;
int pwm_manual = 0;

const uint32_t freqPWM = 10000;
const uint8_t resolucaoPWM = 8;
bool pwmAttached = false;

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
// PARÂMETROS IDENTIFICADOS
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


// ============================================================
// AJUSTES DE BANCADA
// ============================================================
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
int currentW = 15;   // atualizado para refletir a bancada recente

float rawBuf[MAX_W];
float signalBuf[MAX_W];
float ambientBuf[MAX_W];
float spadBuf[MAX_W];
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

bool detailedOutput = false;
bool lastMeasurementValid = false;


// ============================================================
// AUTO SWEEP / EXPERIMENTO
// ============================================================
enum AutoState {
  AUTO_IDLE,
  AUTO_PREPARE_BASELINE,
  AUTO_COLLECT_BASELINE,
  AUTO_VALIDATE_BASELINE,
  AUTO_APPLY_PWM,
  AUTO_DEAD_TIME,
  AUTO_PERCEPTION,
  AUTO_STABILIZATION,
  AUTO_CLASSIFY_AND_LOG,
  AUTO_RESET_SETTLE,
  AUTO_ANALYZE_COARSE,
  AUTO_ANALYZE_FINE,
  AUTO_HOLD_PREPARE_BASELINE,
  AUTO_HOLD_COLLECT_BASELINE,
  AUTO_HOLD_VALIDATE_BASELINE,
  AUTO_HOLD_APPLY_PWM,
  AUTO_HOLD_OBSERVE,
  AUTO_FINISHED,
  AUTO_ABORT
};

enum SearchPhase {
  PHASE_NONE,
  PHASE_COARSE,
  PHASE_FINE,
  PHASE_HOLD
};

enum TestClass {
  TC_UNKNOWN,
  TC_NO_EFFECT,
  TC_SMALL_RESPONSE,
  TC_CLEAR_RESPONSE,
  TC_STRONG_PULL,
  TC_CAPTURE_OR_SATURATION,
  TC_UNRELIABLE_TEST
};

struct BaselineAccumulator {
  int n;
  float sum;
  float sumSq;
  float minV;
  float maxV;
  float startValue;
  float lastValue;
  int invalidCount;
  int unreliableCount;
};

struct BaselineStats {
  bool valid;
  int n;
  float mean;
  float std;
  float minV;
  float maxV;
  float range;
  float startValue;
  float endValue;
  int invalidCount;
  int unreliableCount;
};

struct ResponseAccumulator {
  float peakAbsDelta;
  float peakSignedDelta;
  float steadyMean;
  float steadyStd;
  float steadySlope;
  float lastDelta;
  bool responseDetected;
  bool stabilized;
  bool captured;
  bool unreliable;
  unsigned long responseTimeMs;
  unsigned long settleTimeMs;
  int nAll;
  int nSteady;
  int unreliableCount;
  int invalidCount;
  float opticalRiskSum;
  float jitterScoreSum;
  float steadyBuf[32];
};

struct TestResult {
  int testNumber;
  SearchPhase phase;
  int pwmCommand;
  int pwmApplied;
  float baseline;
  float baselineStd;
  float peakAbsDelta;
  float peakSignedDelta;
  float steadyDelta;
  float steadyStd;
  float steadySlope;
  float meanOpticalRisk;
  float meanJitterScore;
  unsigned long responseTimeMs;
  unsigned long settleTimeMs;
  int invalidCount;
  int unreliableCount;
  bool reliable;
  bool captured;
  bool stabilized;
  TestClass klass;
};

struct RangeCandidate {
  int pwmLastNoEffect;
  int pwmFirstResponse;
  int pwmFirstCapture;
  bool valid;
};

const int MAX_COARSE_TESTS = 16;
const int MAX_FINE_TESTS = 64;

const int coarseValuesTemplate[] = {0, 25, 50, 75, 100, 125, 150, 175, 200, 225};
const int coarseCount = sizeof(coarseValuesTemplate) / sizeof(coarseValuesTemplate[0]);

int fineValues[MAX_FINE_TESTS];
int fineCount = 0;

TestResult coarseResults[MAX_COARSE_TESTS];
int coarseResultsCount = 0;

TestResult fineResults[MAX_FINE_TESTS];
int fineResultsCount = 0;

AutoState autoState = AUTO_IDLE;
SearchPhase currentPhase = PHASE_NONE;
bool autoEnabled = false;
unsigned long stateStartMs = 0;
unsigned long testApplyMs = 0;
int currentTestIndex = 0;
int currentAutoPWM = 0;
int chosenEquilibriumPWM = -1;

BaselineAccumulator baselineAcc;
BaselineStats lastBaseline;
ResponseAccumulator responseAcc;
RangeCandidate coarseRange;

// tempos (ms)
unsigned long BASELINE_WINDOW_MS = 1200;
unsigned long DEAD_TIME_MS = 250;
unsigned long PERCEPTION_WINDOW_MS = 900;
unsigned long STABILIZATION_WINDOW_MS = 1200;
unsigned long RESET_SETTLE_MS = 3000;
unsigned long HOLD_TEST_MS = 3000;

// thresholds
float BASELINE_STD_MAX_MM = 0.80f;
float BASELINE_RANGE_MAX_MM = 3.00f;
float DETECT_DELTA_MM = 2.0f;
float CLEAR_RESPONSE_MM = 4.0f;
float STRONG_PULL_MM = 8.0f;
float CAPTURE_DELTA_MM = 12.0f;
float STABLE_SLOPE_MAX_MMPS = 3.0f;
float STABLE_STD_MAX_MM = 1.0f;
float OPTICAL_RISK_MAX_TEST = 0.35f;
float JITTER_SCORE_MAX_TEST = 0.90f;


// ============================================================
// DIRAC / TESTE DE PULSO
// ============================================================
enum DiracState {
  DIRAC_IDLE,
  DIRAC_SETTLE,
  DIRAC_PULSE,
  DIRAC_INTERVAL
};

bool diracEnabled = false;
DiracState diracState = DIRAC_IDLE;
unsigned long diracStateStartMs = 0;
unsigned long diracCycleCounter = 0;

int diracBasePWM = 0;
int diracPeakPWM = 0;
unsigned long diracPulseMs = 100;
unsigned long diracIntervalMs = 1000;
unsigned long diracSettleMs = 1000;
float diracBaseline = NAN;



// ============================================================
// HELPERS
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

const char* autoStateName(AutoState s) {
  switch (s) {
    case AUTO_IDLE: return "IDLE";
    case AUTO_PREPARE_BASELINE: return "PREP_BASE";
    case AUTO_COLLECT_BASELINE: return "COLLECT_BASE";
    case AUTO_VALIDATE_BASELINE: return "VALID_BASE";
    case AUTO_APPLY_PWM: return "APPLY_PWM";
    case AUTO_DEAD_TIME: return "DEAD_TIME";
    case AUTO_PERCEPTION: return "PERCEPTION";
    case AUTO_STABILIZATION: return "STABILIZE";
    case AUTO_CLASSIFY_AND_LOG: return "CLASSIFY";
    case AUTO_RESET_SETTLE: return "RESET";
    case AUTO_ANALYZE_COARSE: return "AN_COARSE";
    case AUTO_ANALYZE_FINE: return "AN_FINE";
    case AUTO_HOLD_PREPARE_BASELINE: return "HOLD_PREP";
    case AUTO_HOLD_COLLECT_BASELINE: return "HOLD_BASE";
    case AUTO_HOLD_VALIDATE_BASELINE: return "HOLD_VALID";
    case AUTO_HOLD_APPLY_PWM: return "HOLD_APPLY";
    case AUTO_HOLD_OBSERVE: return "HOLD_OBS";
    case AUTO_FINISHED: return "FINISHED";
    case AUTO_ABORT: return "ABORT";
    default: return "?";
  }
}

const char* phaseName(SearchPhase p) {
  switch (p) {
    case PHASE_NONE: return "NONE";
    case PHASE_COARSE: return "COARSE";
    case PHASE_FINE: return "FINE";
    case PHASE_HOLD: return "HOLD";
    default: return "?";
  }
}

const char* className(TestClass c) {
  switch (c) {
    case TC_UNKNOWN: return "UNKNOWN";
    case TC_NO_EFFECT: return "NO_EFFECT";
    case TC_SMALL_RESPONSE: return "SMALL_RESPONSE";
    case TC_CLEAR_RESPONSE: return "CLEAR_RESPONSE";
    case TC_STRONG_PULL: return "STRONG_PULL";
    case TC_CAPTURE_OR_SATURATION: return "CAPTURE_OR_SATURATION";
    case TC_UNRELIABLE_TEST: return "UNRELIABLE_TEST";
    default: return "?";
  }
}

const char* diracStateName(DiracState s) {
  switch (s) {
    case DIRAC_IDLE: return "IDLE";
    case DIRAC_SETTLE: return "SETTLE";
    case DIRAC_PULSE: return "PULSE";
    case DIRAC_INTERVAL: return "INTERVAL";
    default: return "?";
  }
}

void logEvent(const char* msg) {
  if (autoEnabled) return;
  Serial.print("EVENT,msg:");
  Serial.print(msg);
  Serial.print(",state:");
  Serial.print(autoStateName(autoState));
  Serial.print(",phase:");
  Serial.println(phaseName(currentPhase));
}

void transitionTo(AutoState nextState, const char* reason = nullptr) {
  autoState = nextState;
  stateStartMs = millis();
  if (!autoEnabled) {
    if (reason != nullptr) {
      Serial.print("TRANSITION,to:");
      Serial.print(autoStateName(nextState));
      Serial.print(",phase:");
      Serial.print(phaseName(currentPhase));
      Serial.print(",reason:");
      Serial.println(reason);
    }
  }
}

void writePWMHardware(int pwm) {
  pwm_valor = constrain(pwm, 0, 255);
  if (pwmAttached) {
    ledcWrite(pinPWM, (uint32_t)pwm_valor);
  }
}

void applyPWMImmediate(int pwm) {
  writePWMHardware(pwm);
}

void clearBuffers() {
  bufHead = 0;
  bufCount = 0;
  for (int i = 0; i < MAX_W; i++) {
    rawBuf[i] = 0.0f;
    signalBuf[i] = 0.0f;
    ambientBuf[i] = 0.0f;
    spadBuf[i] = 0.0f;
    statusBuf[i] = -1;
  }
}

void pushSample(float raw, float signal, float ambient, float spad, int status) {
  rawBuf[bufHead] = raw;
  signalBuf[bufHead] = signal;
  ambientBuf[bufHead] = ambient;
  spadBuf[bufHead] = spad;
  statusBuf[bufHead] = status;
  bufHead = (bufHead + 1) % MAX_W;
  if (bufCount < MAX_W) bufCount++;
}

void resetEstimator(bool clearAllBuffers = true) {
  kalmanInitialized = false;
  x_pos = 0.0f;
  x_vel = 0.0f;

  P11 = 2.0f; P12 = 0.0f;
  P21 = 0.0f; P22 = 1.0f;

  pred_mm = actual_mm = actual_v = NAN;
  meanW = stdW = madW = medianAbsDeltaW = slopeW = residStdW = coherenceW = NAN;
  signalMedW = signalStdW = ambientMedW = ambientStdW = spadMedW = spadStdW = badStatusRatioW = NAN;
  trendScore = jitterScore = opticalRisk = 0.0f;
  qPosUsed = QPOS_MIN_BASE;
  qVelUsed = 0.20f;
  rEffUsed = R0_BASE;
  innovationUsed = 0.0f;
  innovationNormUsed = 0.0f;
  measurementUsed = 0;
  restCounter = 0;
  restMode = false;
  lastMeasurementValid = false;

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
  int statusTmp[MAX_W];

  copyRecentSamples(rawTmp, rawBuf, n);
  copyRecentSamples(signalTmp, signalBuf, n);
  copyRecentSamples(ambientTmp, ambientBuf, n);
  copyRecentSamples(spadTmp, spadBuf, n);
  copyRecentStatus(statusTmp, statusBuf, n);

  meanW = meanArray(rawTmp, n);
  stdW = stdArray(rawTmp, n, meanW);
  madW = madStdFromArray(rawTmp, n);
  medianAbsDeltaW = medianAbsDelta(rawTmp, n);
  coherenceW = directionCoherence(rawTmp, n);

  float interceptDummy = 0.0f;
  linearFit(rawTmp, n, slopeW, interceptDummy, residStdW);

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
  float slopeNorm = 0.0f;
  if (SLOPE_HARD_MM_PER_SAMPLE > SLOPE_SOFT_MM_PER_SAMPLE) {
    slopeNorm = (fabsf(slopeW) - SLOPE_SOFT_MM_PER_SAMPLE) /
                (SLOPE_HARD_MM_PER_SAMPLE - SLOPE_SOFT_MM_PER_SAMPLE);
  }
  slopeNorm = clampf(slopeNorm, 0.0f, 1.0f);

  float coherenceNorm = clampf((coherenceW - 0.55f) / (0.90f - 0.55f), 0.0f, 1.0f);

  float scale = max(2.0f * SIGMA_STOP, max(2.0f * madW, 1.0f));
  float offsetNorm = fabsf(meanW - predictedPos) / scale;
  offsetNorm = clampf((offsetNorm - 0.75f) / (2.0f - 0.75f), 0.0f, 1.0f);

  float score = 0.55f * slopeNorm + 0.30f * coherenceNorm + 0.15f * offsetNorm;
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

  float P11Pred = P11 + dt_used * (P21 + P12) + dt_used * dt_used * P22 + qPosUsed;
  float P12Pred = P12 + dt_used * P22;
  float P21Pred = P21 + dt_used * P22;
  float P22Pred = P22 + qVelUsed;

  innovationUsed = 0.0f;
  innovationNormUsed = 0.0f;
  measurementUsed = 0;

  if (!measurementValid) {
    x_pos = pred_mm;
    x_vel = pred_v;
    P11 = P11Pred; P12 = P12Pred;
    P21 = P21Pred; P22 = P22Pred;
    actual_mm = x_pos;
    actual_v = x_vel;
    rEffUsed = R0_BASE * 25.0f;
    return;
  }

  float rFactor = 1.35f + 2.10f * jitterScore + 1.10f * opticalRisk - 0.25f * trendScore;
  rFactor = clampf(rFactor, 1.2f, 20.0f);
  float Rpre = R0_BASE * rFactor;

  float innovation = z - pred_mm;

  if (restMode && fabsf(innovation) < innovationDeadbandRest) {
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
// DIRAC HELPERS
// ============================================================
void stopDirac(const char* reason) {
  (void)reason;
  diracEnabled = false;
  diracState = DIRAC_IDLE;
  diracStateStartMs = millis();
  diracCycleCounter = 0;
  diracBaseline = NAN;
  applyPWMImmediate(0);
  pwm_manual = 0;
}

void startDirac(int basePWM, int peakPWM, unsigned long pulseMs, unsigned long intervalMs, unsigned long settleMs) {
  if (autoEnabled) stopAutoSweep("dirac_start");

  diracBasePWM = constrain(basePWM, 0, 255);
  diracPeakPWM = constrain(peakPWM, 0, 255);
  if (diracPeakPWM < diracBasePWM) diracPeakPWM = diracBasePWM;

  diracPulseMs = max(1UL, pulseMs);
  diracIntervalMs = max(1UL, intervalMs);
  diracSettleMs = max(1UL, settleMs);

  diracEnabled = true;
  diracState = DIRAC_SETTLE;
  diracStateStartMs = millis();
  diracCycleCounter = 0;
  diracBaseline = NAN;

  pwm_manual = diracBasePWM;
  applyPWMImmediate(diracBasePWM);
}

void runDiracStateMachine() {
  if (!diracEnabled) return;

  switch (diracState) {
    case DIRAC_IDLE:
      applyPWMImmediate(0);
      break;

    case DIRAC_SETTLE:
      applyPWMImmediate(diracBasePWM);
      if (millis() - diracStateStartMs >= diracSettleMs) {
        diracBaseline = actual_mm;
        diracState = DIRAC_PULSE;
        diracStateStartMs = millis();
      }
      break;

    case DIRAC_PULSE:
      applyPWMImmediate(diracPeakPWM);
      if (millis() - diracStateStartMs >= diracPulseMs) {
        diracCycleCounter++;
        diracState = DIRAC_INTERVAL;
        diracStateStartMs = millis();
      }
      break;

    case DIRAC_INTERVAL:
      applyPWMImmediate(diracBasePWM);
      if (millis() - diracStateStartMs >= diracIntervalMs) {
        diracBaseline = actual_mm;
        diracState = DIRAC_PULSE;
        diracStateStartMs = millis();
      }
      break;
  }
}

// ============================================================
// AUTO HELPERS
// ============================================================
void clearBaselineAccumulator() {
  baselineAcc.n = 0;
  baselineAcc.sum = 0.0f;
  baselineAcc.sumSq = 0.0f;
  baselineAcc.minV = 1e9f;
  baselineAcc.maxV = -1e9f;
  baselineAcc.startValue = NAN;
  baselineAcc.lastValue = NAN;
  baselineAcc.invalidCount = 0;
  baselineAcc.unreliableCount = 0;
}

void updateBaselineAccumulator() {
  if (!isfinite(actual_mm)) {
    baselineAcc.invalidCount++;
    return;
  }

  if (baselineAcc.n == 0) baselineAcc.startValue = actual_mm;
  baselineAcc.lastValue = actual_mm;
  baselineAcc.sum += actual_mm;
  baselineAcc.sumSq += actual_mm * actual_mm;
  if (actual_mm < baselineAcc.minV) baselineAcc.minV = actual_mm;
  if (actual_mm > baselineAcc.maxV) baselineAcc.maxV = actual_mm;
  baselineAcc.n++;

  if (!lastMeasurementValid) baselineAcc.invalidCount++;
  if (opticalRisk > OPTICAL_RISK_MAX_TEST || jitterScore > JITTER_SCORE_MAX_TEST) baselineAcc.unreliableCount++;
}

BaselineStats finalizeBaselineAccumulator() {
  BaselineStats b;
  b.valid = false;
  b.n = baselineAcc.n;
  b.mean = NAN;
  b.std = NAN;
  b.minV = baselineAcc.minV;
  b.maxV = baselineAcc.maxV;
  b.range = NAN;
  b.startValue = baselineAcc.startValue;
  b.endValue = baselineAcc.lastValue;
  b.invalidCount = baselineAcc.invalidCount;
  b.unreliableCount = baselineAcc.unreliableCount;

  if (baselineAcc.n < 3) return b;

  b.mean = baselineAcc.sum / (float)baselineAcc.n;
  float var = (baselineAcc.sumSq - baselineAcc.sum * baselineAcc.sum / (float)baselineAcc.n) / max(1, baselineAcc.n - 1);
  if (var < 0.0f) var = 0.0f;
  b.std = sqrtf(var);
  b.range = baselineAcc.maxV - baselineAcc.minV;
  b.valid = true;
  return b;
}

bool baselineLooksStable(const BaselineStats &b) {
  if (!b.valid) return false;
  if (!isfinite(b.mean) || !isfinite(b.std)) return false;
  if (b.invalidCount > 1) return false;
  if (b.unreliableCount > 1) return false;
  if (b.std > BASELINE_STD_MAX_MM) return false;
  if (b.range > BASELINE_RANGE_MAX_MM) return false;
  return true;
}

void clearResponseAccumulator() {
  responseAcc.peakAbsDelta = 0.0f;
  responseAcc.peakSignedDelta = 0.0f;
  responseAcc.steadyMean = NAN;
  responseAcc.steadyStd = NAN;
  responseAcc.steadySlope = NAN;
  responseAcc.lastDelta = 0.0f;
  responseAcc.responseDetected = false;
  responseAcc.stabilized = false;
  responseAcc.captured = false;
  responseAcc.unreliable = false;
  responseAcc.responseTimeMs = 0;
  responseAcc.settleTimeMs = 0;
  responseAcc.nAll = 0;
  responseAcc.nSteady = 0;
  responseAcc.unreliableCount = 0;
  responseAcc.invalidCount = 0;
  responseAcc.opticalRiskSum = 0.0f;
  responseAcc.jitterScoreSum = 0.0f;
  for (int i = 0; i < 32; i++) responseAcc.steadyBuf[i] = 0.0f;
}

void updateResponseAccumulator(const BaselineStats &b, bool feedSteadyBuffer) {
  if (!isfinite(actual_mm)) {
    responseAcc.invalidCount++;
    return;
  }

  float signedDelta = actual_mm - b.mean;
  float absDelta = fabsf(signedDelta);

  responseAcc.lastDelta = signedDelta;
  if (absDelta > responseAcc.peakAbsDelta) {
    responseAcc.peakAbsDelta = absDelta;
    responseAcc.peakSignedDelta = signedDelta;
  }

  if (!responseAcc.responseDetected && absDelta >= DETECT_DELTA_MM) {
    responseAcc.responseDetected = true;
    responseAcc.responseTimeMs = millis() - testApplyMs;
  }

  if (absDelta >= CAPTURE_DELTA_MM) {
    responseAcc.captured = true;
  }

  if (!lastMeasurementValid) responseAcc.invalidCount++;

  if (opticalRisk > OPTICAL_RISK_MAX_TEST || jitterScore > JITTER_SCORE_MAX_TEST) {
    responseAcc.unreliable = true;
    responseAcc.unreliableCount++;
  }

  responseAcc.opticalRiskSum += opticalRisk;
  responseAcc.jitterScoreSum += jitterScore;
  responseAcc.nAll++;

  if (feedSteadyBuffer && responseAcc.nSteady < 32) {
    responseAcc.steadyBuf[responseAcc.nSteady++] = actual_mm;
  }
}

void finalizeResponseSteadyStats() {
  if (responseAcc.nSteady < 3) {
    responseAcc.steadyMean = actual_mm;
    responseAcc.steadyStd = 999.0f;
    responseAcc.steadySlope = 999.0f;
    responseAcc.stabilized = false;
    return;
  }

  responseAcc.steadyMean = meanArray(responseAcc.steadyBuf, responseAcc.nSteady);
  responseAcc.steadyStd = stdArray(responseAcc.steadyBuf, responseAcc.nSteady, responseAcc.steadyMean);

  float intercept = 0.0f;
  float resid = 0.0f;
  linearFit(responseAcc.steadyBuf, responseAcc.nSteady, responseAcc.steadySlope, intercept, resid);
  responseAcc.steadySlope = responseAcc.steadySlope / max(dt_used, 1e-3f); // mm/s

  responseAcc.stabilized = (fabsf(responseAcc.steadySlope) <= STABLE_SLOPE_MAX_MMPS &&
                            responseAcc.steadyStd <= STABLE_STD_MAX_MM);

  if (responseAcc.stabilized) {
    responseAcc.settleTimeMs = millis() - testApplyMs;
  }
}

TestClass classifyCurrentTest(const BaselineStats &b) {
  if (responseAcc.unreliable) return TC_UNRELIABLE_TEST;
  if (responseAcc.captured) return TC_CAPTURE_OR_SATURATION;

  float steadyDelta = NAN;
  if (isfinite(responseAcc.steadyMean)) steadyDelta = responseAcc.steadyMean - b.mean;

  if (responseAcc.peakAbsDelta >= STRONG_PULL_MM && !responseAcc.stabilized) {
    return TC_STRONG_PULL;
  }

  if (responseAcc.peakAbsDelta >= CLEAR_RESPONSE_MM) {
    if (responseAcc.stabilized) return TC_CLEAR_RESPONSE;
    return TC_STRONG_PULL;
  }

  if (responseAcc.peakAbsDelta >= DETECT_DELTA_MM) {
    return TC_SMALL_RESPONSE;
  }

  (void)steadyDelta;
  return TC_NO_EFFECT;
}

TestResult buildCurrentResult() {
  TestResult r;
  r.testNumber = currentTestIndex + 1;
  r.phase = currentPhase;
  r.pwmCommand = currentAutoPWM;
  r.pwmApplied = currentAutoPWM;
  r.baseline = lastBaseline.mean;
  r.baselineStd = lastBaseline.std;
  r.peakAbsDelta = responseAcc.peakAbsDelta;
  r.peakSignedDelta = responseAcc.peakSignedDelta;
  r.steadyDelta = isfinite(responseAcc.steadyMean) ? (responseAcc.steadyMean - lastBaseline.mean) : NAN;
  r.steadyStd = responseAcc.steadyStd;
  r.steadySlope = responseAcc.steadySlope;
  r.meanOpticalRisk = (responseAcc.nAll > 0) ? (responseAcc.opticalRiskSum / (float)responseAcc.nAll) : NAN;
  r.meanJitterScore = (responseAcc.nAll > 0) ? (responseAcc.jitterScoreSum / (float)responseAcc.nAll) : NAN;
  r.responseTimeMs = responseAcc.responseTimeMs;
  r.settleTimeMs = responseAcc.settleTimeMs;
  r.invalidCount = responseAcc.invalidCount;
  r.unreliableCount = responseAcc.unreliableCount;
  r.captured = responseAcc.captured;
  r.stabilized = responseAcc.stabilized;
  r.klass = classifyCurrentTest(lastBaseline);
  r.reliable = (r.klass != TC_UNRELIABLE_TEST && r.invalidCount <= 2);
  return r;
}

void logTestResult(const TestResult &r) {
  Serial.print("TEST=");
  Serial.print(r.testNumber);
  Serial.print(";PHASE=");
  Serial.print(phaseName(r.phase));
  Serial.print(";PWM_CMD=");
  Serial.print(r.pwmCommand);
  Serial.print(";PWM_APPLIED=");
  Serial.print(r.pwmApplied);
  Serial.print(";BASELINE=");
  Serial.print(r.baseline, 3);
  Serial.print(";BASE_STD=");
  Serial.print(r.baselineStd, 3);
  Serial.print(";PEAK_DELTA=");
  Serial.print(r.peakAbsDelta, 3);
  Serial.print(";PEAK_SIGNED=");
  Serial.print(r.peakSignedDelta, 3);
  Serial.print(";STEADY_DELTA=");
  Serial.print(r.steadyDelta, 3);
  Serial.print(";STEADY_STD=");
  Serial.print(r.steadyStd, 3);
  Serial.print(";STEADY_SLOPE=");
  Serial.print(r.steadySlope, 3);
  Serial.print(";RESP_MS=");
  Serial.print(r.responseTimeMs);
  Serial.print(";SETTLE_MS=");
  Serial.print(r.settleTimeMs);
  Serial.print(";FINAL=");
  Serial.println(className(r.klass));
}

void saveCurrentResult(const TestResult &r) {
  if (r.phase == PHASE_COARSE) {
    if (coarseResultsCount < MAX_COARSE_TESTS) coarseResults[coarseResultsCount++] = r;
  } else if (r.phase == PHASE_FINE) {
    if (fineResultsCount < MAX_FINE_TESTS) fineResults[fineResultsCount++] = r;
  }
}

void buildFineSearchRange() {
  coarseRange.valid = false;
  coarseRange.pwmLastNoEffect = -1;
  coarseRange.pwmFirstResponse = -1;
  coarseRange.pwmFirstCapture = -1;

  for (int i = 0; i < coarseResultsCount; i++) {
    const TestResult &r = coarseResults[i];
    if (!r.reliable) continue;

    if (r.klass == TC_NO_EFFECT) {
      coarseRange.pwmLastNoEffect = r.pwmCommand;
    }

    if (coarseRange.pwmFirstResponse < 0 &&
        (r.klass == TC_SMALL_RESPONSE || r.klass == TC_CLEAR_RESPONSE || r.klass == TC_STRONG_PULL)) {
      coarseRange.pwmFirstResponse = r.pwmCommand;
    }

    if (coarseRange.pwmFirstCapture < 0 &&
        (r.klass == TC_STRONG_PULL || r.klass == TC_CAPTURE_OR_SATURATION)) {
      coarseRange.pwmFirstCapture = r.pwmCommand;
    }
  }

  if (coarseRange.pwmFirstResponse < 0) return;

  int lo = (coarseRange.pwmLastNoEffect >= 0) ? coarseRange.pwmLastNoEffect : max(0, coarseRange.pwmFirstResponse - 25);
  int hi = (coarseRange.pwmFirstCapture >= 0) ? coarseRange.pwmFirstCapture : min(255, coarseRange.pwmFirstResponse + 25);

  if (hi <= lo) return;

  fineCount = 0;
  for (int pwm = lo; pwm <= hi && fineCount < MAX_FINE_TESTS; pwm += 3) {
    fineValues[fineCount++] = pwm;
  }

  if (fineCount > 0) coarseRange.valid = true;
}

int chooseBestEquilibriumPWM() {
  int bestPWM = -1;
  float bestScore = -1e9f;

  for (int i = 0; i < fineResultsCount; i++) {
    const TestResult &r = fineResults[i];
    if (!r.reliable) continue;
    if (r.klass == TC_CAPTURE_OR_SATURATION) continue;
    if (!isfinite(r.steadyDelta)) continue;

    float absSteady = fabsf(r.steadyDelta);
    float score = 0.0f;
    score += 2.4f * absSteady;
    score += (r.stabilized ? 2.0f : -1.0f);
    score -= 0.5f * r.steadyStd;
    score -= 0.25f * fabsf(r.steadySlope);
    score -= 2.5f * r.meanOpticalRisk;
    score -= 2.0f * r.meanJitterScore;
    if (r.klass == TC_CLEAR_RESPONSE) score += 2.0f;
    if (r.klass == TC_SMALL_RESPONSE) score += 0.5f;
    if (r.klass == TC_STRONG_PULL) score -= 3.0f;

    if (score > bestScore) {
      bestScore = score;
      bestPWM = r.pwmCommand;
    }
  }

  return bestPWM;
}

void resetExperimentData() {
  coarseResultsCount = 0;
  fineResultsCount = 0;
  fineCount = 0;
  currentTestIndex = 0;
  currentAutoPWM = 0;
  chosenEquilibriumPWM = -1;
  coarseRange.valid = false;
  coarseRange.pwmLastNoEffect = -1;
  coarseRange.pwmFirstResponse = -1;
  coarseRange.pwmFirstCapture = -1;
  lastBaseline.valid = false;
  clearBaselineAccumulator();
  clearResponseAccumulator();
}

void startAutoSweep() {
  if (!kalmanInitialized) {
    logEvent("Kalman ainda nao inicializado");
    return;
  }
  autoEnabled = true;
  resetExperimentData();
  currentPhase = PHASE_COARSE;
  pwm_manual = 0;
  applyPWMImmediate(0);
  transitionTo(AUTO_PREPARE_BASELINE, "auto_start");
}

void stopAutoSweep(const char* reason) {
  autoEnabled = false;
  currentPhase = PHASE_NONE;
  applyPWMImmediate(0);
  transitionTo(AUTO_IDLE, reason);
}

bool currentPhaseFinished() {
  if (currentPhase == PHASE_COARSE) return currentTestIndex >= coarseCount;
  if (currentPhase == PHASE_FINE) return currentTestIndex >= fineCount;
  return true;
}

int currentPhasePWMAtIndex(int idx) {
  if (currentPhase == PHASE_COARSE) {
    if (idx < 0 || idx >= coarseCount) return 0;
    return coarseValuesTemplate[idx];
  }
  if (currentPhase == PHASE_FINE) {
    if (idx < 0 || idx >= fineCount) return 0;
    return fineValues[idx];
  }
  return 0;
}

void runAutoStateMachine() {
  if (!autoEnabled) return;

  switch (autoState) {
    case AUTO_IDLE:
      break;

    case AUTO_PREPARE_BASELINE:
      applyPWMImmediate(0);
      clearBaselineAccumulator();
      clearResponseAccumulator();
      transitionTo(AUTO_COLLECT_BASELINE, "prepare_baseline");
      break;

    case AUTO_COLLECT_BASELINE:
      updateBaselineAccumulator();
      if (millis() - stateStartMs >= BASELINE_WINDOW_MS) {
        transitionTo(AUTO_VALIDATE_BASELINE, "baseline_window_done");
      }
      break;

    case AUTO_VALIDATE_BASELINE:
      lastBaseline = finalizeBaselineAccumulator();
      if (!baselineLooksStable(lastBaseline)) {
        transitionTo(AUTO_ABORT, "baseline_invalid");
        break;
      }
      if (currentPhaseFinished()) {
        if (currentPhase == PHASE_COARSE) transitionTo(AUTO_ANALYZE_COARSE, "coarse_done");
        else if (currentPhase == PHASE_FINE) transitionTo(AUTO_ANALYZE_FINE, "fine_done");
        else transitionTo(AUTO_ABORT, "phase_done_unexpected");
        break;
      }
      currentAutoPWM = currentPhasePWMAtIndex(currentTestIndex);
      transitionTo(AUTO_APPLY_PWM, "baseline_ok");
      break;

    case AUTO_APPLY_PWM:
      clearResponseAccumulator();
      testApplyMs = millis();
      applyPWMImmediate(currentAutoPWM);
      transitionTo(AUTO_DEAD_TIME, "pwm_applied");
      break;

    case AUTO_DEAD_TIME:
      updateResponseAccumulator(lastBaseline, false);
      if (responseAcc.captured) {
        applyPWMImmediate(0);
        transitionTo(AUTO_CLASSIFY_AND_LOG, "capture_in_dead_time");
        break;
      }
      if (millis() - stateStartMs >= DEAD_TIME_MS) {
        transitionTo(AUTO_PERCEPTION, "dead_done");
      }
      break;

    case AUTO_PERCEPTION:
      updateResponseAccumulator(lastBaseline, false);
      if (responseAcc.captured) {
        applyPWMImmediate(0);
        transitionTo(AUTO_CLASSIFY_AND_LOG, "capture_in_perception");
        break;
      }
      if (millis() - stateStartMs >= PERCEPTION_WINDOW_MS) {
        transitionTo(AUTO_STABILIZATION, "perception_done");
      }
      break;

    case AUTO_STABILIZATION:
      updateResponseAccumulator(lastBaseline, true);
      if (responseAcc.captured) {
        applyPWMImmediate(0);
        transitionTo(AUTO_CLASSIFY_AND_LOG, "capture_in_stabilization");
        break;
      }
      if (millis() - stateStartMs >= STABILIZATION_WINDOW_MS) {
        finalizeResponseSteadyStats();
        transitionTo(AUTO_CLASSIFY_AND_LOG, "stabilization_done");
      }
      break;

    case AUTO_CLASSIFY_AND_LOG: {
      applyPWMImmediate(0);
      if (!isfinite(responseAcc.steadyMean)) finalizeResponseSteadyStats();
      TestResult r = buildCurrentResult();
      logTestResult(r);
      saveCurrentResult(r);
      transitionTo(AUTO_RESET_SETTLE, "result_logged");
      break;
    }

    case AUTO_RESET_SETTLE:
      applyPWMImmediate(0);
      if (millis() - stateStartMs >= RESET_SETTLE_MS) {
        currentTestIndex++;
        transitionTo(AUTO_PREPARE_BASELINE, "next_test");
      }
      break;

    case AUTO_ANALYZE_COARSE:
      buildFineSearchRange();
      if (!coarseRange.valid) {
        transitionTo(AUTO_ABORT, "coarse_no_valid_range");
        break;
      }
      Serial.print("SUMMARY;PHASE=COARSE;LAST_NO_EFFECT=");
      Serial.print(coarseRange.pwmLastNoEffect);
      Serial.print(";FIRST_RESPONSE=");
      Serial.print(coarseRange.pwmFirstResponse);
      Serial.print(";FIRST_CAPTURE=");
      Serial.print(coarseRange.pwmFirstCapture);
      Serial.print(";FINE_COUNT=");
      Serial.println(fineCount);

      currentPhase = PHASE_FINE;
      currentTestIndex = 0;
      transitionTo(AUTO_PREPARE_BASELINE, "start_fine");
      break;

    case AUTO_ANALYZE_FINE:
      chosenEquilibriumPWM = chooseBestEquilibriumPWM();
      if (chosenEquilibriumPWM < 0) {
        transitionTo(AUTO_ABORT, "fine_no_candidate");
        break;
      }
      Serial.print("SUMMARY;PHASE=FINE;BEST_PWM=");
      Serial.println(chosenEquilibriumPWM);
      currentPhase = PHASE_HOLD;
      transitionTo(AUTO_HOLD_PREPARE_BASELINE, "start_hold");
      break;

    case AUTO_HOLD_PREPARE_BASELINE:
      applyPWMImmediate(0);
      clearBaselineAccumulator();
      clearResponseAccumulator();
      transitionTo(AUTO_HOLD_COLLECT_BASELINE, "hold_prepare");
      break;

    case AUTO_HOLD_COLLECT_BASELINE:
      updateBaselineAccumulator();
      if (millis() - stateStartMs >= BASELINE_WINDOW_MS) {
        transitionTo(AUTO_HOLD_VALIDATE_BASELINE, "hold_base_done");
      }
      break;

    case AUTO_HOLD_VALIDATE_BASELINE:
      lastBaseline = finalizeBaselineAccumulator();
      if (!baselineLooksStable(lastBaseline)) {
        transitionTo(AUTO_ABORT, "hold_baseline_invalid");
        break;
      }
      transitionTo(AUTO_HOLD_APPLY_PWM, "hold_baseline_ok");
      break;

    case AUTO_HOLD_APPLY_PWM:
      clearResponseAccumulator();
      currentAutoPWM = chosenEquilibriumPWM;
      testApplyMs = millis();
      applyPWMImmediate(currentAutoPWM);
      transitionTo(AUTO_HOLD_OBSERVE, "hold_pwm_applied");
      break;

    case AUTO_HOLD_OBSERVE:
      updateResponseAccumulator(lastBaseline, true);
      if (responseAcc.captured) {
        applyPWMImmediate(0);
        transitionTo(AUTO_ABORT, "hold_capture");
        break;
      }
      if (responseAcc.unreliable) {
        applyPWMImmediate(0);
        transitionTo(AUTO_ABORT, "hold_unreliable");
        break;
      }
      if (millis() - stateStartMs >= HOLD_TEST_MS) {
        finalizeResponseSteadyStats();
        Serial.print("TEST=FINAL;PHASE=HOLD;PWM_CMD=");
        Serial.print(chosenEquilibriumPWM);
        Serial.print(";PWM_APPLIED=");
        Serial.print(chosenEquilibriumPWM);
        Serial.print(";BASELINE=");
        Serial.print(lastBaseline.mean, 3);
        Serial.print(";BASE_STD=");
        Serial.print(lastBaseline.std, 3);
        Serial.print(";PEAK_DELTA=");
        Serial.print(responseAcc.peakAbsDelta, 3);
        Serial.print(";STEADY_DELTA=");
        Serial.print(responseAcc.steadyMean - lastBaseline.mean, 3);
        Serial.print(";STEADY_STD=");
        Serial.print(responseAcc.steadyStd, 3);
        Serial.print(";STEADY_SLOPE=");
        Serial.print(responseAcc.steadySlope, 3);
        Serial.print(";RESP_MS=");
        Serial.print(responseAcc.responseTimeMs);
        Serial.print(";SETTLE_MS=");
        Serial.print(responseAcc.settleTimeMs);
        Serial.print(";FINAL=");
        Serial.println(responseAcc.stabilized ? "HOLD_OK" : "HOLD_NOT_STABLE");
        transitionTo(AUTO_FINISHED, "hold_done");
      }
      break;

    case AUTO_FINISHED:
      applyPWMImmediate(0);
      Serial.print("SUMMARY;PHASE=FINAL;BEST_PWM=");
      Serial.println(chosenEquilibriumPWM);
      stopAutoSweep("finished");
      break;

    case AUTO_ABORT:
      applyPWMImmediate(0);
      Serial.print("SUMMARY;PHASE=ABORT;BEST_PWM=");
      Serial.println(chosenEquilibriumPWM);
      stopAutoSweep("abort");
      break;
  }
}


// ============================================================
// SERIAL
// ============================================================
void printHelp() {
  Serial.println();
  Serial.println("Comandos manuais:");
  Serial.println("  <numero>     -> PWM manual (0..255) quando auto esta desligado");
  Serial.println("  R            -> reset estimador + buffers");
  Serial.println("  W=15         -> muda janela W (5..21)");
  Serial.println("  T=60         -> intervalo nominal em ms");
  Serial.println("  D=0/1        -> compacta / detalhada");
  Serial.println("  QS=0.25      -> escala do Q maximo em movimento");
  Serial.println("  RF=12        -> piso de R em repouso");
  Serial.println("  VL=0.65      -> leak da velocidade em repouso");
  Serial.println("  DB=0.50      -> deadband da inovacao em repouso");
  Serial.println();
  Serial.println("Comandos auto sweep / dirac:");
  Serial.println("  AUTO         -> inicia coarse -> fine -> hold");
  Serial.println("  DIRAC=(B,P,PM,IM,SM) -> base B, pico P, pulso PM ms, intervalo IM ms, settle SM ms");
  Serial.println("  STOP         -> aborta auto/dirac e zera PWM");
  Serial.println("  STATUS       -> imprime estado atual");
  Serial.println("  BW=1200      -> baseline window ms");
  Serial.println("  DW=250       -> dead time ms");
  Serial.println("  PW=900       -> perception window ms");
  Serial.println("  SW=1200      -> stabilization window ms");
  Serial.println("  RW=3000      -> espera entre testes / reset settle ms");
  Serial.println("  HW=3000      -> hold test ms");
  Serial.println("  TH=2.0       -> limiar de deteccao delta mm");
  Serial.println("  CL=4.0       -> limiar resposta clara mm");
  Serial.println("  ST=8.0       -> limiar strong pull mm");
  Serial.println("  CP=12.0      -> limiar capture mm");
  Serial.println("  OR=0.35      -> opticalRisk max por ensaio");
  Serial.println("  JR=0.90      -> jitterScore max por ensaio");
  Serial.println("  ?            -> ajuda");
  Serial.println();
}

void printStatus() {
  Serial.print("STATUS,auto_enabled:");
  Serial.print(autoEnabled ? 1 : 0);
  Serial.print(",auto_state:");
  Serial.print(autoStateName(autoState));
  Serial.print(",phase:");
  Serial.print(phaseName(currentPhase));
  Serial.print(",manual_pwm:");
  Serial.print(pwm_manual);
  Serial.print(",current_pwm:");
  Serial.print(pwm_valor);
  Serial.print(",current_test_index:");
  Serial.print(currentTestIndex);
  Serial.print(",chosen_pwm:");
  Serial.print(chosenEquilibriumPWM);
  Serial.print(",dirac_base:");
  Serial.print(diracBasePWM);
  Serial.print(",dirac_peak:");
  Serial.print(diracPeakPWM);
  Serial.print(",dirac_pulse_ms:");
  Serial.print(diracPulseMs);
  Serial.print(",dirac_interval_ms:");
  Serial.print(diracIntervalMs);
  Serial.print(",dirac_settle_ms:");
  Serial.print(diracSettleMs);
  Serial.print(",coarse_results:");
  Serial.print(coarseResultsCount);
  Serial.print(",fine_results:");
  Serial.print(fineResultsCount);
  Serial.print(",W:");
  Serial.print(currentW);
  Serial.print(",dt:");
  Serial.print(dt_used, 4);
  Serial.print(",actual_mm:");
  Serial.print(actual_mm, 3);
  Serial.print(",rest:");
  Serial.print(restMode ? 1 : 0);
  Serial.print(",trend:");
  Serial.print(trendScore, 4);
  Serial.print(",jitter:");
  Serial.print(jitterScore, 4);
  Serial.print(",optical:");
  Serial.println(opticalRisk, 4);
}

void processSerialCommand() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.equalsIgnoreCase("R") || cmd.equalsIgnoreCase("RESET")) {
    resetEstimator(true);
    if (autoEnabled) stopAutoSweep("manual_reset");
    if (diracEnabled) stopDirac("manual_reset");
    return;
  }

  if (cmd.equals("?") || cmd.equalsIgnoreCase("HELP")) {
    printHelp();
    return;
  }

  if (cmd.equalsIgnoreCase("STATUS")) {
    printStatus();
    return;
  }

  if (cmd.equalsIgnoreCase("AUTO")) {
    if (diracEnabled) stopDirac("auto_start");
    startAutoSweep();
    return;
  }

  if (cmd.equalsIgnoreCase("STOP") || cmd.equalsIgnoreCase("MANUAL")) {
    if (autoEnabled) stopAutoSweep("manual_stop");
    if (diracEnabled) stopDirac("manual_stop");
    pwm_manual = 0;
    return;
  }

  if (cmd.startsWith("DIRAC=") || cmd.startsWith("dirac=")) {
    int open = cmd.indexOf('(');
    int close = cmd.lastIndexOf(')');
    if (open >= 0 && close > open) {
      String payload = cmd.substring(open + 1, close);
      int vals[5] = {0,0,0,0,0};
      int idx = 0;
      int startPos = 0;
      while (idx < 5) {
        int comma = payload.indexOf(',', startPos);
        String token = (comma >= 0) ? payload.substring(startPos, comma) : payload.substring(startPos);
        token.trim();
        vals[idx++] = token.toInt();
        if (comma < 0) break;
        startPos = comma + 1;
      }
      if (idx == 5) {
        startDirac(vals[0], vals[1], (unsigned long)max(1, vals[2]), (unsigned long)max(1, vals[3]), (unsigned long)max(1, vals[4]));
      }
    }
    return;
  }

  if (cmd.startsWith("W=") || cmd.startsWith("w=")) {
    int newW = cmd.substring(2).toInt();
    if (newW < 5) newW = 5;
    if (newW > MAX_W) newW = MAX_W;
    currentW = newW;
    return;
  }

  if (cmd.startsWith("T=") || cmd.startsWith("t=")) {
    long newInterval = cmd.substring(2).toInt();
    if (newInterval < 20) newInterval = 20;
    if (newInterval > 500) newInterval = 500;
    intervalo_ms = (unsigned long)newInterval;
    return;
  }

  if (cmd.startsWith("D=") || cmd.startsWith("d=")) {
    int v = cmd.substring(2).toInt();
    detailedOutput = (v != 0);
    return;
  }

  if (cmd.startsWith("QS=") || cmd.startsWith("qs=")) {
    float v = cmd.substring(3).toFloat();
    qMaxScale = clampf(v, 0.10f, 1.00f);
    return;
  }

  if (cmd.startsWith("RF=") || cmd.startsWith("rf=")) {
    float v = cmd.substring(3).toFloat();
    restRFloor = clampf(v, 2.0f, 50.0f);
    return;
  }

  if (cmd.startsWith("VL=") || cmd.startsWith("vl=")) {
    float v = cmd.substring(3).toFloat();
    velLeakRest = clampf(v, 0.10f, 0.98f);
    return;
  }

  if (cmd.startsWith("DB=") || cmd.startsWith("db=")) {
    float v = cmd.substring(3).toFloat();
    innovationDeadbandRest = clampf(v, 0.0f, 3.0f);
    return;
  }

  if (cmd.startsWith("BW=") || cmd.startsWith("bw=")) {
    unsigned long v = (unsigned long)max(200L, cmd.substring(3).toInt());
    BASELINE_WINDOW_MS = v;
    return;
  }

  if (cmd.startsWith("DW=") || cmd.startsWith("dw=")) {
    unsigned long v = (unsigned long)max(50L, cmd.substring(3).toInt());
    DEAD_TIME_MS = v;
    return;
  }

  if (cmd.startsWith("PW=") || cmd.startsWith("pw=")) {
    unsigned long v = (unsigned long)max(100L, cmd.substring(3).toInt());
    PERCEPTION_WINDOW_MS = v;
    return;
  }

  if (cmd.startsWith("SW=") || cmd.startsWith("sw=")) {
    unsigned long v = (unsigned long)max(100L, cmd.substring(3).toInt());
    STABILIZATION_WINDOW_MS = v;
    return;
  }

  if (cmd.startsWith("RW=") || cmd.startsWith("rw=")) {
    unsigned long v = (unsigned long)max(100L, cmd.substring(3).toInt());
    RESET_SETTLE_MS = v;
    return;
  }

  if (cmd.startsWith("HW=") || cmd.startsWith("hw=")) {
    unsigned long v = (unsigned long)max(100L, cmd.substring(3).toInt());
    HOLD_TEST_MS = v;
    return;
  }

  if (cmd.startsWith("TH=") || cmd.startsWith("th=")) {
    DETECT_DELTA_MM = clampf(cmd.substring(3).toFloat(), 0.5f, 50.0f);
    return;
  }

  if (cmd.startsWith("CL=") || cmd.startsWith("cl=")) {
    CLEAR_RESPONSE_MM = clampf(cmd.substring(3).toFloat(), DETECT_DELTA_MM, 80.0f);
    return;
  }

  if (cmd.startsWith("ST=") || cmd.startsWith("st=")) {
    STRONG_PULL_MM = clampf(cmd.substring(3).toFloat(), CLEAR_RESPONSE_MM, 100.0f);
    return;
  }

  if (cmd.startsWith("CP=") || cmd.startsWith("cp=")) {
    CAPTURE_DELTA_MM = clampf(cmd.substring(3).toFloat(), STRONG_PULL_MM, 120.0f);
    return;
  }

  if (cmd.startsWith("OR=") || cmd.startsWith("or=")) {
    OPTICAL_RISK_MAX_TEST = clampf(cmd.substring(3).toFloat(), 0.05f, 1.0f);
    return;
  }

  if (cmd.startsWith("JR=") || cmd.startsWith("jr=")) {
    JITTER_SCORE_MAX_TEST = clampf(cmd.substring(3).toFloat(), 0.05f, 1.0f);
    return;
  }

  // comando numérico: PWM manual somente fora do auto
  bool numeric = true;
  int start = 0;
  if (cmd[0] == '-') start = 1;
  for (int i = start; i < cmd.length(); i++) {
    if (!isDigit(cmd[i])) { numeric = false; break; }
  }

  if (numeric) {
    if (!autoEnabled && !diracEnabled) {
      pwm_manual = constrain(cmd.toInt(), 0, 255);
    } else {
      logEvent("PWM manual ignorado: modo automatico ativo");
    }
  }
}


// ============================================================
// TELEMETRIA
// ============================================================
void printCompactTelemetry() {
  Serial.print("raw_mm:");
  Serial.print(raw_mm, 2);
  Serial.print(",");

  Serial.print("pred_mm:");
  Serial.print(pred_mm, 2);
  Serial.print(",");

  Serial.print("actual_mm:");
  Serial.print(actual_mm, 2);
  Serial.print(",");

  Serial.print("actual_v:");
  Serial.print(actual_v, 4);
  Serial.print(",");

  Serial.print("rest:");
  Serial.print(restMode ? 1 : 0);
  Serial.print(",");

  Serial.print("trend:");
  Serial.print(trendScore, 4);
  Serial.print(",");

  Serial.print("jitter:");
  Serial.print(jitterScore, 4);
  Serial.print(",");

  Serial.print("optical:");
  Serial.print(opticalRisk, 4);
  Serial.print(",");

  Serial.print("R_eff:");
  Serial.print(rEffUsed, 4);
  Serial.print(",");

  Serial.print("Q_pos:");
  Serial.print(qPosUsed, 4);
  Serial.print(",");

  Serial.print("dt:");
  Serial.print(dt_used, 4);
  Serial.print(",");

  Serial.print("used:");
  Serial.print(measurementUsed);
  Serial.print(",");

  Serial.print("W:");
  Serial.print(currentW);
  Serial.print(",");

  Serial.print("pwm:");
  Serial.print(pwm_valor);
  Serial.print(",");

  Serial.print("auto:");
  Serial.print(autoEnabled ? 1 : 0);
  Serial.print(",");

  Serial.print("a_state:");
  Serial.print(autoStateName(autoState));
  Serial.print(",");

  Serial.print("phase:");
  Serial.println(phaseName(currentPhase));
}

void printDetailedTelemetry() {
  Serial.print("raw_mm:");
  Serial.print(raw_mm, 2);
  Serial.print(",");

  Serial.print("pred_mm:");
  Serial.print(pred_mm, 2);
  Serial.print(",");

  Serial.print("actual_mm:");
  Serial.print(actual_mm, 2);
  Serial.print(",");

  Serial.print("actual_v:");
  Serial.print(actual_v, 4);
  Serial.print(",");

  Serial.print("rest:");
  Serial.print(restMode ? 1 : 0);
  Serial.print(",");

  Serial.print("mean_w:");
  Serial.print(meanW, 2);
  Serial.print(",");

  Serial.print("std_w:");
  Serial.print(stdW, 3);
  Serial.print(",");

  Serial.print("slope_w:");
  Serial.print(slopeW, 4);
  Serial.print(",");

  Serial.print("coh_w:");
  Serial.print(coherenceW, 4);
  Serial.print(",");

  Serial.print("trend:");
  Serial.print(trendScore, 4);
  Serial.print(",");

  Serial.print("jitter:");
  Serial.print(jitterScore, 4);
  Serial.print(",");

  Serial.print("optical:");
  Serial.print(opticalRisk, 4);
  Serial.print(",");

  Serial.print("R_eff:");
  Serial.print(rEffUsed, 4);
  Serial.print(",");

  Serial.print("Q_pos:");
  Serial.print(qPosUsed, 4);
  Serial.print(",");

  Serial.print("innov:");
  Serial.print(innovationUsed, 3);
  Serial.print(",");

  Serial.print("innov_n:");
  Serial.print(innovationNormUsed, 3);
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

  Serial.print("status:");
  Serial.print(range_status);
  Serial.print(",");

  Serial.print("dt:");
  Serial.print(dt_used, 4);
  Serial.print(",");

  Serial.print("used:");
  Serial.print(measurementUsed);
  Serial.print(",");

  Serial.print("W:");
  Serial.print(currentW);
  Serial.print(",");

  Serial.print("pwm:");
  Serial.print(pwm_valor);
  Serial.print(",");

  Serial.print("auto:");
  Serial.print(autoEnabled ? 1 : 0);
  Serial.print(",");

  Serial.print("a_state:");
  Serial.print(autoStateName(autoState));
  Serial.print(",");

  Serial.print("phase:");
  Serial.println(phaseName(currentPhase));
}

void printGlobalPlotTelemetry() {
  Serial.print("pwm:");
  Serial.print(pwm_valor);
  Serial.print(",");

  Serial.print("raw:");
  Serial.print(raw_mm, 2);
  Serial.print(",");

  Serial.print("actual:");
  Serial.print(actual_mm, 2);
  Serial.print(",");

  Serial.print("scale70:");
  Serial.print(70.0f, 2);
  Serial.print(",");

  Serial.print("scale80:");
  Serial.println(80.0f, 2);
}


// ============================================================
// SETUP / LOOP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(pinPWM, OUTPUT);
  pinMode(pinEnable, OUTPUT);
  pinMode(pinDirecao, OUTPUT);

  digitalWrite(pinEnable, HIGH);
  digitalWrite(pinDirecao, LOW);

  pwmAttached = ledcAttach(pinPWM, freqPWM, resolucaoPWM);
  if (!pwmAttached) {
    Serial.println("erro_pwm:1");
    while (1) {}
  }
  writePWMHardware(pwm_valor);

  // Wire.begin(21, 22);
  Wire.setClock(100000);

  bool ok = lox.begin(0x29, false, &Wire, Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_ACCURACY);
  if (!ok) {
    Serial.println("erro_sensor:1");
    while (1) {}
  }

  lox.setMeasurementTimingBudgetMicroSeconds(50000);

  resetEstimator(true);
  stopAutoSweep("boot");

  Serial.println("pwm:-1,raw:-1,actual:-1,scale70:70,scale80:80");
  printHelp();
}

void loop() {
  unsigned long nowMs = millis();

  if (nowMs - tempoAnteriorMs >= intervalo_ms) {
    tempoAnteriorMs = nowMs;

    unsigned long nowUs = micros();
    if (lastStepMicros == 0) lastStepMicros = nowUs;
    float dtMeas = (nowUs - lastStepMicros) * 1e-6f;
    lastStepMicros = nowUs;
    dt_used = clampf(dtMeas, 0.02f, 0.20f);

    processSerialCommand();

    bool sensorOk = readSensor();
    bool measValid = sensorOk && measurementPhysicallyValid();
    lastMeasurementValid = measValid;

    if (measValid) {
      pushSample(raw_mm, signal_mcps, ambient_mcps, spad_eff, range_status);
    }

    bool enoughWindow = computeWindowFeatures();

    if (!kalmanInitialized) {
      if (measValid && isfinite(raw_mm)) {
        seedEstimator(raw_mm);
      }
      printGlobalPlotTelemetry();
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
      signalMedW = signal_mcps;
      ambientMedW = ambient_mcps;
      spadMedW = spad_eff;
      signalStdW = ambientStdW = spadStdW = 0.0f;
      badStatusRatioW = (range_status == 0) ? 0.0f : 1.0f;
      trendScore = 0.0f;
      jitterScore = 0.0f;
      opticalRisk = (range_status == 0) ? 0.0f : 1.0f;
      restMode = false;
      restCounter = 0;
    } else {
      float predictedPosSimple = x_pos + dt_used * x_vel;
      trendScore = computeTrendScore(predictedPosSimple);
      jitterScore = computeJitterScore(trendScore);
      opticalRisk = computeOpticalRisk();
      updateRestMode();
    }

    kalmanStep(raw_mm, measValid);

    if (autoEnabled) {
      runAutoStateMachine();
      writePWMHardware(pwm_valor);
    } else if (diracEnabled) {
      runDiracStateMachine();
      writePWMHardware(pwm_valor);
    } else {
      pwm_valor = pwm_manual;
      writePWMHardware(pwm_valor);
    }

    printGlobalPlotTelemetry();
  }
}
