#ifndef LEITURA_SENSOR_H
#define LEITURA_SENSOR_H

#include <Arduino.h>

enum LeituraSensorVL53L0XSensorType : uint8_t {
  LEITURA_SENSOR_VL53L0X_AGGRESSIVE = 1,
  LEITURA_SENSOR_VL53L0X_MEDIUM = 2,
  LEITURA_SENSOR_VL53L0X_CONSERVATIVE = 3
};

struct LeituraSensorVL53L0XData {
  float raw_mm;
  float filtered_mm;
  float velocity_mm_s;
  float signal_mcps;
  float ambient_mcps;
  float spad_eff;
  float predict_dt_s;
  float measurement_dt_s;
  float measurement_rate_hz;
  float measurement_age_s;
  uint32_t timing_budget_us;
  uint32_t i2c_clock_hz;
  uint8_t sensor_type;
  int range_status;
  bool measurement_valid;
  bool new_measurement;
  bool rest_mode;
  bool initialized;
};

void leituraSensorVL53L0XSetSensorType(uint8_t sensorType);
uint8_t leituraSensorVL53L0XGetSensorType();
uint32_t leituraSensorVL53L0XGetTimingBudgetUs();
uint32_t leituraSensorVL53L0XGetI2CClockHz();

bool leituraSensorVL53L0XBegin(uint8_t sdaPin = 21, uint8_t sclPin = 22);
bool leituraSensorVL53L0XUpdate(float dtSeconds);

float leituraSensorVL53L0XGetRawMm();
float leituraSensorVL53L0XGetFilteredMm();
float leituraSensorVL53L0XGetVelocityMmS();
float leituraSensorVL53L0XGetSignalMcps();
float leituraSensorVL53L0XGetAmbientMcps();
float leituraSensorVL53L0XGetSpadEff();
float leituraSensorVL53L0XGetPredictDtSeconds();
float leituraSensorVL53L0XGetMeasurementDtSeconds();
float leituraSensorVL53L0XGetMeasurementRateHz();
float leituraSensorVL53L0XGetMeasurementAgeSeconds();
int leituraSensorVL53L0XGetRangeStatus();
bool leituraSensorVL53L0XHasNewMeasurement();
bool leituraSensorVL53L0XMeasurementValid();
bool leituraSensorVL53L0XIsInitialized();
LeituraSensorVL53L0XData leituraSensorVL53L0XGetData();
void leituraSensorVL53L0XResetEstimator();

#endif
