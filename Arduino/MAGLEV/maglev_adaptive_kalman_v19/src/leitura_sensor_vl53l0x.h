#ifndef LEITURA_SENSOR_VL53L0X_H
#define LEITURA_SENSOR_VL53L0X_H

#include <Arduino.h>

struct LeituraSensorVL53L0XData {
  float raw_mm;
  float filtered_mm;
  bool initialized;
};

bool leituraSensorVL53L0XBegin(uint8_t sdaPin = 21, uint8_t sclPin = 22);
bool leituraSensorVL53L0XUpdate(float dtSeconds);
float leituraSensorVL53L0XGetRawMm();
float leituraSensorVL53L0XGetFilteredMm();
bool leituraSensorVL53L0XIsInitialized();
LeituraSensorVL53L0XData leituraSensorVL53L0XGetData();
void leituraSensorVL53L0XResetEstimator();

#endif
