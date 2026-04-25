#include "Adafruit_VL53L0X.h"

// Cria o objeto do sensor VL53L0X
Adafruit_VL53L0X lox = Adafruit_VL53L0X();

const int pinPWM = 27;
const int pinEnable = 14;
const int pinDirecao = 26;

//CONSTANTES
float referencia = 75;  //mm
float Kp = 40;
float Kd = 0;
float Ki = 0;  //2;

unsigned long tempo_inicio = 0;
bool inicializado = false;

float soma_y = 0;
int contador = 0;

// Variáveis iniciais
float integral = 0.0;
float erro_anterior = 0.0;
float erro = 0.0;
float y = 0.0;
float u = 0.0;
int dutyCycle = 0;  //pwm
float y_anterior = 0.0;

float dt = 40; // ms
float currentT = 0.0, previousT = 0.0;  // Elapsed time in loop() function


void lerSerial() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');

    float kp, ki, kd;

    // tenta ler no formato [Kp,Ki,Kd]
    if (sscanf(cmd.c_str(), "[%f,%f,%f]", &kp, &ki, &kd) == 3) {
      Kp = kp;
      Ki = ki;
      Kd = kd;

      integral = 0;  // importante

      // Serial.println("PID atualizado:");
      // Serial.print("Kp: ");
      Serial.println(Kp);
      // Serial.print("Ki: ");
      Serial.println(Ki);
      // Serial.print("Kd: ");
      Serial.println(Kd);

      // Reseta a execução
      inicializado = false;
      tempo_inicio = millis();

    } else {
      Serial.println("Formato inválido. Use: [Kp,Ki,Kd]");
    }
  }
}

void setup() {
  // Inicialização do Serial
  Serial.begin(115200);
  delay(1000);  // tempo para o monitor serial abrir

  Serial.println("Teste com VL53L0X e Wemos D1 R32");

  // Inicializa o sensor
  if (!lox.begin()) {
    Serial.println(F("Falha ao iniciar o VL53L0X"));
    while (1)
      ;  // trava aqui se falhar
  }

  lerSerial();

  // Configura o pino com frequência e resolução, em vez de ledcSetup/AttachPin
  
  int frequencia = 30000;
  int resolucao = 8;
  ledcAttach(pinPWM, frequencia, resolucao);

  tempo_inicio = millis();  // começa contagem
}

void loop() {

    currentT = millis();
    if ((currentT - previousT) >= dt) {
    previousT = currentT;

    // lerSerial();

    VL53L0X_RangingMeasurementData_t measure;

    // Obtém uma medição do sensor
    lox.rangingTest(&measure, false);

    y_anterior = y;

    float y_anterior = y;
    float alpha = 0.5;
    y = alpha * measure.RangeMilliMeter + (1 - alpha) * y_anterior;

    erro_anterior = erro;

    // Calcula o erro naquele momento
    erro = referencia - y;

    // Incrementa a integral
    integral += erro * dt;

    float integral_erro = Ki * (integral);
    float proporcional = Kp * erro;
    float derivativo = Kd * (y_anterior - y) / dt;

    // Calcula o sinal de controle
    u = proporcional + integral_erro + derivativo;

    u = constrain(u, 0, 255);


    // PID
    // Serial.print("y: ");
    Serial.print(y);
    Serial.print(" ");
    // Serial.print("  erro : ");
    Serial.print(erro);
    Serial.print(" ");

    // Serial.print(" | P: ");
    Serial.print(proporcional);
    Serial.print(" ");
    // Serial.print(" | I: ");
    Serial.print(integral_erro);
    Serial.print(" ");
    // Serial.print(" | D: ");
    Serial.print(derivativo);
    Serial.print(" ");
    Serial.print(referencia);
    Serial.print(" ");
    // Serial.print(" | u: ");
    Serial.println(u);

    // aplica PWM continuamente
    ledcWrite(pinPWM, int(u));
  }
}
