// ============================================================
// TESTE PWM EM IN1 (PHASE CHOPPING / SLOW DECAY)
// L298N + ESP32
//
// Ideia:
// - ENA fica sempre em HIGH
// - IN2 fica sempre em LOW
// - O PWM entra em IN1
//
// Comandos no Serial Monitor:
//   auto  -> inicia varredura automática
//   stop  -> interrompe rotina e zera PWM
//   0..255 -> aplica PWM manualmente
// ============================================================

// ------------------- CONFIGURAÇÃO DE PINOS -------------------
// Ajuste estes pinos conforme a sua montagem.
// IMPORTANTE:
//   pinPWM deve estar ligado ao IN1 do L298N
//   pinENA deve estar ligado ao ENA do L298N
//   pinIN2 deve estar ligado ao IN2 do L298N
const int pinPWM = 27;   // AGORA ESTE PINO DEVE IR PARA O IN1
const int pinENA = 14;   // ENA fixo em HIGH
const int pinIN2 = 26;   // IN2 fixo em LOW

// ------------------- PWM -------------------
const int freqPWM = 5000;
const int resolucaoPWM = 8;   // 0 a 255
const int pwmMin = 0;
const int pwmMax = 255;
const int passoPWM = 10;
const unsigned long tempoPorPasso = 10000UL; // 10 s

// ------------------- ESTADO -------------------
bool modoAutomatico = false;
int pwmAtual = 0;
unsigned long instanteUltimaTroca = 0;

// ============================================================
// FUNÇÕES AUXILIARES
// ============================================================
void aplicarPWM(int valor) {
  if (valor < pwmMin) valor = pwmMin;
  if (valor > pwmMax) valor = pwmMax;

  pwmAtual = valor;
  ledcWrite(pinPWM, pwmAtual);

  Serial.print("PWM aplicado: ");
  Serial.println(pwmAtual);
}

void pararRotina() {
  modoAutomatico = false;
  aplicarPWM(0);
  Serial.println("Rotina parada.");
}

void iniciarRotinaAutomatica() {
  modoAutomatico = true;
  pwmAtual = pwmMin;
  aplicarPWM(pwmAtual);
  instanteUltimaTroca = millis();

  Serial.println("Modo automatico iniciado.");
  Serial.println("A cada 10 s o PWM sobe de 10 em 10 ate 255.");
}

void processarComando(String comando) {
  comando.trim();
  comando.toLowerCase();

  if (comando.length() == 0) return;

  if (comando == "auto") {
    iniciarRotinaAutomatica();
    return;
  }

  if (comando == "stop") {
    pararRotina();
    return;
  }

  bool ehNumero = true;
  for (unsigned int i = 0; i < comando.length(); i++) {
    if (!isDigit(comando[i])) {
      ehNumero = false;
      break;
    }
  }

  if (ehNumero) {
    int valor = comando.toInt();
    if (valor < pwmMin || valor > pwmMax) {
      Serial.println("Digite um valor entre 0 e 255.");
      return;
    }

    modoAutomatico = false;
    aplicarPWM(valor);
    Serial.println("Modo manual ativo.");
    return;
  }

  Serial.println("Comando invalido. Use: auto, stop ou um numero entre 0 e 255.");
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // ENA fixo em HIGH
  pinMode(pinENA, OUTPUT);
  digitalWrite(pinENA, HIGH);

  // IN2 fixo em LOW
  pinMode(pinIN2, OUTPUT);
  digitalWrite(pinIN2, LOW);

  // PWM em IN1
  if (!ledcAttach(pinPWM, freqPWM, resolucaoPWM)) {
    Serial.println("Erro ao configurar o PWM no ESP32.");
    while (true) {
      delay(1000);
    }
  }

  aplicarPWM(0);

  Serial.println("Sistema pronto.");
  Serial.println("Ligacao esperada:");
  Serial.println("- pinPWM -> IN1");
  Serial.println("- pinENA -> ENA (sempre HIGH)");
  Serial.println("- pinIN2 -> IN2 (sempre LOW)");
  Serial.println("Comandos: auto | stop | 0..255");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  if (Serial.available()) {
    String comando = Serial.readStringUntil('\n');
    processarComando(comando);
  }

  if (modoAutomatico) {
    unsigned long agora = millis();

    if (agora - instanteUltimaTroca >= tempoPorPasso) {
      if (pwmAtual < pwmMax) {
        int proximoPWM = pwmAtual + passoPWM;
        if (proximoPWM > pwmMax) proximoPWM = pwmMax;

        aplicarPWM(proximoPWM);
        instanteUltimaTroca = agora;
      } else {
        Serial.println("Varredura automatica concluida.");
        modoAutomatico = false;
      }
    }
  }
}
