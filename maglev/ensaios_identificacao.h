#ifndef ENSAIOS_IDENTIFICACAO_H
#define ENSAIOS_IDENTIFICACAO_H

#include <Arduino.h>

// ============================================================
// ENSAIOS DE IDENTIFICACAO - PULSO FIXO PARAMETRIZADO - V34
// ------------------------------------------------------------
// Versao limpa:
// - Sem ToF
// - Sem Kalman
// - Sem Hall filtrado
// - Sem limiar, baseline, pico ou deltas internos
// - Apenas maquina temporal de pulsos parametrizaveis
// ============================================================

enum EstadoIdentificacao : uint8_t {
  IDENT_PARADO = 0,
  IDENT_AGUARDANDO_INICIO = 1,
  IDENT_EXECUTANDO_PULSO = 2,
  IDENT_RECUPERANDO = 3,
  IDENT_FINALIZADO = 4,
  IDENT_ABORTADO = 5
};

struct EntradasIdentificacao {
  unsigned long now_us;
  int pwm_atual;
  uint32_t hall_inf_mv;
  uint32_t hall_sup_mv;
};

struct SaidasIdentificacao {
  bool ativo;
  bool finalizado;
  bool abortado;
  int pwm_cmd;
  EstadoIdentificacao estado;
};

struct PerfilIdentificacao {
  uint8_t repeticoes;              // numero de pulsos
  uint8_t pwm_pulso;               // amplitude do pulso em PWM 0..255
  uint16_t tempo_pulso_ms;         // tempo com PWM=pwm_pulso
  uint16_t tempo_entre_pulsos_ms;  // tempo em PWM=0 apos cada pulso
  uint16_t tempo_pre_inicio_ms;    // tempo em PWM=0 antes do primeiro pulso
};

struct DadosIdentificacao {
  bool ativo;
  bool finalizado;
  bool abortado;
  EstadoIdentificacao estado;

  uint8_t repeticao_atual;
  uint8_t repeticoes_totais;

  int pwm_cmd;
  int pwm_pulso;

  uint16_t tempo_pulso_ms;
  uint16_t tempo_entre_pulsos_ms;
  uint16_t tempo_pre_inicio_ms;

  unsigned long t_estado_us;
  unsigned long t_fase_us;
};

void ensaiosIdentificacaoBegin();
void ensaiosIdentificacaoReset();
void ensaiosIdentificacaoStop();

bool ensaiosIdentificacaoStart(const EntradasIdentificacao& in);
bool ensaiosIdentificacaoStartComPerfil(const EntradasIdentificacao& in, const PerfilIdentificacao& perfil);

void ensaiosIdentificacaoUpdate(const EntradasIdentificacao& in, SaidasIdentificacao& out);
bool ensaiosIdentificacaoEstaAtivo();
DadosIdentificacao ensaiosIdentificacaoGetDados();
PerfilIdentificacao ensaiosIdentificacaoGetPerfilAtivo();

const char* ensaiosIdentificacaoNomeEstado(EstadoIdentificacao estado);

#endif
