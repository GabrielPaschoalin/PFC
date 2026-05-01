#ifndef ENSAIOS_IDENTIFICACAO_H
#define ENSAIOS_IDENTIFICACAO_H

#include <Arduino.h>

// ============================================================
// ENSAIOS DE IDENTIFICACAO - PULSO FIXO PARAMETRIZADO - V33
// ------------------------------------------------------------
// Esta versao NAO depende do ToF para iniciar, encontrar limiar
// ou validar resposta. O ToF pode continuar sendo logado pelo .ino,
// mas o ensaio usa somente tempo, PWM e leituras Hall.
// ============================================================

enum TipoSinalIdentificacao : uint8_t {
  TIPO_SINAL_PULSO = 1
};

enum EstadoIdentificacao : uint8_t {
  IDENT_PARADO = 0,
  IDENT_AGUARDANDO_INICIO = 1,
  IDENT_EXECUTANDO_PULSO = 4,
  IDENT_RECUPERANDO = 5,
  IDENT_FINALIZADO = 6,
  IDENT_ABORTADO = 7,

  // Aliases mantidos para compatibilidade com logs/scripts antigos.
  IDENT_CAPTURANDO_BASELINE = IDENT_AGUARDANDO_INICIO,
  IDENT_BUSCANDO_LIMIAR = 2,
  IDENT_DESGRUDANDO = 3
};

struct EntradasIdentificacao {
  unsigned long now_us;
  int pwm_atual;

  uint32_t hall_raw_mv;
  uint32_t hall_filt_mv;
  uint32_t hall_cima_raw_mv;
  uint32_t hall_cima_filt_mv;

  // Mantidos para compatibilidade de telemetria, mas nao usados
  // pela logica do ensaio de pulso fixo.
  float tof_raw_mm;
  float tof_actual_mm;
  bool tof_valido;
  bool novo_tof;
  int range_status;
};

struct SaidasIdentificacao {
  bool ativo;
  bool finalizado;
  bool abortado;
  int pwm_cmd;
  EstadoIdentificacao estado;
};

struct PerfilIdentificacao {
  char nome[24];
  char test_id[32];

  uint8_t repeticoes;              // numero de pulsos
  uint8_t pwm_pulso;               // tamanho/amplitude do pulso em PWM 0..255
  uint16_t tempo_pulso_ms;         // tempo com PWM=pwm_pulso
  uint16_t tempo_entre_pulsos_ms;  // tempo em PWM=0 apos cada pulso
  uint16_t tempo_pre_inicio_ms;    // tempo em PWM=0 antes do primeiro pulso
};

struct DadosIdentificacao {
  bool ativo;
  bool finalizado;
  bool abortado;
  EstadoIdentificacao estado;
  TipoSinalIdentificacao tipo_sinal;

  uint8_t repeticao_atual;
  uint8_t repeticoes_totais;
  uint8_t fase_execucao;

  int pwm_cmd;
  int pwm_limiar;   // compatibilidade: permanece -1 nesta versao
  int pwm_teste;    // compatibilidade: igual ao pwm_pulso
  int pwm_pulso;

  uint32_t hall_raw_mv;
  uint32_t hall_filt_mv;
  uint32_t hall_cima_raw_mv;
  uint32_t hall_cima_filt_mv;

  uint32_t hall_base_mv;
  uint32_t hall_cima_base_mv;
  uint32_t hall_limiar_mv;              // compatibilidade: igual ao hall_base_mv
  uint32_t hall_cima_limiar_mv;         // compatibilidade: igual ao hall_cima_base_mv
  uint32_t hall_pulso_pico_mv;
  uint32_t hall_cima_pulso_pico_mv;
  int32_t hall_delta_pico_mv;
  int32_t hall_cima_delta_pico_mv;

  // Mantidos para compatibilidade com o log antigo. Nesta versao
  // ficam NAN, pois o ToF nao participa da identificacao.
  float tof_base_mm;
  float lim_actual_mm;
  float lim_raw_mm;
  float delta_lim_mm;
  float pulso_pico_actual_mm;
  float delta_pulso_mm;

  uint16_t tempo_pulso_ms;
  uint16_t tempo_entre_pulsos_ms;
  uint16_t tempo_pre_inicio_ms;

  unsigned long t_estado_us;
  unsigned long t_fase_us;

  char test_id[32];
  char test_param[192];
  char motivo[64];
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
const char* ensaiosIdentificacaoNomeTipo(TipoSinalIdentificacao tipo);

#endif
