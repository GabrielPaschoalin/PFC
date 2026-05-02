#include "ensaios_identificacao.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

// ============================================================
// PARAMETROS DEFAULT DO ENSAIO - V33
// ------------------------------------------------------------
// Comando serial equivalente:
//   PULSO 200 30 1000 10
// Onde:
//   200  = PWM do pulso
//   30   = tempo do pulso em ms
//   1000 = tempo em PWM=0 entre pulsos em ms
//   10   = numero de pulsos
// ============================================================
static const PerfilIdentificacao PERFIL_ATIVO = {
  "PULSO_FIXO",
  "IDENT_PULSO_FIXO",
  10,    // repeticoes
  200,   // pwm_pulso
  30,    // tempo_pulso_ms
  1000,  // tempo_entre_pulsos_ms
  0      // tempo_pre_inicio_ms
};

struct EstadoInterno {
  bool ativo = false;
  bool finalizado = false;
  bool abortado = false;

  EstadoIdentificacao estado = IDENT_PARADO;
  TipoSinalIdentificacao tipo_sinal = TIPO_SINAL_PULSO;
  PerfilIdentificacao perfil = PERFIL_ATIVO;

  int pwm_cmd = 0;
  int pwm_limiar = -1;
  int pwm_teste = PERFIL_ATIVO.pwm_pulso;

  uint32_t hall_raw_mv = 0;
  uint32_t hall_filt_mv = 0;
  uint32_t hall_cima_raw_mv = 0;
  uint32_t hall_cima_filt_mv = 0;

  uint32_t hall_base_mv = 0;
  uint32_t hall_cima_base_mv = 0;
  uint32_t hall_limiar_mv = 0;
  uint32_t hall_cima_limiar_mv = 0;
  uint32_t hall_pulso_pico_mv = 0;
  uint32_t hall_cima_pulso_pico_mv = 0;
  int32_t hall_delta_pico_mv = 0;
  int32_t hall_cima_delta_pico_mv = 0;

  float tof_base_mm = NAN;
  float lim_actual_mm = NAN;
  float lim_raw_mm = NAN;
  float delta_lim_mm = NAN;
  float pulso_pico_actual_mm = NAN;
  float delta_pulso_mm = NAN;

  uint8_t repeticao_atual = 0;
  uint8_t fase_execucao = 0;

  unsigned long inicio_estado_us = 0;
  unsigned long inicio_fase_us = 0;

  char test_id[32] = "";
  char test_param[192] = "";
  char motivo[64] = "NULL";
};

EstadoInterno g;

long clampLong(long x, long lo, long hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

unsigned long elapsedMs(unsigned long nowUs, unsigned long startUs) {
  if (startUs == 0UL) return 0UL;
  return (unsigned long)((nowUs - startUs) / 1000UL);
}

void copiarTexto(char* dst, size_t dstSize, const char* src) {
  if (!dst || dstSize == 0) return;
  if (!src || src[0] == '\0') src = "NULL";
  strncpy(dst, src, dstSize - 1);
  dst[dstSize - 1] = '\0';
}

void sanitizarPerfil(PerfilIdentificacao& p) {
  if (p.nome[0] == '\0') copiarTexto(p.nome, sizeof(p.nome), PERFIL_ATIVO.nome);
  if (p.test_id[0] == '\0') copiarTexto(p.test_id, sizeof(p.test_id), PERFIL_ATIVO.test_id);

  if (p.repeticoes == 0) p.repeticoes = 1;
  p.pwm_pulso = (uint8_t)clampLong((long)p.pwm_pulso, 0L, 255L);

  if (p.tempo_pulso_ms == 0) p.tempo_pulso_ms = 1;
  // tempo_entre_pulsos_ms pode ser 0 se quisermos pulsos colados.
  // tempo_pre_inicio_ms tambem pode ser 0.
}

void preencherTestParam() {
  snprintf(g.test_param, sizeof(g.test_param),
           "SIG=PUL_FIXO;PWM=%u;TP=%u;TE=%u;REP=%u;PRE=%u;TOF=IGNORADO",
           (unsigned)g.perfil.pwm_pulso,
           (unsigned)g.perfil.tempo_pulso_ms,
           (unsigned)g.perfil.tempo_entre_pulsos_ms,
           (unsigned)g.perfil.repeticoes,
           (unsigned)g.perfil.tempo_pre_inicio_ms);
}

void setMotivo(const char* motivo) {
  copiarTexto(g.motivo, sizeof(g.motivo), motivo);
}

void definirEstado(EstadoIdentificacao novoEstado, unsigned long nowUs, uint8_t novaFase = 0) {
  g.estado = novoEstado;
  g.inicio_estado_us = nowUs;
  g.inicio_fase_us = nowUs;
  g.fase_execucao = novaFase;
}

void resetValoresDinamicos() {
  g.pwm_cmd = 0;
  g.pwm_limiar = -1;
  g.pwm_teste = g.perfil.pwm_pulso;

  g.hall_raw_mv = 0;
  g.hall_filt_mv = 0;
  g.hall_cima_raw_mv = 0;
  g.hall_cima_filt_mv = 0;

  g.hall_base_mv = 0;
  g.hall_cima_base_mv = 0;
  g.hall_limiar_mv = 0;
  g.hall_cima_limiar_mv = 0;
  g.hall_pulso_pico_mv = 0;
  g.hall_cima_pulso_pico_mv = 0;
  g.hall_delta_pico_mv = 0;
  g.hall_cima_delta_pico_mv = 0;

  g.tof_base_mm = NAN;
  g.lim_actual_mm = NAN;
  g.lim_raw_mm = NAN;
  g.delta_lim_mm = NAN;
  g.pulso_pico_actual_mm = NAN;
  g.delta_pulso_mm = NAN;

  g.repeticao_atual = 0;
  g.fase_execucao = 0;
  g.inicio_estado_us = 0;
  g.inicio_fase_us = 0;
}

void resetInterno() {
  memset(&g, 0, sizeof(g));
  g.estado = IDENT_PARADO;
  g.tipo_sinal = TIPO_SINAL_PULSO;
  g.perfil = PERFIL_ATIVO;
  sanitizarPerfil(g.perfil);
  resetValoresDinamicos();
  copiarTexto(g.test_id, sizeof(g.test_id), g.perfil.test_id);
  preencherTestParam();
  setMotivo("NULL");
}

void carregarEntradas(const EntradasIdentificacao& in) {
  g.hall_raw_mv = in.hall_raw_mv;
  g.hall_filt_mv = in.hall_filt_mv;
  g.hall_cima_raw_mv = in.hall_cima_raw_mv;
  g.hall_cima_filt_mv = in.hall_cima_filt_mv;
}

void iniciarNovoPulso(unsigned long nowUs) {
  if (g.repeticao_atual < g.perfil.repeticoes) {
    g.repeticao_atual++;
  }

  // Reinicia o pico desta repeticao. O pico e definido como o ponto
  // de maior delta absoluto em relacao ao Hall no instante de inicio do teste.
  g.hall_pulso_pico_mv = g.hall_filt_mv;
  g.hall_cima_pulso_pico_mv = g.hall_cima_filt_mv;
  g.hall_delta_pico_mv = (int32_t)g.hall_filt_mv - (int32_t)g.hall_base_mv;
  g.hall_cima_delta_pico_mv = (int32_t)g.hall_cima_filt_mv - (int32_t)g.hall_cima_base_mv;

  g.pwm_cmd = g.perfil.pwm_pulso;
  definirEstado(IDENT_EXECUTANDO_PULSO, nowUs, 2);
}

void atualizarPicoHall(const EntradasIdentificacao& in) {
  int32_t deltaBaixo = (int32_t)in.hall_filt_mv - (int32_t)g.hall_base_mv;
  if (labs(deltaBaixo) >= labs(g.hall_delta_pico_mv)) {
    g.hall_delta_pico_mv = deltaBaixo;
    g.hall_pulso_pico_mv = in.hall_filt_mv;
  }

  int32_t deltaCima = (int32_t)in.hall_cima_filt_mv - (int32_t)g.hall_cima_base_mv;
  if (labs(deltaCima) >= labs(g.hall_cima_delta_pico_mv)) {
    g.hall_cima_delta_pico_mv = deltaCima;
    g.hall_cima_pulso_pico_mv = in.hall_cima_filt_mv;
  }
}

void finalizar(unsigned long nowUs) {
  g.ativo = false;
  g.finalizado = true;
  g.abortado = false;
  g.pwm_cmd = 0;
  definirEstado(IDENT_FINALIZADO, nowUs, 0);
  setMotivo("OK");
}

}  // namespace

void ensaiosIdentificacaoBegin() {
  resetInterno();
}

void ensaiosIdentificacaoReset() {
  resetInterno();
}

void ensaiosIdentificacaoStop() {
  resetInterno();
}

bool ensaiosIdentificacaoStart(const EntradasIdentificacao& in) {
  return ensaiosIdentificacaoStartComPerfil(in, PERFIL_ATIVO);
}

bool ensaiosIdentificacaoStartComPerfil(const EntradasIdentificacao& in, const PerfilIdentificacao& perfil) {
  resetInterno();

  g.perfil = perfil;
  sanitizarPerfil(g.perfil);
  copiarTexto(g.test_id, sizeof(g.test_id), g.perfil.test_id);
  preencherTestParam();

  g.ativo = true;
  g.finalizado = false;
  g.abortado = false;
  g.tipo_sinal = TIPO_SINAL_PULSO;

  carregarEntradas(in);

  g.hall_base_mv = in.hall_filt_mv;
  g.hall_cima_base_mv = in.hall_cima_filt_mv;
  g.hall_limiar_mv = g.hall_base_mv;  // compatibilidade com coluna antiga
  g.hall_cima_limiar_mv = g.hall_cima_base_mv;
  g.pwm_limiar = -1;
  g.pwm_teste = g.perfil.pwm_pulso;
  g.pwm_cmd = 0;

  setMotivo("EM_ANDAMENTO");

  if (g.perfil.tempo_pre_inicio_ms > 0) {
    definirEstado(IDENT_AGUARDANDO_INICIO, in.now_us, 1);
  } else {
    iniciarNovoPulso(in.now_us);
  }

  return true;
}

void ensaiosIdentificacaoUpdate(const EntradasIdentificacao& in, SaidasIdentificacao& out) {
  carregarEntradas(in);

  if (g.ativo) {
    switch (g.estado) {
      case IDENT_AGUARDANDO_INICIO: {
        g.pwm_cmd = 0;
        if (elapsedMs(in.now_us, g.inicio_estado_us) >= g.perfil.tempo_pre_inicio_ms) {
          iniciarNovoPulso(in.now_us);
        }
      } break;

      case IDENT_EXECUTANDO_PULSO: {
        g.pwm_cmd = g.perfil.pwm_pulso;
        atualizarPicoHall(in);

        if (elapsedMs(in.now_us, g.inicio_estado_us) >= g.perfil.tempo_pulso_ms) {
          g.pwm_cmd = 0;
          definirEstado(IDENT_RECUPERANDO, in.now_us, 3);
        }
      } break;

      case IDENT_RECUPERANDO: {
        g.pwm_cmd = 0;
        atualizarPicoHall(in);

        if (elapsedMs(in.now_us, g.inicio_estado_us) >= g.perfil.tempo_entre_pulsos_ms) {
          if (g.repeticao_atual >= g.perfil.repeticoes) {
            finalizar(in.now_us);
          } else {
            iniciarNovoPulso(in.now_us);
          }
        }
      } break;

      case IDENT_FINALIZADO:
      case IDENT_ABORTADO:
      case IDENT_PARADO:
      default:
        break;
    }
  }

  out.ativo = g.ativo;
  out.finalizado = g.finalizado;
  out.abortado = g.abortado;
  out.pwm_cmd = g.pwm_cmd;
  out.estado = g.estado;
}

bool ensaiosIdentificacaoEstaAtivo() {
  return g.ativo;
}

DadosIdentificacao ensaiosIdentificacaoGetDados() {
  DadosIdentificacao d;
  memset(&d, 0, sizeof(d));

  d.ativo = g.ativo;
  d.finalizado = g.finalizado;
  d.abortado = g.abortado;
  d.estado = g.estado;
  d.tipo_sinal = g.tipo_sinal;
  d.repeticao_atual = g.repeticao_atual;
  d.repeticoes_totais = g.perfil.repeticoes;
  d.fase_execucao = g.fase_execucao;

  d.pwm_cmd = g.pwm_cmd;
  d.pwm_limiar = g.pwm_limiar;
  d.pwm_teste = g.pwm_teste;
  d.pwm_pulso = g.perfil.pwm_pulso;

  d.hall_raw_mv = g.hall_raw_mv;
  d.hall_filt_mv = g.hall_filt_mv;
  d.hall_cima_raw_mv = g.hall_cima_raw_mv;
  d.hall_cima_filt_mv = g.hall_cima_filt_mv;
  d.hall_base_mv = g.hall_base_mv;
  d.hall_cima_base_mv = g.hall_cima_base_mv;
  d.hall_limiar_mv = g.hall_limiar_mv;
  d.hall_cima_limiar_mv = g.hall_cima_limiar_mv;
  d.hall_pulso_pico_mv = g.hall_pulso_pico_mv;
  d.hall_cima_pulso_pico_mv = g.hall_cima_pulso_pico_mv;
  d.hall_delta_pico_mv = g.hall_delta_pico_mv;
  d.hall_cima_delta_pico_mv = g.hall_cima_delta_pico_mv;

  d.tof_base_mm = g.tof_base_mm;
  d.lim_actual_mm = g.lim_actual_mm;
  d.lim_raw_mm = g.lim_raw_mm;
  d.delta_lim_mm = g.delta_lim_mm;
  d.pulso_pico_actual_mm = g.pulso_pico_actual_mm;
  d.delta_pulso_mm = g.delta_pulso_mm;

  d.tempo_pulso_ms = g.perfil.tempo_pulso_ms;
  d.tempo_entre_pulsos_ms = g.perfil.tempo_entre_pulsos_ms;
  d.tempo_pre_inicio_ms = g.perfil.tempo_pre_inicio_ms;

  d.t_estado_us = g.inicio_estado_us;
  d.t_fase_us = g.inicio_fase_us;

  copiarTexto(d.test_id, sizeof(d.test_id), g.test_id);
  copiarTexto(d.test_param, sizeof(d.test_param), g.test_param);
  copiarTexto(d.motivo, sizeof(d.motivo), g.motivo);

  return d;
}

PerfilIdentificacao ensaiosIdentificacaoGetPerfilAtivo() {
  return g.perfil;
}

const char* ensaiosIdentificacaoNomeEstado(EstadoIdentificacao estado) {
  switch (estado) {
    case IDENT_PARADO: return "PARADO";
    case IDENT_AGUARDANDO_INICIO: return "AGUARDANDO_INICIO";
    case IDENT_BUSCANDO_LIMIAR: return "BUSCANDO_LIMIAR_UNUSED";
    case IDENT_DESGRUDANDO: return "DESGRUDANDO_UNUSED";
    case IDENT_EXECUTANDO_PULSO: return "EXECUTANDO_PULSO";
    case IDENT_RECUPERANDO: return "RECUPERANDO";
    case IDENT_FINALIZADO: return "FINALIZADO";
    case IDENT_ABORTADO: return "ABORTADO";
    default: return "NULL";
  }
}

const char* ensaiosIdentificacaoNomeTipo(TipoSinalIdentificacao tipo) {
  switch (tipo) {
    case TIPO_SINAL_PULSO: return "PULSO_FIXO";
    default: return "NULL";
  }
}
