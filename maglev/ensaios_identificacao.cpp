#include "ensaios_identificacao.h"

#include <string.h>

namespace {

// ============================================================
// DEFAULT DO ENSAIO - V34
// ------------------------------------------------------------
// Comando serial equivalente:
//   PULSO 200 30 1000 10
// Onde:
//   200  = PWM do pulso
//   30   = tempo do pulso em ms
//   1000 = tempo em PWM=0 entre pulsos
//   10   = numero de pulsos
// ============================================================
static const PerfilIdentificacao PERFIL_ATIVO = {
  10,    // repeticoes
  200,   // pwm_pulso
  30,    // tempo_pulso_ms
  1000,  // tempo_entre_pulsos_ms
  0      // tempo_pre_inicio_ms
};

struct EstadoInterno {
  bool ativo;
  bool finalizado;
  bool abortado;

  EstadoIdentificacao estado;
  PerfilIdentificacao perfil;

  uint8_t repeticao_atual;
  int pwm_cmd;

  unsigned long inicio_estado_us;
  unsigned long inicio_fase_us;
};

EstadoInterno g;

unsigned long msToUs(uint16_t ms) {
  return (unsigned long)ms * 1000UL;
}

bool tempoDecorrido(unsigned long nowUs, unsigned long startUs, uint16_t duracaoMs) {
  return (unsigned long)(nowUs - startUs) >= msToUs(duracaoMs);
}

void sanitizarPerfil(PerfilIdentificacao& p) {
  if (p.repeticoes == 0) p.repeticoes = 1;
  if (p.tempo_pulso_ms == 0) p.tempo_pulso_ms = 1;
  // tempo_entre_pulsos_ms e tempo_pre_inicio_ms podem ser 0.
}

void preencherSaida(SaidasIdentificacao& out) {
  out.ativo = g.ativo;
  out.finalizado = g.finalizado;
  out.abortado = g.abortado;
  out.pwm_cmd = g.pwm_cmd;
  out.estado = g.estado;
}

void definirEstado(EstadoIdentificacao novoEstado, unsigned long nowUs) {
  g.estado = novoEstado;
  g.inicio_estado_us = nowUs;
  g.inicio_fase_us = nowUs;
}

void resetInterno() {
  memset(&g, 0, sizeof(g));
  g.ativo = false;
  g.finalizado = false;
  g.abortado = false;
  g.estado = IDENT_PARADO;
  g.perfil = PERFIL_ATIVO;
  sanitizarPerfil(g.perfil);
  g.repeticao_atual = 0;
  g.pwm_cmd = 0;
  g.inicio_estado_us = 0UL;
  g.inicio_fase_us = 0UL;
}

void iniciarNovoPulso(unsigned long nowUs) {
  if (g.repeticao_atual < g.perfil.repeticoes) {
    g.repeticao_atual++;
  }

  g.pwm_cmd = g.perfil.pwm_pulso;
  definirEstado(IDENT_EXECUTANDO_PULSO, nowUs);
}

void finalizar(unsigned long nowUs) {
  g.ativo = false;
  g.finalizado = true;
  g.abortado = false;
  g.pwm_cmd = 0;
  definirEstado(IDENT_FINALIZADO, nowUs);
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

  g.ativo = true;
  g.finalizado = false;
  g.abortado = false;
  g.repeticao_atual = 0;
  g.pwm_cmd = 0;

  if (g.perfil.tempo_pre_inicio_ms > 0) {
    definirEstado(IDENT_AGUARDANDO_INICIO, in.now_us);
  } else {
    iniciarNovoPulso(in.now_us);
  }

  return true;
}

void ensaiosIdentificacaoUpdate(const EntradasIdentificacao& in, SaidasIdentificacao& out) {
  // Os Halls ficam na entrada para manter a interface pronta para a proxima etapa
  // de estimacao por Hall, mas a V34 nao usa Hall para decidir o ensaio.
  (void)in.pwm_atual;
  (void)in.hall_inf_mv;
  (void)in.hall_sup_mv;

  if (g.ativo) {
    switch (g.estado) {
      case IDENT_AGUARDANDO_INICIO:
        g.pwm_cmd = 0;
        if (tempoDecorrido(in.now_us, g.inicio_estado_us, g.perfil.tempo_pre_inicio_ms)) {
          iniciarNovoPulso(in.now_us);
        }
        break;

      case IDENT_EXECUTANDO_PULSO:
        g.pwm_cmd = g.perfil.pwm_pulso;
        if (tempoDecorrido(in.now_us, g.inicio_estado_us, g.perfil.tempo_pulso_ms)) {
          g.pwm_cmd = 0;
          definirEstado(IDENT_RECUPERANDO, in.now_us);
        }
        break;

      case IDENT_RECUPERANDO:
        g.pwm_cmd = 0;
        if (tempoDecorrido(in.now_us, g.inicio_estado_us, g.perfil.tempo_entre_pulsos_ms)) {
          if (g.repeticao_atual >= g.perfil.repeticoes) {
            finalizar(in.now_us);
          } else {
            iniciarNovoPulso(in.now_us);
          }
        }
        break;

      case IDENT_FINALIZADO:
      case IDENT_ABORTADO:
      case IDENT_PARADO:
      default:
        break;
    }
  }

  preencherSaida(out);
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

  d.repeticao_atual = g.repeticao_atual;
  d.repeticoes_totais = g.perfil.repeticoes;

  d.pwm_cmd = g.pwm_cmd;
  d.pwm_pulso = g.perfil.pwm_pulso;

  d.tempo_pulso_ms = g.perfil.tempo_pulso_ms;
  d.tempo_entre_pulsos_ms = g.perfil.tempo_entre_pulsos_ms;
  d.tempo_pre_inicio_ms = g.perfil.tempo_pre_inicio_ms;

  d.t_estado_us = g.inicio_estado_us;
  d.t_fase_us = g.inicio_fase_us;

  return d;
}

PerfilIdentificacao ensaiosIdentificacaoGetPerfilAtivo() {
  return g.perfil;
}

const char* ensaiosIdentificacaoNomeEstado(EstadoIdentificacao estado) {
  switch (estado) {
    case IDENT_PARADO: return "PARADO";
    case IDENT_AGUARDANDO_INICIO: return "AGUARDANDO_INICIO";
    case IDENT_EXECUTANDO_PULSO: return "EXECUTANDO_PULSO";
    case IDENT_RECUPERANDO: return "RECUPERANDO";
    case IDENT_FINALIZADO: return "FINALIZADO";
    case IDENT_ABORTADO: return "ABORTADO";
    default: return "NULL";
  }
}
