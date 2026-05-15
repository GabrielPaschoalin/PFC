"""
Análise do modelo linearizado do levitador eletromagnético.

Este script converte uma representação em espaço de estados para função de
transferência, calcula zeros, polos, ganho, autovalores e controlabilidade.

Dependências:
    pip install numpy scipy
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def limpar_coeficientes(coefs: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Remove coeficientes numericamente desprezíveis e zeros à esquerda."""
    coefs = np.asarray(coefs, dtype=float)
    coefs[np.abs(coefs) < tol] = 0.0

    indices_nao_nulos = np.where(np.abs(coefs) > tol)[0]

    if len(indices_nao_nulos) == 0:
        return np.array([0.0])

    primeiro = indices_nao_nulos[0]
    return coefs[primeiro:]


def formatar_numero(valor: complex, casas: int = 4) -> str:
    """Formata números reais e complexos para impressão."""
    valor = complex(valor)

    if abs(valor.imag) < 1e-9:
        return f"{valor.real:.{casas}f}"

    sinal = "+" if valor.imag >= 0 else "-"
    return f"{valor.real:.{casas}f} {sinal} {abs(valor.imag):.{casas}f}j"


def formatar_polinomio(coefs: np.ndarray, var: str = "s", casas: int = 4) -> str:
    """Formata um polinômio a partir de coeficientes em ordem decrescente."""
    coefs = limpar_coeficientes(coefs)
    grau = len(coefs) - 1
    termos: list[tuple[str, str]] = []

    for i, coef in enumerate(coefs):
        potencia = grau - i

        if abs(coef) < 1e-9:
            continue

        sinal = "-" if coef < 0 else "+"
        coef_abs = abs(coef)

        if potencia == 0:
            termo = f"{coef_abs:.{casas}f}"
        elif potencia == 1:
            termo = var if abs(coef_abs - 1.0) < 1e-9 else f"{coef_abs:.{casas}f}{var}"
        else:
            termo = f"{var}^{potencia}" if abs(coef_abs - 1.0) < 1e-9 else f"{coef_abs:.{casas}f}{var}^{potencia}"

        termos.append((sinal, termo))

    if not termos:
        return "0"

    primeiro_sinal, primeiro_termo = termos[0]
    texto = primeiro_termo if primeiro_sinal == "+" else f"-{primeiro_termo}"

    for sinal, termo in termos[1:]:
        texto += f" {sinal} {termo}"

    return texto


def matriz_controlabilidade(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calcula a matriz de controlabilidade: [B AB A^2B ... A^(n-1)B]."""
    n = A.shape[0]
    blocos = [B]

    for k in range(1, n):
        blocos.append(np.linalg.matrix_power(A, k) @ B)

    return np.hstack(blocos)


def analisar_modelo_linear(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray | float = 0.0,
    tol: float = 1e-9,
    casas: int = 4,
) -> dict:
    """
    Analisa o modelo linear em espaço de estados:

        x_dot = A x + B u
        y     = C x + D u
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    if np.isscalar(D):
        D = np.array([[D]], dtype=float)
    else:
        D = np.asarray(D, dtype=float)

    n = A.shape[0]

    num, den = signal.ss2tf(A, B, C, D)
    num = num[0]

    num_limpo = limpar_coeficientes(num, tol=tol)
    den_limpo = limpar_coeficientes(den, tol=tol)

    den_lider = den_limpo[0]
    num_limpo = num_limpo / den_lider
    den_limpo = den_limpo / den_lider

    zeros = np.roots(num_limpo) if len(num_limpo) > 1 else np.array([])
    polos = np.roots(den_limpo)
    ganho = num_limpo[0]

    autovalores = np.linalg.eigvals(A)

    Ctrb = matriz_controlabilidade(A, B)
    posto_ctrb = np.linalg.matrix_rank(Ctrb, tol=tol)

    sistema_controlavel = posto_ctrb == n
    sistema_estavel_malha_aberta = np.all(np.real(autovalores) < 0)

    numerador_txt = formatar_polinomio(num_limpo, casas=casas)
    denominador_txt = formatar_polinomio(den_limpo, casas=casas)

    ft_expandida = f"G(s) = ({numerador_txt}) / ({denominador_txt})"

    if len(zeros) == 0:
        num_zpk_txt = f"{ganho:.{casas}f}"
    else:
        fatores_zeros = "".join(
            [f"(s - ({formatar_numero(z, casas=casas)}))" for z in zeros]
        )
        num_zpk_txt = f"{ganho:.{casas}f}{fatores_zeros}"

    fatores_polos = "".join(
        [f"(s - ({formatar_numero(p, casas=casas)}))" for p in polos]
    )
    ft_zpk = f"G(s) = {num_zpk_txt} / {fatores_polos}"

    return {
        "numerador_coeficientes": num_limpo,
        "denominador_coeficientes": den_limpo,
        "funcao_transferencia_expandida": ft_expandida,
        "zeros": zeros,
        "polos": polos,
        "ganho": ganho,
        "funcao_transferencia_zpk": ft_zpk,
        "autovalores_A": autovalores,
        "matriz_controlabilidade": Ctrb,
        "posto_controlabilidade": posto_ctrb,
        "sistema_controlavel": sistema_controlavel,
        "sistema_estavel_malha_aberta": sistema_estavel_malha_aberta,
    }


def imprimir_resultados(resultados: dict, casas: int = 4) -> None:
    """Imprime os resultados de forma organizada."""
    print("\n=== Função de Transferência Expandida ===")
    print(resultados["funcao_transferencia_expandida"])

    print("\n=== Função de Transferência em Zeros, Polos e Ganho ===")
    print(resultados["funcao_transferencia_zpk"])

    print("\nZeros:")
    if len(resultados["zeros"]) == 0:
        print("Nenhum zero finito")
    else:
        for z in resultados["zeros"]:
            print(formatar_numero(z, casas=casas))

    print("\nPolos:")
    for p in resultados["polos"]:
        print(formatar_numero(p, casas=casas))

    print("\nGanho:")
    print(f"{resultados['ganho']:.{casas}f}")

    print("\n=== Autovalores da matriz A ===")
    for lamb in resultados["autovalores_A"]:
        print(formatar_numero(lamb, casas=casas))

    print("\n=== Matriz de Controlabilidade ===")
    print(resultados["matriz_controlabilidade"])

    print("\nPosto da matriz de controlabilidade:")
    print(resultados["posto_controlabilidade"])

    print("\nSistema controlável?")
    print("Sim" if resultados["sistema_controlavel"] else "Não")

    print("\nSistema estável em malha aberta?")
    print("Sim" if resultados["sistema_estavel_malha_aberta"] else "Não")


def main() -> None:
    """Exemplo de uso com as matrizes do modelo linearizado do Maglev."""
    A = np.array([
        [-32.74,     0.0,   -0.49],
        [  0.0,      0.0,    1.0 ],
        [-18.23,  1962.0,    0.0 ],
    ])

    B = np.array([
        [2.43],
        [0.0 ],
        [0.0 ],
    ])

    # Saída escolhida: posição da esfera, isto é, y = x.
    C = np.array([
        [0.0, 1.0, 0.0],
    ])

    D = np.array([
        [0.0],
    ])

    resultados = analisar_modelo_linear(A, B, C, D)
    imprimir_resultados(resultados)


if __name__ == "__main__":
    main()
