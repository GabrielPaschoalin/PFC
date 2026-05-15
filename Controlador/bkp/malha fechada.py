import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================================
# 1) PLANTA ORIGINAL COM GANHO NEGATIVO
# ============================================================

# G(s) = -44.2989 / (s^3 + 32.74 s^2 - 1970.9327 s - 64235.88)

b = 44.2989

den_G = np.array([1.0, 32.74, -1970.9327, -64235.88])
num_G = np.array([-b])

print("\n==================== 1) PLANTA ORIGINAL ====================")
print("G(s) = -44.2989 / (s^3 + 32.74 s^2 - 1970.9327 s - 64235.88)")
print("\nNumerador de G(s):")
print(num_G)
print("\nDenominador de G(s):")
print(den_G)

polos_G = np.roots(den_G)
zeros_G = np.roots(num_G)

print("\nPolos da planta G(s):")
for p in polos_G:
    print(f"{p.real:.6f} {p.imag:+.6f}j")

if len(zeros_G) == 0:
    print("\nZeros da planta G(s): nenhum zero finito")
else:
    print("\nZeros da planta G(s):")
    for z in zeros_G:
        print(f"{z.real:.6f} {z.imag:+.6f}j")


# ============================================================
# 2) CONTROLADOR PID
# ============================================================

# C(s) = Kp + Ki/s + Kd*s
# C(s) = (Kd*s^2 + Kp*s + Ki) / s

print("\n==================== 2) CONTROLADOR PID ====================")
print("C(s) = Kp + Ki/s + Kd*s")
print("C(s) = (Kd*s^2 + Kp*s + Ki) / s")


# ============================================================
# 3) EQUAÇÃO CARACTERÍSTICA DA MALHA FECHADA
# ============================================================

print("\n==================== 3) EQUAÇÃO CARACTERÍSTICA ====================")
print("T(s) = G(s)C(s) / (1 + G(s)C(s))")
print("Equação característica:")
print("s*D(s) - b*(Kd*s^2 + Kp*s + Ki) = 0")
print()
print("Expandindo:")
print("s^4 + 32.74*s^3")
print("- (1970.9327 + b*Kd)*s^2")
print("- (64235.88 + b*Kp)*s")
print("- b*Ki = 0")


# ============================================================
# 4) ESPECIFICAÇÕES DO PAR DOMINANTE
# ============================================================

zeta = 0.7
wn = 1.2

pa = (32.74 - 2*zeta*wn) / 2

print("\n==================== 4) POLINÔMIO DESEJADO ====================")
print(f"zeta = {zeta}")
print(f"wn   = {wn}")
print()
print("Par dominante:")
print("s^2 + 2*zeta*wn*s + wn^2")
print(f"s^2 + {2*zeta*wn:.6f}*s + {wn**2:.6f}")
print()
print("Restrição do coeficiente de s^3:")
print("2*zeta*wn + 2*pa = 32.74")
print(f"{2*zeta*wn:.6f} + 2*pa = 32.74")
print(f"pa = {pa:.6f}")
print()
print("Polinômio desejado:")
print("Pd(s) = (s^2 + 2*zeta*wn*s + wn^2)*(s + pa)^2")


# ============================================================
# 5) EXPANSÃO DO POLINÔMIO DESEJADO
# ============================================================

poly_dominante = np.array([1.0, 2*zeta*wn, wn**2])
poly_auxiliar = np.array([1.0, 2*pa, pa**2])

poly_desejado = np.convolve(poly_dominante, poly_auxiliar)

alpha4, alpha3, alpha2, alpha1, alpha0 = poly_desejado

print("\n==================== 5) EXPANSÃO DE Pd(s) ====================")
print("Pd(s) = s^4 + alpha3*s^3 + alpha2*s^2 + alpha1*s + alpha0")
print()
print(f"alpha3 = {alpha3:.6f}")
print(f"alpha2 = {alpha2:.6f}")
print(f"alpha1 = {alpha1:.6f}")
print(f"alpha0 = {alpha0:.6f}")

print("\nPolinômio desejado expandido:")
print(poly_desejado)

polos_desejados = np.roots(poly_desejado)

print("\nPolos desejados:")
for p in polos_desejados:
    print(f"{p.real:.6f} {p.imag:+.6f}j")


# ============================================================
# 6) CÁLCULO DOS GANHOS PID
# ============================================================

Kd = -(alpha2 + 1970.9327) / b
Kp = -(alpha1 + 64235.88) / b
Ki = -alpha0 / b

print("\n==================== 6) GANHOS PID ====================")
print("Como a planta original tem ganho negativo, os ganhos saem negativos nesta convenção.")
print()
print("Kd = -(alpha2 + 1970.9327)/b")
print("Kp = -(alpha1 + 64235.88)/b")
print("Ki = -alpha0/b")
print()
print(f"Kp = {Kp:.6f}")
print(f"Ki = {Ki:.6f}")
print(f"Kd = {Kd:.6f}")


# ============================================================
# 7) POLOS E ZEROS DO PID
# ============================================================

num_C = np.array([Kd, Kp, Ki])
den_C = np.array([1.0, 0.0])

zeros_C = np.roots(num_C)
polos_C = np.roots(den_C)

print("\n==================== 7) POLOS E ZEROS DO PID ====================")
print("C(s) = (Kd*s^2 + Kp*s + Ki)/s")

print("\nPolo do PID:")
for p in polos_C:
    print(f"{p.real:.6f} {p.imag:+.6f}j")

print("\nZeros do PID:")
for z in zeros_C:
    print(f"{z.real:.6f} {z.imag:+.6f}j")


# ============================================================
# 8) MALHA ABERTA COMPENSADA G(s)C(s)
# ============================================================

num_GC = np.polymul(num_G, num_C)
den_GC = np.polymul(den_G, den_C)

print("\n==================== 8) MALHA ABERTA COMPENSADA ====================")
print("G(s)C(s) = num_GC(s) / den_GC(s)")
print()
print("Numerador de G(s)C(s):")
print(num_GC)
print()
print("Denominador de G(s)C(s):")
print(den_GC)

polos_GC = np.roots(den_GC)
zeros_GC = np.roots(num_GC)

print("\nPolos de G(s)C(s):")
for p in polos_GC:
    print(f"{p.real:.6f} {p.imag:+.6f}j")

print("\nZeros de G(s)C(s):")
for z in zeros_GC:
    print(f"{z.real:.6f} {z.imag:+.6f}j")


# ============================================================
# 9) MALHA FECHADA T(s) = GC/(1 + GC)
# ============================================================

num_GC_padded = np.pad(num_GC, (len(den_GC) - len(num_GC), 0), mode="constant")

num_T = num_GC
den_T = den_GC + num_GC_padded

print("\n==================== 9) MALHA FECHADA ====================")
print("T(s) = G(s)C(s) / (1 + G(s)C(s))")
print()
print("Numerador de T(s):")
print(num_T)
print()
print("Denominador de T(s):")
print(den_T)

polos_T = np.roots(den_T)
zeros_T = np.roots(num_T)

print("\nPolos da malha fechada:")
for p in polos_T:
    print(f"{p.real:.6f} {p.imag:+.6f}j")

print("\nZeros da malha fechada:")
for z in zeros_T:
    print(f"{z.real:.6f} {z.imag:+.6f}j")


# ============================================================
# 10) RESPOSTA AO DEGRAU DE 1 mm
# ============================================================

degrau_m = 0.001
degrau_mm = degrau_m * 1000

sys_T = signal.TransferFunction(num_T, den_T)

t_final = 10.0
t = np.linspace(0, t_final, 3000)

t_out, y_unit = signal.step(sys_T, T=t)

y_m = degrau_m * y_unit
y_mm = y_m * 1000

ref_mm = degrau_mm * np.ones_like(t_out)


# ============================================================
# 11) MÉTRICAS SIMPLES DA RESPOSTA
# ============================================================

valor_final_mm = y_mm[-1]
erro_estacionario_mm = degrau_mm - valor_final_mm

maximo_mm = np.max(y_mm)
sobressinal_percent = ((maximo_mm - degrau_mm) / degrau_mm) * 100

banda_2_percent_mm = 0.02 * abs(degrau_mm)

erro_abs = np.abs(y_mm - degrau_mm)

tempo_acomodacao = None
for idx in range(len(t_out)):
    if np.all(erro_abs[idx:] <= banda_2_percent_mm):
        tempo_acomodacao = t_out[idx]
        break

print("\n==================== 10) RESPOSTA AO DEGRAU ====================")
print(f"Degrau aplicado: {degrau_mm:.6f} mm")
print(f"Valor final aproximado: {valor_final_mm:.6f} mm")
print(f"Erro estacionário aproximado: {erro_estacionario_mm:.6f} mm")
print(f"Máximo valor da resposta: {maximo_mm:.6f} mm")
print(f"Sobressinal aproximado: {sobressinal_percent:.6f} %")

if tempo_acomodacao is not None:
    print(f"Tempo de acomodação aproximado, banda 2%: {tempo_acomodacao:.6f} s")
else:
    print("Tempo de acomodação: não entrou na banda de 2% no tempo simulado.")


# ============================================================
# 12) GRÁFICO DA RESPOSTA AO DEGRAU
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(t_out, y_mm, label=r"Resposta $\Delta x(t)$")
plt.plot(t_out, ref_mm, "--", label="Referência: 1 mm")

plt.axhline(degrau_mm + banda_2_percent_mm, linestyle=":", label="Banda 2%")
plt.axhline(degrau_mm - banda_2_percent_mm, linestyle=":")

plt.title("Resposta ao degrau de 1 mm - Malha fechada com PID")
plt.xlabel("Tempo (s)")
plt.ylabel(r"$\Delta x(t)$ [mm]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("resposta_degrau_1mm_pid.png", dpi=300)
plt.show()


# ============================================================
# 13) LUGAR DAS RAÍZES DO SISTEMA COMPENSADO
# ============================================================

# O lugar das raízes mostra os polos de malha fechada quando
# um ganho escalar K multiplica a malha aberta compensada:
#
# L_K(s) = K * G(s) * C(s)
#
# A equação característica fica:
#
# den_GC(s) + K*num_GC(s) = 0
#
# Como num_GC tem grau menor que den_GC, completamos com zeros
# à esquerda para somar os polinômios.

print("\n==================== 11) LUGAR DAS RAÍZES ====================")
print("L_K(s) = K * G(s) * C(s)")
print("Equação característica do lugar das raízes:")
print("den_GC(s) + K*num_GC(s) = 0")

K_values = np.concatenate(([0.0], np.logspace(-3, 4, 1200)))

roots_locus = []

for K in K_values:
    char_poly = den_GC + K * num_GC_padded
    roots = np.roots(char_poly)
    roots_locus.append(roots)

roots_locus = np.array(roots_locus)

# Gráfico do lugar das raízes
plt.figure(figsize=(10, 7))
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)

# Trajetórias dos ramos
for i in range(roots_locus.shape[1]):
    plt.plot(roots_locus[:, i].real, roots_locus[:, i].imag, linewidth=1.2)

# Polos de malha aberta compensada
plt.scatter(polos_GC.real, polos_GC.imag, marker="x", s=120, label="Polos de G(s)C(s)")

# Zeros de malha aberta compensada
plt.scatter(
    zeros_GC.real,
    zeros_GC.imag,
    marker="o",
    s=120,
    facecolors="none",
    label="Zeros de G(s)C(s)"
)

plt.title("Lugar das raízes do sistema compensado com PID")
plt.xlabel("Parte real")
plt.ylabel("Parte imaginária")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lugar_das_raizes_pid_compensado.png", dpi=300)
plt.show()


# ============================================================
# 14) POLOS DE MALHA FECHADA PARA ALGUNS VALORES DE K
# ============================================================

print("\nPolos de malha fechada para alguns valores de K no lugar das raízes:")
for K_test in [0, 0.1, 1, 10, 100, 1000]:
    char_poly = den_GC + K_test * num_GC_padded
    roots_test = np.roots(char_poly)

    print(f"\nK = {K_test}")
    for r in roots_test:
        print(f"{r.real:.6f} {r.imag:+.6f}j")