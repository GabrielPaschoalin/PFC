import sympy as sp
import numpy as np

# ============================================================
# 1) Definição simbólica
# ============================================================

# Estados e entrada
i, x, v, u = sp.symbols("i x v u", real=True)

# Parâmetros físicos
R, m, g = sp.symbols("R m g", positive=True, real=True)

# Parâmetros eletromagnéticos identificados
Kfm, kL, Lb = sp.symbols("Kfm kL Lb", positive=True, real=True)

# Vetor de estados e entrada
X = sp.Matrix([i, x, v])
U = sp.Matrix([u])

# ============================================================
# 2) Funções do modelo não linear
# ============================================================

# Indutância
L = kL/x + Lb

# Força magnética
Fm = Kfm * i**2 / x**2

# Fator de força
Bl = Kfm * i / x**2

# Modelo não linear:
# i_dot = u/L(x) - R*i/L(x) - Bl(i,x)*v/L(x)
# x_dot = v
# v_dot = g - Fm(i,x)/m

f = sp.Matrix([
    (u - R*i - Bl*v)/L,
    v,
    g - Fm/m
])

# Saída escolhida: y = x
h = sp.Matrix([x])

# ============================================================
# 3) Linearização simbólica por jacobianas
# ============================================================

A_sym = sp.simplify(f.jacobian(X))
B_sym = sp.simplify(f.jacobian(U))
C_sym = sp.simplify(h.jacobian(X))
D_sym = sp.simplify(h.jacobian(U))

print("\nA simbólica geral:")
sp.pprint(A_sym)

print("\nB simbólica geral:")
sp.pprint(B_sym)

print("\nC simbólica:")
sp.pprint(C_sym)

print("\nD simbólica:")
sp.pprint(D_sym)

# ============================================================
# 4) Ponto de operação
# ============================================================

# Parâmetros numéricos
params = {
    R: 13.5,
    m: 0.022,
    g: 9.81,
    Kfm: 1.8632e-5,
    kL: 0.0000415,
    Lb: 0.4082238
}

# Posição de operação
x0 = 0.01       # 1 cm = 0.01 m
v0 = 0.0

# Corrente de equilíbrio: mg = Kfm*i0^2/x0^2
i0 = x0 * np.sqrt(params[m] * params[g] / params[Kfm])

# Tensão de equilíbrio: u0 = R*i0
u0 = params[R] * i0

operating_point = {
    i: i0,
    x: x0,
    v: v0,
    u: u0
}

# ============================================================
# 5) Avaliação numérica das matrizes no ponto de operação
# ============================================================

A_num = sp.N(A_sym.subs(params).subs(operating_point), 8)
B_num = sp.N(B_sym.subs(params).subs(operating_point), 8)
C_num = sp.N(C_sym.subs(params).subs(operating_point), 8)
D_num = sp.N(D_sym.subs(params).subs(operating_point), 8)

# Converte para numpy, caso queira usar em scipy/control/matlab etc.
A = np.array(A_num.tolist(), dtype=float)
B = np.array(B_num.tolist(), dtype=float)
C = np.array(C_num.tolist(), dtype=float)
D = np.array(D_num.tolist(), dtype=float)

np.set_printoptions(precision=6, suppress=True)

print("\n================ PONTO DE OPERAÇÃO ================")
print(f"x0 = {x0:.6f} m")
print(f"v0 = {v0:.6f} m/s")
print(f"i0 = {i0:.6f} A")
print(f"u0 = {u0:.6f} V")

print("\n================ MATRIZES LINEARIZADAS ================")
print("\nA =")
print(A)

print("\nB =")
print(B)

print("\nC =")
print(C)

print("\nD =")
print(D)

# ============================================================
# 6) Checagem: f(X0,u0) deve ser aproximadamente zero
# ============================================================

f_eq = sp.N(f.subs(params).subs(operating_point), 10)

print("\n================ CHECAGEM DO EQUILÍBRIO ================")
print("f(X0, u0) =")
sp.pprint(f_eq)