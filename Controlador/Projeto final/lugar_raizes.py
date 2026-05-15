import numpy as np
import control as ct
import matplotlib.pyplot as plt

# Função de transferência da planta
# Pelo que apareceu no MATLAB:
# G(s) = -44.21 / (s^3 + 32.74 s^2 - 1971 s - 6.423e4)

G = ct.tf([-44.21], [1, 32.74, -1971, -6.423e4])

# ============================================================
# 1) Lugar das raízes sem controlador
# ============================================================

plt.figure(figsize=(8, 6))
ct.root_locus(-G, grid=False, plot=True)
plt.title('Lugar das raízes sem controlador')
plt.xlabel('Eixo real (s⁻¹)')
plt.ylabel('Eixo imaginário (s⁻¹)')
plt.axvline(x=0, color='black', linewidth=1.2)
plt.grid(True, linewidth=0.4, alpha=0.5)

plt.show()



# ============================================================
# 2) Controlador
# ============================================================

C = ct.tf([-200, -2200, -2000], [1, 0])

# Malha aberta compensada
L = C * G

# Malha fechada com realimentação unitária
T = ct.feedback(L, 1)

# Polos de malha fechada
polos_mf = ct.poles(T)

print("Controlador C(s):")
print(C)

print("\nMalha aberta compensada C(s)G(s):")
print(L)

print("\nPolos de malha fechada com o controlador escolhido:")
for p in polos_mf:
    print(p)
    
# ============================================================
# 3) Lugar das raízes com controlador
# ============================================================

plt.figure(figsize=(10, 4))

ct.root_locus(L, grid=False, plot=True)

plt.title('Lugar das raízes com controlador')
plt.xlabel('Eixo real (s⁻¹)')
plt.ylabel('Eixo imaginário (s⁻¹)')

plt.xlim([-50, 50])
plt.ylim([-120, 120])

plt.axvline(x=0, color='black', linewidth=1.2)
plt.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')

plt.grid(True, linewidth=0.4, alpha=0.5)

plt.show()