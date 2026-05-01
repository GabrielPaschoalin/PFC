import numpy as np
import matplotlib.pyplot as plt

# Correntes avaliadas
corrente = np.array([0.8, 1.0, 1.2, 1.5, 1.8])

# Dados para distância de 0,5 cm
medido_05 = np.array([7905, 9800, 11500, 14210, 16843])
mu100_05  = np.array([7936, 9920, 11904, 14880, 17857])
mu500_05  = np.array([9078, 11348, 13617, 17022, 20426])
mu900_05  = np.array([9246, 11557, 13869, 17336, 20803])
mu1300_05 = np.array([9313, 11642, 13970, 17463, 20955])

# Dados para distância de 1,0 cm
medido_10 = np.array([4632, 6366, 7348, 8295, 9850])
mu100_10  = np.array([4690, 5863, 7036, 8795, 10554])
mu500_10  = np.array([5395, 6744, 8093, 10116, 12139])
mu900_10  = np.array([5499, 6873, 8248, 10310, 12372])
mu1300_10 = np.array([5540, 6926, 8311, 10388, 12466])

def erro_medio_percentual(medido, simulado):
    return np.mean(np.abs(simulado - medido) / medido) * 100

# Cálculo dos erros médios percentuais
casos = {
    "0,5 cm": {
        r"$\mu_r=100$": erro_medio_percentual(medido_05, mu100_05),
        r"$\mu_r=500$": erro_medio_percentual(medido_05, mu500_05),
        r"$\mu_r=900$": erro_medio_percentual(medido_05, mu900_05),
        r"$\mu_r=1300$": erro_medio_percentual(medido_05, mu1300_05),
    },
    "1,0 cm": {
        r"$\mu_r=100$": erro_medio_percentual(medido_10, mu100_10),
        r"$\mu_r=500$": erro_medio_percentual(medido_10, mu500_10),
        r"$\mu_r=900$": erro_medio_percentual(medido_10, mu900_10),
        r"$\mu_r=1300$": erro_medio_percentual(medido_10, mu1300_10),
    }
}

print("Erro médio percentual absoluto:")
for distancia, erros in casos.items():
    print(f"\nDistância {distancia}")
    for caso, erro in erros.items():
        print(f"{caso}: {erro:.2f}%")

# Criação da figura
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)

# Gráfico 0,5 cm
axes[0].plot(corrente, medido_05, marker="o", label="Medido")
axes[0].plot(corrente, mu100_05, marker="s", label=r"$\mu_r=100$")
axes[0].plot(corrente, mu500_05, marker="^", label=r"$\mu_r=500$")
axes[0].plot(corrente, mu900_05, marker="D", label=r"$\mu_r=900$")
axes[0].plot(corrente, mu1300_05, marker="x", label=r"$\mu_r=1300$")
axes[0].set_title("Distância de 0,5 cm")
axes[0].set_xlabel("Corrente (A)")
axes[0].set_ylabel(r"Campo magnético axial $B_z$ ($\mu$T)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Gráfico 1,0 cm
axes[1].plot(corrente, medido_10, marker="o", label="Medido")
axes[1].plot(corrente, mu100_10, marker="s", label=r"$\mu_r=100$")
axes[1].plot(corrente, mu500_10, marker="^", label=r"$\mu_r=500$")
axes[1].plot(corrente, mu900_10, marker="D", label=r"$\mu_r=900$")
axes[1].plot(corrente, mu1300_10, marker="x", label=r"$\mu_r=1300$")
axes[1].set_title("Distância de 1,0 cm")
axes[1].set_xlabel("Corrente (A)")
axes[1].set_ylabel(r"Campo magnético axial $B_z$ ($\mu$T)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle("Comparação entre campo magnético medido e simulado", fontsize=14)
fig.tight_layout()

plt.savefig("comparacao_permeabilidade_nucleo.png", dpi=300, bbox_inches="tight")
plt.show()