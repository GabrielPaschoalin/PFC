import math

# ALTERAR
altura_bobina_cm = 4 

# CONSTANTES
raio_interno_mm= 8
espessura_fio_mm=0.643 # 22 AWG
altura_bobina_mm= altura_bobina_cm * 10
numero_voltas=2000

# Calcula o número de voltas por camada (se VPC for 0, a altura é pequena demais )
voltas_por_camada = math.floor(altura_bobina_mm / espessura_fio_mm)

if voltas_por_camada <= 0:
    raise ValueError("Altura insuficiente para ao menos 1 volta por camada com o passo definido.")

numero_camadas = math.ceil(numero_voltas / voltas_por_camada)
raio_externo_mm = (
    raio_interno_mm
    + numero_camadas * espessura_fio_mm
)


diametro_externo_mm = 2 * raio_externo_mm

resultado =  {
    "atura em cm": altura_bobina_cm,
    "voltas_por_camada": voltas_por_camada,
    "numero_camadas": numero_camadas,
    "raio em cm ": raio_externo_mm / 10,
    "raio + 100% em cm": (raio_externo_mm * 2) / 10,
}

for k, v in resultado.items():
    print(f"{k}: {v}")