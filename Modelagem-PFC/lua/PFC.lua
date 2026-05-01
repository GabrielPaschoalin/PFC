-- Parâmetros (em cm)
local h_nucleo = 11        -- altura do núcleo
local r_nucleo = 0.8      -- raio do núcleo
local saida_nucleo = 0.35

-- Bobina (espessura radial a partir do núcleo)
local r_bobina  =  5.5/2   -- largura radial da bobina
local h_bobina  = 9  -- altura da bobina
local y_bobina0 = 0    -- base inferior da bobina

-- Envoltório (1010 Steel) ao redor da bobina
local t_env = 0.03        -- espessura do envoltório

--  Semi-esfera (parâmetros em cm) ====
local raio_esfera     = 1.7/2      -- raio da esfera
local distancia_esfera = -0.5      -- distância vertical do centro até a base do núcleo

-- Envoltório externo
local r_env        = h_nucleo*1.5      -- raio do envoltório (garanta folga > 2x maior que a peça)
local z_env_centro = 0---h_nucleo/2  -- centro vertical (pode ajustar)

-- ===== Parâmetros da bobina =====
local N_turns   = 2000     -- nº de espiras
local I_coil    = 1      -- corrente [A]
local mesh_bob  = 0.1      -- malha alvo na bobina (cm)

-- ================== CÓDIGO =========================================

-- Novo documento
newdocument(0)
mi_probdef(0, "centimeters", "axi", 1e-8)

-- MATERIAIS
mi_addmaterial("1010 Steel (mi100)", 100, 100, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0)
mi_addmaterial("1010 Steel (linear)", 902.6, 902.6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
mi_getmaterial("Air")          -- domínio externo
mi_getmaterial("22 AWG")

-- NÚCLEO
local nucleo_altura_positiva = h_nucleo - saida_nucleo

mi_addnode(0,-saida_nucleo);  mi_addnode(0,nucleo_altura_positiva)
mi_addnode(r_nucleo,nucleo_altura_positiva);  mi_addnode(r_nucleo,0)
mi_addnode(r_nucleo,-saida_nucleo)
mi_addsegment(0,nucleo_altura_positiva,r_nucleo,nucleo_altura_positiva)
mi_addsegment(r_nucleo,nucleo_altura_positiva,r_nucleo,-saida_nucleo); 
mi_addsegment(r_nucleo,-saida_nucleo,0,-saida_nucleo); 

mi_clearselected()

mi_addblocklabel(r_nucleo/2, h_nucleo/2)  -- posição interna do núcleo
mi_selectlabel(r_nucleo/2, h_nucleo/2)
mi_setblockprop("1010 Steel (mi100)", 1, 0, "", 0, 0, 0)


-- Bobina (retângulo: de x=r até x=r+r_bobina; de y=y_bobina0 até y+y_h)
local x1 = r_nucleo
local x2 = r_bobina
local y1 = y_bobina0
local y2 = y_bobina0 + h_bobina

mi_addnode(x1,y1); mi_addnode(x2,y1)
mi_addnode(x2,y2); mi_addnode(x1,y2)
mi_addsegment(x1,y1,x2,y1)
mi_addsegment(x2,y1,x2,y2)
mi_addsegment(x2,y2,x1,y2)
mi_addsegment(x1,y2,x1,y1)

-- Cria o circuito da bobina (série)
mi_addcircprop("Coil", I_coil, 1)

-- Define bloco da bobina (região entre x1,x2,y1,y2)
mi_addblocklabel( (x1+x2)/2, (y1+y2)/2 )
mi_selectlabel(  (x1+x2)/2, (y1+y2)/2 )
mi_setblockprop("22 AWG", 0, mesh_bob, "Coil", 0, 0, N_turns)
mi_clearselected()


-- Envoltório: cresce t_env para fora (x2->x2+t_env) e para cima/baixo (y1- t_env, y2+ t_env)
local xo = x2 + t_env      -- limite radial externo do envoltório
local yb = y1 - t_env      -- base inferior do envoltório
local yt = y2 + t_env      -- topo do envoltório

-- Tampa superior
mi_addnode(x1, y2); mi_addnode(xo, y2)
mi_addnode(xo, yt); mi_addnode(x1, yt)
mi_addsegment(xo,y2, xo,yt)
mi_addsegment(xo,yt, x1,yt)

-- Parede direita
mi_addnode(x2, y1); mi_addnode(xo, y1)
mi_addnode(xo, y2); mi_addnode(x2, y2)
mi_addsegment(xo,y1, xo,y2)
-- mi_addsegment(x2,y2  , x2,y1)
mi_addsegment(x1,yb, xo,yb)

mi_addblocklabel( x2 + t_env/2, (y1+y2)/2 )
mi_selectlabel(  x2 + t_env/2, (y1+y2)/2 )
mi_setblockprop("1010 Steel (linear)", 1, 0, "", 0, 0, 0) -- ***default da biblioteca***
mi_clearselected()


-- ==== Semi-esfera (parâmetros em cm) ==== 

local z_esfera_centro = distancia_esfera - raio_esfera - saida_nucleo
local yb_esf = z_esfera_centro - raio_esfera -- ponto inferior no eixo
local yt_esf = z_esfera_centro + raio_esfera -- ponto superior no eixo 

mi_addnode(0, yb_esf) mi_addnode(0, yt_esf) 

-- -- arco semicircular voltado para a direita 
mi_addarc(0, yb_esf, 0, yt_esf, 180, 1)

-- -- fecha pela linha do eixo (necessário para formar a região) 
mi_addsegment(0, yb_esf, 0, yt_esf) 

mi_addblocklabel( 0.3*raio_esfera, z_esfera_centro ) -- um ponto seguro dentro 
mi_selectlabel( 0.3*raio_esfera, z_esfera_centro )
mi_setblockprop("1010 Steel (linear)", 1, 0, "", 0, 4, 0) -- ***default da biblioteca*** mi_clearselected()

mi_clearselected()

-- ===== Envoltório externo (AR) do sistema =====
-- nós extremos do diâmetro no eixo r=0
local yb_env = z_env_centro - r_env
local yt_env = z_env_centro + r_env
mi_addnode(0, yb_env)
mi_addnode(0, yt_env)

-- arco externo (semicírculo para a direita) + fechamento pelo eixo
mi_addarc(0, yb_env, 0, yt_env, 180, 40)
mi_addsegment(0, yb_env, 0, yt_env)

mi_addblocklabel(r_env*0.5, z_env_centro)   -- ponto dentro do ar
mi_selectlabel(r_env*0.5, z_env_centro)
mi_setblockprop("Air", 1, 0, "", 0, 0, 0)
mi_clearselected()


-- Grade/zoom
mi_setgrid(0.05, "cart")
mi_zoomnatural()


-- -- ===============================================================
-- -- === SOLUÇÃO E CÁLCULOS ========================================
-- -- ===============================================================

-- 1. Resolve o problema
mi_saveas("solenoide_unico.fem")
mi_analyze(1) 
mi_loadsolution()

-- 2. Obtém a Força (Fy) e a Indutância (L)
-- Força magnética na esfera
mo_groupselectblock(4)
local Fy = mo_blockintegral(19) -- (19) = Força na direção y (N)
mi_clearselected()

-- Indutância (L)
-- Henry (H) = Wb / A ( Weber por Ampere) - Fluxo magnético por corrente 
local flux, volt, amps, energy = mo_getcircuitproperties("Coil")
local L = flux / amps

-- 3. Imprime os resultados
print("\n--- Resultados de Simulação ---")
print("Corrente (I): " .. I_coil .. " A")
print("Distância Centro (d): " .. distancia_esfera .. " cm")
print("Força Magnética (Fy): " .. Fy .. " N")
print("Indutância (L): " .. L .. " H")