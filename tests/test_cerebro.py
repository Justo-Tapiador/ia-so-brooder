"""Tests del cerebro (arquitectura, máscaras, persistencia)."""
import torch

from brooder.cerebro import (
    CerebroBrooder,
    enmascarar_logits_argumento,
    entropia_conjunta,
)
from brooder.constantes import (
    ARG_BUS,
    MASCARA_ARGUMENTOS,
    N_ARGUMENTOS,
    N_PRIMITIVAS,
    OBS_DIM,
    Primitiva,
)


# ------------------------------------------------------------------
# arquitectura
# ------------------------------------------------------------------
def test_dimensiones_de_fabrica():
    cerebro = CerebroBrooder()
    assert cerebro.dim_entrada == OBS_DIM
    assert cerebro.n_primitivas == N_PRIMITIVAS
    assert cerebro.n_argumentos == N_ARGUMENTOS


def test_paso_produce_formas_correctas():
    cerebro = CerebroBrooder()
    h, M = cerebro.estado_inicial(1)
    obs = torch.zeros(1, OBS_DIM)
    logits_p, g, valor, h2, M2 = cerebro.paso(obs, h, M)
    assert logits_p.shape == (1, N_PRIMITIVAS)
    assert valor.shape == (1,)
    assert h2.shape == (1, cerebro.oculto)
    assert M2.shape == (1, cerebro.mem_ranuras, cerebro.mem_dim)
    # sin NaN
    assert torch.isfinite(logits_p).all()
    assert torch.isfinite(M2).all()


def test_memoria_asociativa_cambia():
    """La memoria direccionable por contenido debe evolucionar."""
    cerebro = CerebroBrooder()
    h, M = cerebro.estado_inicial(1)
    obs = torch.randn(1, OBS_DIM)
    _, _, _, _, M2 = cerebro.paso(obs, h, M)
    assert not torch.allclose(M, M2), "la memoria no se actualizó"


def test_secuencia_sigue_la_pista():
    cerebro = CerebroBrooder()
    h, M = cerebro.estado_inicial(1)
    obs_seq = torch.randn(1, 7, OBS_DIM)
    logits_p, g, valores, h_f, M_f = cerebro.secuencia(obs_seq, h, M)
    assert logits_p.shape == (1, 7, N_PRIMITIVAS)
    assert valores.shape == (1, 7)


# ------------------------------------------------------------------
# máscaras de tipos
# ------------------------------------------------------------------
def test_mascara_datos_solo_bus():
    assert MASCARA_ARGUMENTOS[int(Primitiva.MOSTRAR_EN_PANTALLA)][ARG_BUS]
    assert sum(MASCARA_ARGUMENTOS[int(Primitiva.MOSTRAR_EN_PANTALLA)]) == 1
    assert sum(MASCARA_ARGUMENTOS[int(Primitiva.ESCRIBIR_DISCO)]) == 1
    assert sum(MASCARA_ARGUMENTOS[int(Primitiva.CPU_SUMAR)]) == 1


def test_mascara_direccion_literal_y_bus():
    m = MASCARA_ARGUMENTOS[int(Primitiva.MOVER_CABEZAL_DISCO)]
    assert sum(m) == 11  # 0..9 + BUS
    assert m[ARG_BUS] and m[0] and m[9]


def test_enmascarar_quita_probabilidad_a_lo_invalido():
    cerebro = CerebroBrooder()
    logits = torch.zeros(1, N_ARGUMENTOS)
    prim = torch.tensor([int(Primitiva.MOSTRAR_EN_PANTALLA)])
    enmascarados = enmascarar_logits_argumento(logits, prim)
    assert enmascarados[0, ARG_BUS] == 0.0
    assert torch.isinf(enmascarados[0, 0]) and enmascarados[0, 0] < 0


def test_decidir_solo_propone_argumentos_validos():
    cerebro = CerebroBrooder()
    h, M = cerebro.estado_inicial(1)
    obs = torch.zeros(OBS_DIM)
    for _ in range(200):
        prim, arg, _, h, M = cerebro.decidir(obs, h, M, determinista=False)
        assert MASCARA_ARGUMENTOS[prim][arg], (
            f"propuso ({Primitiva(prim).name}, {arg}) que el kernel rechazaría"
        )


def test_entropia_segura_con_infinitos():
    logits_p = torch.zeros(2, 3)
    logits_a = torch.zeros(2, N_ARGUMENTOS)
    logits_a[:, 1:] = float("-inf")
    ent = entropia_conjunta(logits_p, logits_a)
    assert torch.isfinite(ent).all()
    assert ent[0] > 0  # la única opción válida no aporta entropía de argumento


# ------------------------------------------------------------------
# persistencia
# ------------------------------------------------------------------
def test_guardar_y_cargar(tmp_path):
    cerebro = CerebroBrooder(oculto=32, mem_ranuras=4, mem_dim=8)
    ruta = tmp_path / "cerebro.pt"
    cerebro.guardar(ruta)
    cargado = CerebroBrooder.cargar(ruta)
    assert cargado.oculto == 32
    assert cargado.mem_ranuras == 4

    obs = torch.randn(1, OBS_DIM)
    h1, M1 = cargado.estado_inicial(1)
    h2, M2 = cerebro.estado_inicial(1)
    lp1, _, _, _, _ = cargado.paso(obs, h1, M1)
    lp2, _, _, _, _ = cerebro.paso(obs, h2, M2)
    assert torch.allclose(lp1, lp2)


# ------------------------------------------------------------------
# el sesgo inductivo del BUS existe pero no decide
# ------------------------------------------------------------------
def test_sesgo_bus_moderado():
    """En direccionamiento (11 opciones), p(BUS) inicial debe
    superar claramente el uniforme (1/11) sin llegar a imponerse."""
    cerebro = CerebroBrooder()
    h, M = cerebro.estado_inicial(1)
    obs = torch.zeros(1, OBS_DIM)
    logits_p, g, _, h, M = cerebro.paso(obs, h, M)
    prim = torch.tensor([int(Primitiva.MOVER_CABEZAL_DISCO)])
    logits_a = cerebro.logits_argumento(g, prim)  # sin enmascarar
    p_bus = torch.softmax(logits_a, -1)[0, ARG_BUS].item()
    assert p_bus > 1.0 / 11.0  # ventaja clara sobre el uniforme
    assert p_bus < 0.99        # pero no imposición
