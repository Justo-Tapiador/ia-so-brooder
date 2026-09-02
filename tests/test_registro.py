"""Tests del registro del sistema (macro-primitiva REGISTRAR_LOG).

Fase 0 del escalado del contrato de hardware: la primera
macro-primitiva. Cubre el vocabulario cerrado de mensajes, el anillo
del kernel, el panel de la TUI, el aislamiento del estado de usuario
y la compatibilidad de los cerebros incubados con el contrato viejo
(17 salidas).
"""
import pytest
import torch

from brooder.constantes import (
    MENSAJES_LOG,
    MASCARA_ARGUMENTOS,
    N_MENSAJES_LOG,
    N_PRIMITIVAS,
    Primitiva,
    REGISTRO_CAPACIDAD,
    TABLA_PRIMITIVAS,
    formatear_registro,
)
from brooder.primitivas.virtual import PCVirtual


@pytest.fixture
def maquina():
    m = PCVirtual()
    m.reiniciar_registros()
    return m


# ------------------------------------------------------------------
# contrato
# ------------------------------------------------------------------
def test_ids_estables_para_compatibilidad():
    """Las macro-primitivas se AÑADEN AL FINAL: nada se renumera.

    Fase 0 añadió REGISTRAR_LOG (17) y la Fase 1 añadió MONTAR y
    DESMONTAR (18, 19). Los cerebros incubados con contratos viejos
    (17 o 18 salidas) siguen montando: sus cabezas no alcanzan los
    ids nuevos, y sus ids siguen significando lo mismo.
    """
    assert Primitiva.LEER_RED == 16
    assert Primitiva.REGISTRAR_LOG == 17
    assert int(Primitiva.MONTAR_DISPOSITIVO) == 18
    assert int(Primitiva.DESMONTAR_DISPOSITIVO) == 19
    # Fase 1.5: el plan de datos del pendrive cierra el enum
    assert int(Primitiva.ESCRIBIR_DISPOSITIVO) == 22
    assert N_PRIMITIVAS == 23
    assert set(TABLA_PRIMITIVAS) == set(Primitiva)
    assert TABLA_PRIMITIVAS[Primitiva.REGISTRAR_LOG].tipo_argumento == "mensaje"
    assert len(MENSAJES_LOG) == N_MENSAJES_LOG


def test_mascara_solo_permite_ids_de_mensaje():
    """El contrato de tipos filtra la exploración a mensajes válidos.

    Igual que las primitivas de datos exigen BUS: las combinaciones
    que la máquina siempre rechazaría ni siquiera se proponen.
    """
    permitidos = [
        i for i, v in enumerate(MASCARA_ARGUMENTOS[Primitiva.REGISTRAR_LOG]) if v
    ]
    assert permitidos == list(range(N_MENSAJES_LOG))


# ------------------------------------------------------------------
# anillo del kernel
# ------------------------------------------------------------------
def test_registrar_log_anade_entrada(maquina):
    assert maquina.ejecutar(Primitiva.REGISTRAR_LOG, 3)
    assert maquina.instante().registro == ((0, 3),)
    assert "proceso iniciado" in maquina.instante().ultimo_evento


def test_registrar_log_rechaza_mensaje_desconocido(maquina):
    assert not maquina.ejecutar(Primitiva.REGISTRAR_LOG, N_MENSAJES_LOG)
    assert "mensaje" in maquina.instante().ultimo_error


def test_el_anillo_rota_con_capacidad_fija(maquina):
    total = REGISTRO_CAPACIDAD + 3
    for _ in range(total):
        assert maquina.ejecutar(Primitiva.REGISTRAR_LOG, 1)
        maquina.avanzar_paso()
    registro = maquina.instante().registro
    assert len(registro) == REGISTRO_CAPACIDAD
    # las más viejas se pierden; el anillo retiene las últimas
    assert registro[-1][0] == total - 1
    assert registro[0][0] == total - REGISTRO_CAPACIDAD


# ------------------------------------------------------------------
# aislamiento: no toca el estado del usuario
# ------------------------------------------------------------------
def test_registrar_no_toca_estado_de_usuario(maquina):
    maquina.escribir_teclado([1])
    maquina.leer_teclado()  # bus cargado con un token leído
    estado_pantalla = maquina.instante().pantalla
    assert maquina.ejecutar(Primitiva.REGISTRAR_LOG, 0)
    instante = maquina.instante()
    assert instante.pantalla == estado_pantalla == []
    assert instante.acumulador == 0
    # el bus conserva su contenido: la consola no es un dispositivo de datos
    assert instante.bus_valido and instante.bus_valor == 1


# ------------------------------------------------------------------
# persistencia (como dmesg) y panel
# ------------------------------------------------------------------
def test_registro_persiste_entre_solicitudes(maquina):
    """La consola del kernel no se limpia con los registros volátiles."""
    maquina.ejecutar(Primitiva.REGISTRAR_LOG, 3)
    maquina.reiniciar_registros()
    assert maquina.instante().registro == ((0, 3),)
    # ... pero un arranque en frío sí la vacía
    maquina.reiniciar()
    assert maquina.instante().registro == ()


def test_panel_muestra_las_ultimas_lineas(maquina):
    for mensaje in (0, 1, 2, 5, 7):
        maquina.ejecutar(Primitiva.REGISTRAR_LOG, mensaje)
        maquina.avanzar_paso()
    lineas = maquina.panel_registro()
    assert len(lineas) == 4
    assert all(isinstance(linea, str) for linea in lineas)
    # la más vieja (mensaje 0, paso 0) ya no cabe en el panel...
    assert formatear_registro(0, 0) not in lineas
    # ...y la última sí, en la línea inferior
    assert lineas[-1] == formatear_registro(4, 7)


# ------------------------------------------------------------------
# compatibilidad: los cerebros viejos siguen montando
# ------------------------------------------------------------------
def test_cerebro_con_contrato_viejo_no_alcanza_la_nueva_primitiva():
    """montar_ssd construye el cerebro desde la config guardada.

    Los SSD incubados antes de esta primitiva cargan igual (su
    n_primitivas propio) y simplemente no pueden emitir el id 17:
    la macro-primitiva queda disponible para los cerebros que se
    reentrenen con el contrato nuevo.
    """
    from brooder.cerebro import CerebroBrooder

    cerebro_viejo = CerebroBrooder(n_primitivas=17)
    assert cerebro_viejo.n_primitivas == 17
    assert int(Primitiva.REGISTRAR_LOG) >= cerebro_viejo.n_primitivas

    cerebro_nuevo = CerebroBrooder()
    assert cerebro_nuevo.n_primitivas == N_PRIMITIVAS


def test_cerebro_nuevo_puede_decidir_registrar_log():
    """La cabeza nueva alcanza el id 17 y la máscara limita el argumento."""
    from brooder.cerebro import CerebroBrooder, enmascarar_logits_argumento

    cerebro = CerebroBrooder(oculto=16, mem_ranuras=2, mem_dim=4)
    h, M = cerebro.estado_inicial()
    obs = torch.zeros(1, cerebro.dim_entrada)
    logits_p, g, _valor, h, M = cerebro.paso(obs, h, M)
    assert logits_p.shape[-1] == N_PRIMITIVAS

    prim = torch.full((1,), int(Primitiva.REGISTRAR_LOG), dtype=torch.long)
    logits_a = enmascarar_logits_argumento(cerebro.logits_argumento(g, prim), prim)
    validos = [
        i for i in range(logits_a.shape[-1]) if not torch.isinf(logits_a[0, i])
    ]
    assert validos == list(range(N_MENSAJES_LOG))
