"""Fase 1: el pendrive virtual (hot-plug USB) — tests.

Cubre: ids estables de las macro-primitivas nuevas, ciclo de vida del
conector (conectar -> montar -> desmontar -> desconectar), extracción
insegura anotada por el kernel, canales de percepción con
compatibilidad de prefijo (OBS 21 -> 24), la tarea DISPOSITIVO, el
anti-señuelo de recompensa, el oráculo y el TRASPLANTE de contrato
(incubadora.expandir_estado_contrato) que conserva intacto el
comportamiento del cerebro viejo en las tareas clásicas.
"""
import random

import pytest
import torch

from brooder.cerebro import CerebroBrooder
from brooder.constantes import (
    CURRICULO,
    MASCARA_ARGUMENTOS,
    MENSAJE_LOG_DISP_DESMONTADO,
    MENSAJE_LOG_DISP_MONTADO,
    MENSAJE_LOG_EXTRACCION_INSEGURA,
    MENSAJES_LOG,
    N_CANALES_DISPOSITIVO,
    N_MENSAJES_LOG,
    N_PRIMITIVAS,
    N_TAREAS,
    N_TAREAS_CLASICAS,
    OBS_DIM,
    PRESUPUESTO_CICLOS,
    Primitiva,
    Tarea,
)
from brooder.entorno import (
    EntornoBrooder,
    Oraculo,
    R_DISP_OK,
)
from brooder.incubadora import expandir_estado_contrato
from brooder.nucleo import NucleoBrooder
from brooder.percepcion import construir_observacion, nombre_de_canales
from brooder.primitivas.virtual import PCVirtual
from brooder.solicitudes import Solicitud


@pytest.fixture
def maquina():
    m = PCVirtual()
    m.reiniciar_registros()
    return m


def _episodio_de_dispositivo(modo: str):
    """Entorno listo con una solicitud DISPOSITIVO del modo pedido.

    Usa el camino REAL (reiniciar + sorteo de tarea) y espera a que
    salga el modo buscado (50 % por sorteo): ningún estado manual.
    """
    entorno = EntornoBrooder(tareas_activas=[Tarea.DISPOSITIVO], semilla=13)
    for _ in range(200):
        entorno.reiniciar()
        if entorno.solicitud.datos.get("modo") == modo:
            return entorno
    pytest.fail(f"no salió una solicitud de dispositivo en modo {modo}")


# ------------------------------------------------------------------
# contrato: ids estables y tabla
# ------------------------------------------------------------------
def test_ids_estables_para_compatibilidad():
    """Las macro-primitivas nuevas se añaden AL FINAL: nada se renumera."""
    assert int(Primitiva.REGISTRAR_LOG) == 17
    assert int(Primitiva.MONTAR_DISPOSITIVO) == 18
    assert int(Primitiva.DESMONTAR_DISPOSITIVO) == 19
    assert N_PRIMITIVAS == 20
    assert int(Primitiva.DESMONTAR_DISPOSITIVO) == N_PRIMITIVAS - 1


def test_mensajes_de_dispositivo_son_ids_estables():
    assert N_MENSAJES_LOG == 11
    assert MENSAJES_LOG[MENSAJE_LOG_DISP_MONTADO] == ("INFO", "dispositivo montado")
    assert (
        MENSAJES_LOG[MENSAJE_LOG_DISP_DESMONTADO]
        == ("INFO", "dispositivo desmontado")
    )
    assert (
        MENSAJES_LOG[MENSAJE_LOG_EXTRACCION_INSEGURA]
        == ("ERROR", "extraccion insegura")
    )
    # los 8 primeros mensajes siguen donde estaban (Fase 0/0.5)
    assert MENSAJES_LOG[1] == ("INFO", "lectura completada")
    assert MENSAJES_LOG[2] == ("INFO", "escritura completada")


def test_tabla_de_primitivas_documenta_las_nuevas():
    from brooder.constantes import TABLA_PRIMITIVAS

    for primitiva in (Primitiva.MONTAR_DISPOSITIVO, Primitiva.DESMONTAR_DISPOSITIVO):
        info = TABLA_PRIMITIVAS[primitiva]
        assert info.tipo_argumento == "ninguno"
        assert not info.usa_argumento
        assert info.descripcion


def test_mascara_de_argumentos_de_las_nuevas_primitivas():
    """Sin argumento: una sola opción (id 0), como NADA o LEER_CPU."""
    for primitiva in (Primitiva.MONTAR_DISPOSITIVO, Primitiva.DESMONTAR_DISPOSITIVO):
        permitidos = [i for i, ok in enumerate(MASCARA_ARGUMENTOS[primitiva]) if ok]
        assert permitidos == [0]


# ------------------------------------------------------------------
# ciclo de vida del conector (la máquina)
# ------------------------------------------------------------------
def test_montar_sin_dispositivo_da_error(maquina):
    assert not maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    assert "no hay dispositivo" in maquina.instante().ultimo_error


def test_ciclo_completo_montar_desmontar(maquina):
    assert maquina.conectar_dispositivo()
    assert maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    assert maquina.instante().dispositivo_montado
    # ya montado: rechazado como un mount doble
    assert not maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    assert maquina.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)
    assert not maquina.instante().dispositivo_montado
    assert maquina.instante().dispositivo_conectado
    # ya desmontado: rechazado
    assert not maquina.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)


def test_hotplug_es_externo_y_valida(maquina):
    assert maquina.conectar_dispositivo()
    assert not maquina.conectar_dispositivo()  # ya conectado
    assert maquina.desconectar_dispositivo()  # limpia
    assert not maquina.desconectar_dispositivo()  # conector vacío


def test_extraccion_insegura_registra_error_del_kernel(maquina):
    maquina.conectar_dispositivo()
    maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    maquina.avanzar_paso()
    # el mundo retira el pendrive MONTADO: extracción insegura
    assert not maquina.desconectar_dispositivo()
    instantes = maquina.instante()
    assert not instantes.dispositivo_conectado
    assert not instantes.dispositivo_montado
    mensajes = [m for _, m in instantes.registro]
    assert MENSAJE_LOG_EXTRACCION_INSEGURA in mensajes


def test_desconexion_limpia_no_registra_error(maquina):
    maquina.conectar_dispositivo()
    maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    maquina.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)
    assert maquina.desconectar_dispositivo()
    mensajes = [m for _, m in maquina.instante().registro]
    assert MENSAJE_LOG_EXTRACCION_INSEGURA not in mensajes


def test_el_kernel_anota_el_ciclo_del_dispositivo(maquina):
    """Montar/desmontar dejan su línea INFO en el registro (dmesg)."""
    maquina.conectar_dispositivo()
    maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    maquina.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)
    mensajes = [m for _, m in maquina.instante().registro]
    assert mensajes[-2:] == [MENSAJE_LOG_DISP_MONTADO, MENSAJE_LOG_DISP_DESMONTADO]


def test_el_estado_del_dispositivo_persiste_entre_solicitudes(maquina):
    """Es hardware enchufado: no se resetea como los registros del proceso."""
    maquina.conectar_dispositivo()
    maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    maquina.reiniciar_registros()
    instante = maquina.instante()
    assert instante.dispositivo_conectado
    assert instante.dispositivo_montado
    # arranque en frío: el conector vuelve a nacer vacío
    maquina.reiniciar()
    assert not maquina.instante().dispositivo_conectado


# ------------------------------------------------------------------
# percepción: canales nuevos con compatibilidad de prefijo
# ------------------------------------------------------------------
def _obs_de(maquina, tarea, solicitud, ciclos=10):
    return construir_observacion(
        maquina.instante(), tarea, solicitud, ciclos
    )


def test_dimensiones_del_vector_de_percepcion():
    assert OBS_DIM == N_TAREAS_CLASICAS + 16 + N_CANALES_DISPOSITIVO
    assert OBS_DIM == 24
    assert len(nombre_de_canales()) == OBS_DIM
    # las tareas del enum: 5 clásicas + DISPOSITIVO
    assert N_TAREAS == 6


def test_canales_de_dispositivo_reflejan_el_estado(maquina):
    maquina.escribir_teclado([1, 2, 3])
    sol = Solicitud(Tarea.ECO, tokens=[1, 2, 3], esperado=[1, 2, 3])
    obs = _obs_de(maquina, Tarea.ECO, sol)
    assert obs[21] == 0.0 and obs[22] == 0.0 and obs[23] == 0.0

    maquina.conectar_dispositivo()
    obs = _obs_de(maquina, Tarea.ECO, sol)
    assert obs[21] == 0.0 and obs[22] == 1.0 and obs[23] == 0.0

    maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    obs = _obs_de(maquina, Tarea.ECO, sol)
    assert obs[22] == 1.0 and obs[23] == 1.0

    disp = Solicitud(Tarea.DISPOSITIVO, datos={"modo": "montar"})
    obs = _obs_de(maquina, Tarea.DISPOSITIVO, disp)
    # canal de la tarea DISPOSITIVO: al final, no en el one-hot clásico
    assert obs[21] == 1.0
    assert sum(obs[:N_TAREAS_CLASICAS]) == 0.0


def test_prefijo_de_percepcion_es_el_contrato_viejo(maquina):
    """Las 21 primeras posiciones no dependen del estado del pendrive:
    un cerebro del contrato viejo percibe bit a bit lo que percibía."""
    maquina.escribir_teclado([10, 11, 12])
    sol = Solicitud(Tarea.ECO, tokens=[10, 11, 12], esperado=[10, 11, 12])
    obs_vacia = _obs_de(maquina, Tarea.ECO, sol)
    maquina.conectar_dispositivo()
    maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    obs_montado = _obs_de(maquina, Tarea.ECO, sol)
    assert obs_vacia[:21] == obs_montado[:21]
    assert obs_montado[22:] == [1.0, 1.0]


# ------------------------------------------------------------------
# la tarea DISPOSITIVO
# ------------------------------------------------------------------
def test_presupuesto_de_dispositivo():
    assert PRESUPUESTO_CICLOS["DISPOSITIVO"] == 12


def test_exito_de_la_solicitud_de_dispositivo(maquina):
    """La solicitud pide un ESTADO final: montado / listo para retirar.

    modo montar    -> exito cuando está montado y conectado.
    modo desmontar -> exito cuando está desmontado y AÚN CONECTADO
    (la extracción segura lo libera sin perderlo). Nota deliberada:
    si nunca llegó a montarse, "listo para retirar" se satisface de
    forma trivial — en entrenamiento y demo el modo desmontar SIEMPRE
    parte de montado (lo monta el entorno o la solicitud anterior),
    así que la política tiene que desmontar de verdad.
    """
    montar = Solicitud(Tarea.DISPOSITIVO, datos={"modo": "montar"})
    desmontar = Solicitud(Tarea.DISPOSITIVO, datos={"modo": "desmontar"})

    # sin dispositivo: nada satisface
    assert not montar.exito(maquina.instante())
    assert not desmontar.exito(maquina.instante())

    # conectado sin montar: montar aún no resuelve; el estado pedido
    # por desmontar ("listo para retirar") ya se cumple
    maquina.conectar_dispositivo()
    assert not montar.exito(maquina.instante())
    assert desmontar.exito(maquina.instante())
    maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    assert montar.exito(maquina.instante())
    assert not desmontar.exito(maquina.instante())

    # desmontado pero AÚN CONECTADO: extracción segura lograda
    maquina.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)
    assert not montar.exito(maquina.instante())
    assert desmontar.exito(maquina.instante())

    # si el pendrive desaparece (retirado), ya no hay éxito posible
    maquina.desconectar_dispositivo()
    assert not desmontar.exito(maquina.instante())


def test_generador_aleatorio_de_dispositivo(rng):
    for _ in range(40):
        s = Solicitud.aleatoria(Tarea.DISPOSITIVO, rng)
        assert s.tarea == Tarea.DISPOSITIVO
        assert s.tokens == [] and s.esperado == []
        assert s.datos["modo"] in ("montar", "desmontar")
        assert s.presupuesto == 12


def test_desde_texto_montar_y_desmontar():
    s = Solicitud.desde_texto("montar")
    assert s.tarea == Tarea.DISPOSITIVO and s.datos["modo"] == "montar"
    s = Solicitud.desde_texto("DESMONTAR")
    assert s.tarea == Tarea.DISPOSITIVO and s.datos["modo"] == "desmontar"


def test_curriculo_añade_dispositivo_al_final():
    assert Tarea.DISPOSITIVO in CURRICULO[-1]
    for etapa in CURRICULO[:-1]:
        assert Tarea.DISPOSITIVO not in etapa
    # retención: la etapa final conserva las 5 tareas clásicas
    assert all(t in CURRICULO[-1] for t in list(Tarea)[:N_TAREAS_CLASICAS])


# ------------------------------------------------------------------
# entorno: setup del hot-plug y recompensa
# ------------------------------------------------------------------
def test_el_entorno_enchufa_el_pendrive_en_tareas_de_dispositivo(rng):
    for modo in ("montar", "desmontar"):
        entorno = _episodio_de_dispositivo(modo)
        instante = entorno.maquina.instante()
        # el mundo enchufa el pendrive; en modo desmontar lo deja
        # además montado (lo montó "la sesión anterior")
        assert instante.dispositivo_conectado
        assert instante.dispositivo_montado == (modo == "desmontar")
        # y la observación refleja ese estado en los canales nuevos
        obs = entorno.observar()
        assert obs[21] == 1.0
        assert obs[22] == 1.0
        assert obs[23] == (1.0 if modo == "desmontar" else 0.0)


def test_montar_correcto_premia_y_solo_en_tarea_dispositivo(rng):
    # premio en la tarea DISPOSITIVO (modo montar)
    entorno = _episodio_de_dispositivo("montar")
    assert entorno.solicitud.tarea == Tarea.DISPOSITIVO
    _, recompensa, terminada, info = entorno.paso(
        Primitiva.MONTAR_DISPOSITIVO, 0
    )
    assert recompensa >= R_DISP_OK
    assert terminada and info["exito"]

    # anti-señuelo: en ECO no hay pendrive ni premio de dispositivo
    entorno = EntornoBrooder(tareas_activas=[Tarea.ECO], semilla=3)
    entorno.reiniciar()
    _, recompensa, _, _ = entorno.paso(Primitiva.MONTAR_DISPOSITIVO, 0)
    # MONTAR fracasa en la máquina (conector vacío) y no premia:
    # solo queda el ciclo y (como mucho) la curiosidad de novedad
    assert recompensa < R_DISP_OK


def test_desmontar_correcto_premia(rng):
    entorno = _episodio_de_dispositivo("desmontar")
    _, recompensa, terminada, info = entorno.paso(
        Primitiva.DESMONTAR_DISPOSITIVO, 0
    )
    assert recompensa >= R_DISP_OK
    assert terminada and info["exito"]


def test_direccion_equivoca_no_premia(rng):
    """En modo montar, DESMONTAR fracasa en la máquina: sin premio."""
    entorno = _episodio_de_dispositivo("montar")
    _, recompensa, terminada, info = entorno.paso(
        Primitiva.DESMONTAR_DISPOSITIVO, 0
    )
    assert recompensa < R_DISP_OK
    assert not terminada


def test_oraculo_administra_el_dispositivo(rng):
    """El programa ideal resuelve la tarea de dispositivo en ambos modos."""
    for modo in ("montar", "desmontar"):
        entorno = _episodio_de_dispositivo(modo)
        oraculo = Oraculo(entorno.solicitud)
        terminada = False
        while not terminada:
            prim, arg = oraculo.accion()
            oraculo.avanzar()
            _, _, terminada, info = entorno.paso(prim, arg)
        assert info["exito"], f"el oráculo falló en modo {modo}"


def test_oraculo_resuelve_todo_con_dispositivo(rng):
    """La suite clásica: el oráculo alcanza el 100 % en las 6 tareas."""
    entorno = EntornoBrooder(
        tareas_activas=list(Tarea), semilla=42
    )
    fallos = []
    for _ in range(240):
        entorno.reiniciar()
        oraculo = Oraculo(entorno.solicitud)
        terminada = False
        while not terminada:
            prim, arg = oraculo.accion()
            oraculo.avanzar()
            _, _, terminada, info = entorno.paso(prim, arg)
        if not info["exito"]:
            fallos.append((info["tarea"], info["causa"]))
    assert not fallos, f"El oráculo falló en: {fallos[:5]}"


# ------------------------------------------------------------------
# compatibilidad: cerebros del contrato viejo
# ------------------------------------------------------------------
def test_nucleo_recorta_la_observacion_para_cerebros_viejos():
    """Un cerebro de 21 entradas atiende tareas clásicas sin explotar
    (la observación se recorta al prefijo que conoce)."""
    torch.manual_seed(1)
    cerebro_viejo = CerebroBrooder(
        dim_entrada=OBS_DIM - N_CANALES_DISPOSITIVO,
        n_primitivas=18,
    )
    nucleo = NucleoBrooder(
        PCVirtual(), cerebro_viejo, registro_eventos=False
    )
    resultado = nucleo.atender_solicitud(Solicitud.desde_texto("HOLA"))
    assert resultado.causa in ("exito", "presupuesto_agotado")
    assert resultado.ciclos == resultado.solicitud.presupuesto


def test_el_trasplante_conserva_las_decisiones_clasicas():
    """expandir_estado_contrato: el cerebro migrado decide EXACTAMENTE
    igual que el viejo en las tareas clásicas (canales nuevos en 0)."""
    torch.manual_seed(2)
    viejo = CerebroBrooder(dim_entrada=21, n_primitivas=18)
    viejo.eval()
    nuevo = CerebroBrooder()  # contrato actual: 24 entradas / 20 salidas
    nuevo.load_state_dict(expandir_estado_contrato(viejo.state_dict(), 21, 18))
    nuevo.eval()

    rng = random.Random(9)
    for _ in range(25):
        obs = [rng.uniform(-1, 1) for _ in range(OBS_DIM)]
        obs[21:] = [0.0, 0.0, 0.0]  # sin señal de dispositivo
        with torch.no_grad():
            h1, M1 = viejo.estado_inicial()
            h2, M2 = nuevo.estado_inicial()
            # 4 ciclos seguidos: la recurrencia también debe coincidir
            for _ in range(4):
                p1, a1, _, h1, M1 = viejo.decidir(
                    torch.tensor(obs[:21]), h1, M1, determinista=True
                )
                p2, a2, _, h2, M2 = nuevo.decidir(
                    torch.tensor(obs), h2, M2, determinista=True
                )
                assert int(p1) == int(p2)
                assert int(a1) == int(a2)


def test_el_trasplante_arranca_desfavoreciendo_las_primitivas_nuevas():
    torch.manual_seed(3)
    viejo = CerebroBrooder(dim_entrada=21, n_primitivas=18)
    viejo.eval()
    nuevo = CerebroBrooder()
    nuevo.load_state_dict(expandir_estado_contrato(viejo.state_dict(), 21, 18))
    nuevo.eval()
    assert int(nuevo.cabeza_primitiva.bias[Primitiva.MONTAR_DISPOSITIVO]) == -4.0
    assert int(nuevo.cabeza_primitiva.bias[Primitiva.DESMONTAR_DISPOSITIVO]) == -4.0
    # las filas de las primitivas clásicas no se tocan
    assert torch.allclose(
        nuevo.cabeza_primitiva.weight[:18], viejo.cabeza_primitiva.weight
    )
    assert torch.allclose(
        nuevo.cabeza_primitiva.bias[:18], viejo.cabeza_primitiva.bias
    )


def test_post_lista_el_conector_usb():
    nucleo = NucleoBrooder(PCVirtual(), CerebroBrooder(), registro_eventos=False)
    nombres = " ".join(n for n, _ in nucleo.post())
    assert "Conector USB" in nombres
    assert "vacío" in " ".join(d for _, d in nucleo.post())


# ------------------------------------------------------------------
# demo: sección del pendrive
# ------------------------------------------------------------------
def test_demo_pendrive_con_oraculo():
    """El flujo completo de la sección Fase 1 con una política perfecta."""
    from tests.test_trazado import _CerebroOraculo

    maquina = PCVirtual()
    solicitud_montar = Solicitud(Tarea.DISPOSITIVO, datos={"modo": "montar"})
    nucleo = NucleoBrooder(
        maquina, _CerebroOraculo(solicitud_montar), registro_eventos=False
    )
    maquina.conectar_dispositivo()
    r1 = nucleo.atender_solicitud(solicitud_montar)
    assert r1.exito and maquina.dispositivo_montado

    solicitud_desmontar = Solicitud(
        Tarea.DISPOSITIVO, datos={"modo": "desmontar"}
    )
    nucleo.cerebro = _CerebroOraculo(solicitud_desmontar)
    r2 = nucleo.atender_solicitud(solicitud_desmontar)
    assert r2.exito and not maquina.dispositivo_montado
    assert maquina.dispositivo_conectado

    # el panel del registro contiene el ciclo del dispositivo
    mensajes = [m for _, m in maquina.instante().registro]
    assert MENSAJE_LOG_DISP_MONTADO in mensajes
    assert MENSAJE_LOG_DISP_DESMONTADO in mensajes
