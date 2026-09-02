"""Tests del trazado del registro (Fase 0.5).

La Fase 0.5 enseña al cerebro a usar su primera syscall: el entorno
premia emitir REGISTRAR_LOG con el mensaje correcto justo después de
una operación de almacenamiento (trazado). Cubre:

* el premio exacto (y su acotación: 1 por tipo y solicitud),
* la ventana temporal y el mensaje correcto,
* el oráculo como política ideal que traza el 100 %,
* la evidencia del trazado en el núcleo (resultado.trazos).
"""
import pytest
import torch

from brooder.constantes import (
    ARG_BUS,
    MENSAJE_LOG_ESCRITURA,
    MENSAJE_LOG_LECTURA,
    Primitiva,
    Tarea,
)
from brooder.entorno import (
    EntornoBrooder,
    Oraculo,
    R_TRAZO_RUIDO,
    R_TRAZO_VALIDO,
)
from brooder.nucleo import NucleoBrooder
from brooder.primitivas.virtual import PCVirtual
from brooder.solicitudes import Solicitud


def _prefijo_guardar(entorno):
    """Ejecuta el cuerpo de GUARDAR sin trazado: K, cabezal, V, escritura."""
    entorno.paso(Primitiva.LEER_TECLADO, 0)                # K
    entorno.paso(Primitiva.MOVER_CABEZAL_DISCO, ARG_BUS)   # cabezal -> K
    entorno.paso(Primitiva.LEER_TECLADO, 0)                # V -> bus
    entorno.paso(Primitiva.ESCRIBIR_DISCO, ARG_BUS)        # disco[K] <- V


def _entorno_guardar_neutro(semilla: int) -> EntornoBrooder:
    """Entorno GUARDAR con la curiosidad de REGISTRAR_LOG apagada.

    Así las comparaciones de recompensa son exactas: el bonus 1/sqrt(n)
    de la curiosidad no contamina la medida del premio de trazado.
    """
    entorno = EntornoBrooder(tareas_activas=[Tarea.GUARDAR], semilla=semilla)
    entorno.reiniciar()
    entorno._conteos_novedad[Primitiva.REGISTRAR_LOG] = 10**9
    return entorno


# ------------------------------------------------------------------
# el premio de trazado
# ------------------------------------------------------------------
def test_trazar_la_escritura_es_premiado():
    """REGISTRAR_LOG(escritura) tras escribir gana exactamente R_TRAZO."""
    recompensas = {}
    for etiqueta, prim, arg in (
        ("nada", Primitiva.NADA, 0),
        ("trazo", Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA),
    ):
        entorno = _entorno_guardar_neutro(123)
        _prefijo_guardar(entorno)
        _, r, _, info = entorno.paso(prim, arg)
        recompensas[etiqueta] = r
        assert not info["exito"]  # la solicitud aún no termina
    delta = recompensas["trazo"] - recompensas["nada"]
    assert delta == pytest.approx(R_TRAZO_VALIDO, abs=1e-4)


def test_trazar_la_lectura_es_premiada():
    """REGISTRAR_LOG(lectura) tras leer del disco también gana el premio."""
    recompensas = {}
    for etiqueta, prim, arg in (
        ("nada", Primitiva.NADA, 0),
        ("trazo", Primitiva.REGISTRAR_LOG, MENSAJE_LOG_LECTURA),
    ):
        entorno = _entorno_guardar_neutro(321)
        _prefijo_guardar(entorno)
        entorno.paso(Primitiva.LEER_DISCO, 0)  # lectura del dato guardado
        _, r, _, _ = entorno.paso(prim, arg)
        recompensas[etiqueta] = r
    assert recompensas["trazo"] - recompensas["nada"] == pytest.approx(
        R_TRAZO_VALIDO, abs=1e-4
    )


def test_mensaje_equivocado_no_premia():
    """Trazar 'lectura' tras una escritura no gana nada... y luego se
    puede rectificar dentro de la ventana."""
    entorno = _entorno_guardar_neutro(77)
    _prefijo_guardar(entorno)
    _, r_mal, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_LECTURA)
    # sin premio: solo el coste del ciclo
    assert r_mal == pytest.approx(-0.01, abs=1e-4)
    # ...pero la operación sigue pendiente: trazar bien a tiempo premia
    _, r_bien, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)
    assert r_bien == pytest.approx(-0.01 + R_TRAZO_VALIDO, abs=1e-4)


def test_traza_tardia_no_premia():
    """Fuera de la ventana (más de TRAZO_VENTANA ciclos) no hay premio."""
    from brooder.entorno import TRAZO_VENTANA

    entorno = _entorno_guardar_neutro(55)
    _prefijo_guardar(entorno)
    for _ in range(TRAZO_VENTANA + 2):  # esperar más de la ventana
        entorno.paso(Primitiva.NADA, 0)
    _, r, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)
    assert r == pytest.approx(-0.01, abs=1e-4)


def test_como_maximo_un_premio_por_tipo_y_solicitud():
    """El premio es acotado: no se puede cultivar con log-spam."""
    entorno = _entorno_guardar_neutro(99)
    _prefijo_guardar(entorno)
    _, r_1, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)
    assert r_1 > 0.05  # primer trazo premiado
    # re-escribe y vuelve a trazar: nada de premio, y además el
    # mensaje repetido penaliza (supresión del spam de repeticiones)
    entorno.paso(Primitiva.ESCRIBIR_DISCO, ARG_BUS)
    _, r_2, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)
    assert r_2 == pytest.approx(-0.01 + R_TRAZO_RUIDO, abs=1e-4)


def test_mensaje_repetido_penaliza_y_el_distinto_no():
    """El ruido es por MENSAJE repetido, no por registrar de nuevo.

    Patología real de la incubación: el argmax caía en un bucle de
    registrar_log(lectura) tras leer del disco (8 repeticiones en la
    demo). El -0.03 por mensaje repetido rompe el bucle sin impedir
    emitir mensajes distintos.
    """
    entorno = _entorno_guardar_neutro(66)
    _prefijo_guardar(entorno)
    _, r_traza, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)
    assert r_traza == pytest.approx(-0.01 + R_TRAZO_VALIDO, abs=1e-4)
    # repetir el MISMO mensaje: penalización de ruido
    _, r_repetido, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)
    assert r_repetido == pytest.approx(-0.01 + R_TRAZO_RUIDO, abs=1e-4)
    # un mensaje DISTINTO no penaliza (paga solo su ciclo)
    _, r_distinto, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, 3)
    assert r_distinto == pytest.approx(-0.01, abs=1e-4)


def test_tareas_sin_almacenamiento_no_premian_el_trazado():
    """El trazado vive en GUARDAR/RECORDAR: en ECO no hay nada que
    declarar y el premio no existe (evita el señuelo de escribir en
    RAM por escribir para luego trazarlo).

    Regresión de la primera incubación de la Fase 0.5: sin este filtro
    la política cultivaba trazas incidentales en ECO/SUMA y SUMA se
    estancó al 45 % de éxito.
    """
    entorno = EntornoBrooder(tareas_activas=[Tarea.ECO], semilla=13)
    entorno.reiniciar()
    entorno._conteos_novedad[Primitiva.REGISTRAR_LOG] = 10**9
    # alimenta el bus y escribe en RAM: nada que declarar en ECO
    entorno.paso(Primitiva.LEER_TECLADO, 0)
    entorno.paso(Primitiva.ESCRIBIR_MEMORIA, ARG_BUS)
    _, r, _, _ = entorno.paso(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)
    assert r == pytest.approx(-0.01, abs=1e-4)  # solo el coste del ciclo
    assert entorno.tasa_trazado() == {}         # ni siquiera cuenta la oportunidad


# ------------------------------------------------------------------
# métricas
# ------------------------------------------------------------------
def test_tasa_trazado_cuenta_oportunidades():
    """Cada escritura/lectura de almacenamiento es una oportunidad."""
    entorno = EntornoBrooder(
        tareas_activas=[Tarea.GUARDAR], semilla=8
    )
    for _ in range(10):
        entorno.reiniciar()
        oraculo = Oraculo(entorno.solicitud)
        terminada = False
        while not terminada:
            prim, arg = oraculo.accion()
            oraculo.avanzar()
            _, _, terminada, _ = entorno.paso(prim, arg)
    tasa, oportunidades = entorno.tasa_trazado()["GUARDAR"]
    assert oportunidades == 20  # 10 episodios x (1 escritura + 1 lectura)
    assert tasa == 1.0


# ------------------------------------------------------------------
# el oráculo: la política ideal traza el 100 %
# ------------------------------------------------------------------
def test_oraculo_traza_el_100_por_ciento():
    """La política de referencia declara toda su I/O de almacenamiento."""
    entorno = EntornoBrooder(
        tareas_activas=[Tarea.GUARDAR, Tarea.RECORDAR], semilla=11
    )
    fallos = []
    for _ in range(100):
        entorno.reiniciar()
        oraculo = Oraculo(entorno.solicitud)
        terminada = False
        while not terminada:
            prim, arg = oraculo.accion()
            oraculo.avanzar()
            _, _, terminada, info = entorno.paso(prim, arg)
        if not info["exito"]:
            fallos.append(info["causa"])
    assert not fallos, f"el oráculo falló: {fallos[:3]}"
    for tarea in ("GUARDAR", "RECORDAR"):
        tasa, oportunidades = entorno.tasa_trazado()[tarea]
        assert oportunidades > 0
        assert tasa == 1.0


# ------------------------------------------------------------------
# el núcleo entrega la evidencia: resultado.trazos
# ------------------------------------------------------------------
class _CerebroOraculo:
    """Cerebro de prueba que delega sus decisiones en el oráculo."""

    def __init__(self, solicitud):
        self._oraculo = Oraculo(solicitud)

    def eval(self):  # compatibilidad de interfaz
        return None

    def estado_inicial(self, lote=1, dispositivo=None):
        return self._oraculo, None

    def decidir(self, obs, h, M, determinista=True):
        prim, arg = h.accion()
        h.avanzar()
        return prim, arg, 0.0, h, M


def test_resultado_solicitud_lleva_los_trazos():
    """El veredicto anota QUÉ mensajes decidió emitir el cerebro."""
    solicitud = Solicitud.desde_texto("guardar 4 G")
    nucleo = NucleoBrooder(
        PCVirtual(), _CerebroOraculo(solicitud), registro_eventos=False
    )
    resultado = nucleo.atender_solicitud(solicitud)
    assert resultado.exito
    assert resultado.trazos == [MENSAJE_LOG_ESCRITURA, MENSAJE_LOG_LECTURA]
    # y el anillo del kernel contiene esas entradas de verdad
    registro = nucleo.maquina.instante().registro
    assert len(registro) == 2
    assert {m for _, m in registro} == {
        MENSAJE_LOG_ESCRITURA,
        MENSAJE_LOG_LECTURA,
    }


def test_solicitud_sin_almacenamiento_no_deja_trazos():
    """ECO no toca disco ni RAM: el oráculo no traza (nada que declarar)."""
    solicitud = Solicitud.desde_texto("HOLA")
    nucleo = NucleoBrooder(
        PCVirtual(), _CerebroOraculo(solicitud), registro_eventos=False
    )
    resultado = nucleo.atender_solicitud(solicitud)
    assert resultado.exito
    assert resultado.trazos == []


def test_evaluar_informa_del_trazado():
    """La evaluación determinista expone la tasa de trazado (plumbing)."""
    from brooder.cerebro import CerebroBrooder
    from brooder.incubadora import evaluar

    torch.manual_seed(0)
    cerebro = CerebroBrooder(oculto=16, mem_ranuras=2, mem_dim=4)
    exito, trazado = evaluar(
        cerebro, [Tarea.GUARDAR], n_solicitudes=5, con_trazado=True
    )
    assert set(exito) == {"GUARDAR"}
    assert 0.0 <= exito["GUARDAR"] <= 1.0
    # sin entrenar no traza, pero la métrica existe (0 oportunidades
    # cubiertas por una política que no toca el almacenamiento)
    assert all(0.0 <= v <= 1.0 for v in trazado.values())


# ------------------------------------------------------------------
# la incubadora no converge sin trazado (regresión de la 1ª incubación)
# ------------------------------------------------------------------
def test_el_alfabeto_de_entrenamiento_es_completo():
    """Regresión de la Fase 0.5: entrenar solo con A..J dejó al
    cerebro con trazado incapacitado ante valores como 'Z' (el
    vocabulario real del teclado es A..Z)."""
    from brooder.constantes import LETRAS_ENTRENAMIENTO, TOKEN_A, TOKEN_Z

    assert LETRAS_ENTRENAMIENTO == list(range(TOKEN_A, TOKEN_Z + 1))


def test_eco_entrena_con_ecos_largos():
    """Regresión de la Fase 0.5: con ecos de 1..5 letras el cerebro
    con trazado fallaba el eco de 7 ('BROODER' en la demo)."""
    import random as _random

    rng = _random.Random(1)
    largos = set()
    for _ in range(600):
        s = Solicitud.aleatoria(Tarea.ECO, rng)
        largos.add(len(s.tokens))
    assert max(largos) >= 8


def test_ventana_de_trazado_del_entorno():
    """La ventana móvil mide el trazado de las últimas solicitudes."""
    entorno = EntornoBrooder(tareas_activas=[Tarea.GUARDAR], semilla=21)
    for _ in range(5):
        entorno.reiniciar()
        oraculo = Oraculo(entorno.solicitud)
        terminada = False
        while not terminada:
            prim, arg = oraculo.accion()
            oraculo.avanzar()
            _, _, terminada, _ = entorno.paso(prim, arg)
    assert entorno.tasa_trazado_ventana() == 1.0
    # y sin episodios de almacenamiento, la ventana es 0 (sin ops)
    entorno.limpiar_metricas()
    assert entorno.tasa_trazado_ventana() == 0.0


def test_la_convergencia_exige_el_trazado(tmp_path):
    """Sin trazado integrado, la incubadora no se da por convergida.

    Regresión de la primera incubación de la Fase 0.5: 100 % de éxito
    con 0 % de trazado — y un cerebro entregado que nunca usaba su
    syscall.
    """
    from brooder.incubadora import ConfiguracionPPO, Incubadora

    cfg = ConfiguracionPPO(pasos_totales=10)
    inc = Incubadora(cfg=cfg, dir_salida=tmp_path / "inc", silencioso=True)

    # etapa 1 (solo ECO): nada que integrar
    inc.etapa = 0
    assert inc._trazado_integrado({})
    assert inc._trazado_integrado({"GUARDAR": 0.0})

    # etapa con almacenamiento: exige GUARDAR y RECORDAR >= umbral
    inc.etapa = 2
    assert not inc._trazado_integrado({"GUARDAR": 1.0, "RECORDAR": 0.0})
    assert not inc._trazado_integrado({"GUARDAR": 0.4, "RECORDAR": 0.9})
    assert not inc._trazado_integrado({})  # sin datos: no converge
    assert inc._trazado_integrado(
        {"GUARDAR": cfg.umbral_trazado, "RECORDAR": 1.0}
    )
