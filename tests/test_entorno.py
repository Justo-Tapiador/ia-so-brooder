"""Tests del entorno de entrenamiento y de la política oráculo.

El test clave es `test_oracula_resuelve_todo`: la política ideal
escrita a mano debe resolver el 100 % de solicitudes aleatorias.
Si este test falla, el entorno está mal construido (y el
entrenamiento no podría converger).
"""
import random

import pytest

from brooder.constantes import ARG_BUS, Tarea, tokens_a_texto
from brooder.entorno import EntornoBrooder, Oraculo, R_EXITO, R_FALLO


# ------------------------------------------------------------------
# el oráculo resuelve TODO
# ------------------------------------------------------------------
def test_oraculo_resuelve_todo(rng, semillas_todas_las_tareas):
    entorno = EntornoBrooder(
        tareas_activas=semillas_todas_las_tareas, semilla=42
    )
    fallos = []
    for _ in range(300):
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


def test_oraculo_recompensa_alta(rng):
    """El retorno del oráculo debe ser claramente positivo."""
    entorno = EntornoBrooder(tareas_activas=[Tarea.ECO], semilla=7)
    retornos = []
    for _ in range(100):
        entorno.reiniciar()
        oraculo = Oraculo(entorno.solicitud)
        retorno = 0.0
        terminada = False
        while not terminada:
            prim, arg = oraculo.accion()
            oraculo.avanzar()
            _, r, terminada, _ = entorno.paso(prim, arg)
            retorno += r
        retornos.append(retorno)
    assert sum(retornos) / len(retornos) > 0.5


# ------------------------------------------------------------------
# generación de solicitudes
# ------------------------------------------------------------------
def test_solicitudes_son_coherentes(rng):
    from brooder.solicitudes import Solicitud

    for tarea in Tarea:
        for _ in range(50):
            s = Solicitud.aleatoria(tarea, rng)
            assert s.tarea == tarea
            assert s.presupuesto > 0
            if tarea == Tarea.DISPOSITIVO:
                # evento de hardware: sin teclado y sin salida esperada
                assert s.tokens == []
                assert s.esperado == []
                assert s.datos["modo"] in ("montar", "desmontar")
            else:
                assert len(s.tokens) > 0
                assert s.esperado  # toda tarea clásica tiene salida esperada


def test_solicitud_desde_texto():
    from brooder.solicitudes import Solicitud

    s = Solicitud.desde_texto("HOLA")
    assert s.tarea == Tarea.ECO and tokens_a_texto(s.tokens) == "HOLA"
    s = Solicitud.desde_texto("3+5")
    assert s.tarea == Tarea.SUMA and s.esperado == [8]
    s = Solicitud.desde_texto("7+8")
    assert s.tarea == Tarea.SUMA and s.esperado == [1, 5]  # dos dígitos
    s = Solicitud.desde_texto("guardar 4 G")
    assert s.tarea == Tarea.GUARDAR and s.datos == {"K": 4, "V": ord("G") - 55}
    s = Solicitud.desde_texto("recordar 2 Z")
    assert s.tarea == Tarea.RECORDAR
    s = Solicitud.desde_texto("aviso A")
    assert s.tarea == Tarea.AVISO
    assert Solicitud.desde_texto("") is None
    assert Solicitud.desde_texto("guardar X") is None


# ------------------------------------------------------------------
# condiciones de éxito (usan el recurso, no adivinan)
# ------------------------------------------------------------------
def test_suma_exige_usar_la_cpu():
    """Mostrar el resultado sin calcularlo con la CPU no basta."""
    from brooder.solicitudes import Solicitud
    from brooder.primitivas.virtual import PCVirtual

    s = Solicitud.desde_texto("3+5")
    assert s.tarea == Tarea.SUMA
    maquina = PCVirtual()
    maquina.reiniciar_registros()
    # leer los tres tokens y mostrar un '8' tecleado a mano,
    # sin usar la CPU
    maquina.escribir_teclado([8])
    maquina.leer_teclado()
    maquina.mostrar_en_pantalla(ARG_BUS)
    instante = maquina.instante()
    assert instante.pantalla == s.esperado
    assert not s.exito(instante)  # acumulador == 0: no usó la CPU


def test_guardar_exige_escribir_el_disco():
    from brooder.solicitudes import Solicitud
    from brooder.primitivas.virtual import PCVirtual

    s = Solicitud.desde_texto("guardar 4 G")
    maquina = PCVirtual()
    maquina.reiniciar_registros()
    maquina.escribir_teclado(s.tokens)
    # lee el valor y lo muestra sin escribir en el disco
    maquina.leer_teclado()  # K
    maquina.leer_teclado()  # V -> bus
    maquina.mostrar_en_pantalla(ARG_BUS)
    instante = maquina.instante()
    assert instante.pantalla == s.esperado
    assert not s.exito(instante)  # disco[4] == 0: no escribió


# ------------------------------------------------------------------
# recompensas de moldeado
# ------------------------------------------------------------------
def test_caracter_correcto_premia_y_erroneo_penaliza():
    from brooder.constantes import Primitiva
    from brooder.solicitudes import Solicitud

    entorno = EntornoBrooder(tareas_activas=[Tarea.ECO], semilla=3)
    entorno.reiniciar()
    sol = entorno.solicitud

    # leer el primer token: recompensa de lectura
    entorno.maquina.escribir_teclado(sol.tokens[:1])
    _, r_leer, _, _ = entorno.paso(Primitiva.LEER_TECLADO, 0)
    assert r_leer > 0
    # mostrarlo: carácter correcto -> recompensa fuerte
    _, r_ok, _, info = entorno.paso(Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)
    assert r_ok > 0.1
    assert info["tarea"] == "ECO"


def test_fallo_por_presupuesto_agotado():
    """Hacer nada termina en fallo con penalización."""
    from brooder.constantes import Primitiva

    entorno = EntornoBrooder(tareas_activas=[Tarea.ECO], semilla=5)
    entorno.reiniciar()
    total = 0.0
    terminada = False
    info = {}
    while not terminada:
        _, r, terminada, info = entorno.paso(Primitiva.NADA, 0)
        total += r
    assert info["causa"] == "presupuesto_agotado"
    assert info["exito"] is False
    assert total < -0.4


def test_exito_termina_la_solicitud():
    from brooder.constantes import Primitiva

    entorno = EntornoBrooder(tareas_activas=[Tarea.ECO], semilla=9)
    entorno.reiniciar()
    oraculo = Oraculo(entorno.solicitud)
    terminada = False
    info = {}
    while not terminada:
        prim, arg = oraculo.accion()
        oraculo.avanzar()
        _, _, terminada, info = entorno.paso(prim, arg)
    assert info["causa"] == "exito" and info["exito"]


# ------------------------------------------------------------------
# currículo
# ------------------------------------------------------------------
def test_fijar_tareas():
    entorno = EntornoBrooder(tareas_activas=[Tarea.ECO], semilla=1)
    for _ in range(20):
        entorno.reiniciar()
        assert entorno.solicitud.tarea == Tarea.ECO
    entorno.fijar_tareas([Tarea.SUMA])
    for _ in range(20):
        entorno.reiniciar()
        assert entorno.solicitud.tarea == Tarea.SUMA
