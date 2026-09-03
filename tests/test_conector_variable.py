"""Tests del fix OOD (v0.4.0): variabilidad del conector.

EL AGUJERO: hasta v0.3.0 las solicitudes clásicas solo existían con
el conector USB vacío — el hot-plug era exclusivo de DISPOSITIVO—.
El cerebro incubado jamás vio los canales [22]-[25] activos con una
clásica: fuera de distribución (OOD). Con un pendrive montado (con
datos residuales de una "sesión anterior") el cerebro v0.3.0 caía
del ~99 % de acierto al ~45 %.

EL FIX: el conector de las clásicas nace ahora en uno de tres
estados legales (vacío / conectado sin montar / montado con restos).
El pendrive sigue siendo NEUTRO en recompensa para las clásicas
(anti-señuelo) y las condiciones de éxito no lo tocan — la cirugía
es limpia y el oráculo sigue resolviendo el 100 % (el test clave de
este archivo). La incubadora, además, solo converge si las clásicas
se resuelven en los TRES estados (gate de invarianza).
"""
import argparse

import pytest

from brooder.constantes import N_RANURAS_DISPOSITIVO, Tarea
from brooder.entorno import (
    ESTADO_CONECTOR_ALEATORIO,
    ESTADOS_CONECTOR,
    EntornoBrooder,
    Oraculo,
    TAREAS_DISPOSITIVO,
)

CLASICAS = [t for t in Tarea if t not in TAREAS_DISPOSITIVO]


# ------------------------------------------------------------------
# el régimen histórico queda intacto
# ------------------------------------------------------------------
def test_por_defecto_el_conector_de_las_clasicas_nace_vacio():
    """estado_conector=None: el régimen de v0.3.0, bit a bit.

    La evaluación y los tests históricos con semilla fija dependen de
    que None no cambie NI una tirada del rng: el bloque de
    variabilidad no se ejecuta.
    """
    entorno = EntornoBrooder(tareas_activas=CLASICAS, semilla=101)
    for _ in range(200):
        entorno.reiniciar()
        instante = entorno.maquina.instante()
        assert not instante.dispositivo_conectado
        assert not instante.dispositivo_montado


def test_estado_conector_ilegal_lanza():
    """Un estado desconocido se rechaza con ruido (honestidad API)."""
    with pytest.raises(ValueError):
        EntornoBrooder(estado_conector="enchufado")


# ------------------------------------------------------------------
# la mezcla 60/10/30 y el contenido residual
# ------------------------------------------------------------------
def test_aleatorio_baraja_los_tres_estados():
    """La mezcla de entrenamiento: 60 % vacío / 10 % conectado / 30 %
    montado, con margen binomial para 600 sorteos con semilla."""
    entorno = EntornoBrooder(
        tareas_activas=CLASICAS, semilla=7,
        estado_conector=ESTADO_CONECTOR_ALEATORIO,
    )
    conteos = {"vacio": 0, "conectado": 0, "montado": 0}
    n = 600
    for _ in range(n):
        entorno.reiniciar()
        instante = entorno.maquina.instante()
        if instante.dispositivo_montado:
            conteos["montado"] += 1
        elif instante.dispositivo_conectado:
            conteos["conectado"] += 1
        else:
            conteos["vacio"] += 1
    assert sum(conteos.values()) == n
    assert conteos["montado"] >= 0.22 * n
    assert conteos["montado"] <= 0.38 * n
    assert conteos["conectado"] >= 0.05 * n
    assert conteos["conectado"] <= 0.16 * n


def test_montado_trae_datos_residuales_y_cursor_no_rebobinado():
    """El pendrive "recuerda" una sesión anterior: 1-3 ranuras A-Z y
    el cursor donde lo dejaron (el canal [24] hay que LEERLO, no
    asumir que nace en 0)."""
    entorno = EntornoBrooder(
        tareas_activas=CLASICAS, semilla=5, estado_conector="montado",
    )
    punteros = set()
    for _ in range(60):
        entorno.reiniciar()
        instante = entorno.maquina.instante()
        assert instante.dispositivo_conectado
        assert instante.dispositivo_montado
        datos = [t for t in instante.dispositivo_ranuras if t]
        assert 1 <= len(datos) <= 3
        assert all(0 < t for t in datos)  # tokens válidos, no ceros
        punteros.add(instante.dispositivo_puntero)
    assert len(punteros) >= 3
    assert all(0 <= p < N_RANURAS_DISPOSITIVO for p in punteros)


def test_conectado_esta_presente_pero_sin_montar():
    """Hot-plug a medias: presente, sin montar y con el medio vacío."""
    entorno = EntornoBrooder(
        tareas_activas=CLASICAS, semilla=9, estado_conector="conectado",
    )
    for _ in range(40):
        entorno.reiniciar()
        instante = entorno.maquina.instante()
        assert instante.dispositivo_conectado
        assert not instante.dispositivo_montado
        assert not any(instante.dispositivo_ranuras)


def test_la_percepcion_refleja_el_estado_del_conector():
    """Los canales [22]/[23] existen para esto: la observación ve el
    estado real del conector desde el primer ciclo."""
    entorno = EntornoBrooder(
        tareas_activas=CLASICAS, semilla=2, estado_conector="montado",
    )
    obs = entorno.reiniciar()
    assert obs[22] == 1.0   # conectado
    assert obs[23] == 1.0   # montado
    assert 0.0 <= obs[24] <= 1.0  # cursor del medio
    assert obs[25] == 0.0   # aún no hay escrituras

    entorno_vacio = EntornoBrooder(
        tareas_activas=CLASICAS, semilla=2, estado_conector=None,
    )
    obs_vacio = entorno_vacio.reiniciar()
    assert obs_vacio[22] == 0.0
    assert obs_vacio[23] == 0.0


# ------------------------------------------------------------------
# los episodios DISPOSITIVO son inmunes (contrato, no azar)
# ------------------------------------------------------------------
def test_dispositivo_es_inmune_a_la_variabilidad():
    """El pendrive de un episodio DISPOSITIVO llega por contrato de
    la solicitud, no por el estado del conector: bit a bit v0.3.0
    para cualquier estado forzado (incluido el aleatorio)."""
    for estado in (*ESTADOS_CONECTOR, ESTADO_CONECTOR_ALEATORIO):
        entorno = EntornoBrooder(
            tareas_activas=[Tarea.DISPOSITIVO], semilla=3,
            estado_conector=estado,
        )
        modos_vistos = set()
        for _ in range(80):
            entorno.reiniciar()
            sol = entorno.solicitud
            instante = entorno.maquina.instante()
            modo = sol.datos["modo"]
            modos_vistos.add(modo)
            assert instante.dispositivo_conectado
            if modo == "montar":
                # el trabajo de la política ES montarlo
                assert not instante.dispositivo_montado
                assert not any(instante.dispositivo_ranuras)
            else:
                assert instante.dispositivo_montado
                if modo == "leer":
                    # el valor esperado vive SOLO en el medio
                    assert (
                        instante.dispositivo_ranuras[sol.datos["K"]]
                        == sol.datos["V"]
                    )
                else:
                    assert not any(instante.dispositivo_ranuras)
        assert modos_vistos == {"montar", "desmontar", "escribir", "leer"}


# ------------------------------------------------------------------
# el test clave: el entorno sigue siendo 100 % resoluble
# ------------------------------------------------------------------
def test_oraculo_resuelve_todo_con_variabilidad(semillas_todas_las_tareas):
    """La cirugía es limpia: la política ideal ignora el pendrive de
    las clásicas y resuelve igual con el conector en cualquier estado.

    Si este test fallara, la variabilidad estaría contaminando la
    semántica de las tareas (recompensa o éxito dependientes del
    dispositivo) y el entorno estaría mal construido.
    """
    entorno = EntornoBrooder(
        tareas_activas=semillas_todas_las_tareas, semilla=42,
        estado_conector=ESTADO_CONECTOR_ALEATORIO,
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


# ------------------------------------------------------------------
# la evaluación y el gate de invarianza de la incubadora
# ------------------------------------------------------------------
def test_evaluar_respeta_el_estado_forzado():
    """evaluar(estado_conector=...) mide cada distribución por
    separado: el mismo cerebro, tres mundos distintos."""
    import torch

    from brooder.cerebro import CerebroBrooder
    from brooder.incubadora import evaluar

    torch.manual_seed(0)
    cerebro = CerebroBrooder(oculto=16, mem_ranuras=2, mem_dim=4)
    for estado in ESTADOS_CONECTOR:
        exito = evaluar(
            cerebro, [Tarea.ECO], n_solicitudes=5, estado_conector=estado
        )
        assert set(exito) == {"ECO"}
        assert 0.0 <= exito["ECO"] <= 1.0


def test_la_incubadora_nace_con_variabilidad(tmp_path):
    """v0.4.0: la incubación POR DEFECTO entrena con el conector
    variable; conector_variable=False recupera el régimen v0.3.0."""
    from brooder.incubadora import ConfiguracionPPO, Incubadora

    inc = Incubadora(
        cfg=ConfiguracionPPO(pasos_totales=10),
        dir_salida=tmp_path / "inc", silencioso=True,
    )
    assert inc.entorno.estado_conector == ESTADO_CONECTOR_ALEATORIO

    inc_fijo = Incubadora(
        cfg=ConfiguracionPPO(pasos_totales=10, conector_variable=False),
        dir_salida=tmp_path / "fijo", silencioso=True,
    )
    assert inc_fijo.entorno.estado_conector is None


def test_el_gate_de_invarianza_funde_el_minimo(tmp_path, monkeypatch):
    """_evaluar_etapa fusiona los estados con el PEOR resultado.

    Regresión del agujero OOD: una política que resuelve ECO al
    100 % con el conector vacío y al 40 % con el pendrive montado
    queda en el 40 % — y no puede converger. Con la variabilidad
    apagada, la evaluación es la histórica (un solo mundo).
    """
    from brooder.incubadora import ConfiguracionPPO, Incubadora

    def _evaluar_falso(
        cerebro, tareas, n_solicitudes=2, semilla=99_999,
        con_trazado=False, estado_conector=None, **_,
    ):
        if estado_conector == "montado":
            return {"ECO": 0.4}, {}
        return {"ECO": 1.0}, {}

    monkeypatch.setattr("brooder.incubadora.evaluar", _evaluar_falso)

    cfg = ConfiguracionPPO(pasos_totales=10, conector_variable=True)
    inc = Incubadora(cfg=cfg, dir_salida=tmp_path / "inc", silencioso=True)
    inc.etapa = 0
    exito, _ = inc._evaluar_etapa()
    assert exito == {"ECO": 0.4}
    # el detalle por estado queda expuesto para el registro
    assert inc.detalle_eval_estado["vacio"] == {"ECO": 1.0}
    assert inc.detalle_eval_estado["montado"] == {"ECO": 0.4}

    cfg_fijo = ConfiguracionPPO(pasos_totales=10, conector_variable=False)
    inc_fijo = Incubadora(
        cfg=cfg_fijo, dir_salida=tmp_path / "fijo", silencioso=True
    )
    inc_fijo.etapa = 0
    exito_fijo, _ = inc_fijo._evaluar_etapa()
    assert exito_fijo == {"ECO": 1.0}
    assert inc_fijo.detalle_eval_estado == {}


# ------------------------------------------------------------------
# el diagnóstico muestra el antes/después
# ------------------------------------------------------------------
def test_diagnostico_muestra_la_invarianza_del_conector(capsys):
    """El bloque de invarianza del 'brooder diagnostico': aquí se VE
    el agujero (un cerebro sin entrenar falla en todos los estados)."""
    from brooder.cli import cmd_diagnostico

    args = argparse.Namespace(ssd=None, solicitudes=2)
    codigo = cmd_diagnostico(args)
    salida = capsys.readouterr().out
    assert "Invarianza del conector" in salida
    assert "vacio" in salida
    assert "conectado" in salida
    assert "montado" in salida
    # cerebro sin entrenar: dominio parcial
    assert codigo == 1
    assert "dominio parcial" in salida
