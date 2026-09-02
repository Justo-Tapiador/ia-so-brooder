"""Fase 1.5: almacenamiento real en el pendrive montado — tests.

Cubre: el plan de datos del dispositivo (mover/leer/escribir con
montaje obligatorio), el trazado I/O PROPIO del pendrive, la
persistencia real (los datos viven en el medio: sobreviven a
desmontar y a retirar-reenchufar con extracción segura; se PIERDEN
con la extracción insegura), el pendrive con contenido de fábrica,
los modos escribir/leer de la tarea DISPOSITIVO (éxito verificable
por hardware, a prueba de "adivinanzas"), el oráculo como referencia
del 100 %, la persistencia REAL de PCReal (pendrive.json entre
instancias) y el guardián de contrato generalizado (un cerebro Fase 1
24x20 sobre el kernel 26x23 recibe su AVISO con las faltas exactas).
"""
import torch

from brooder.cerebro import CerebroBrooder
from brooder.constantes import (
    ARG_BUS,
    MENSAJE_LOG_ESCRITURA,
    MENSAJE_LOG_LECTURA,
    N_RANURAS_DISPOSITIVO,
    N_PRIMITIVAS,
    OBS_DIM,
    Primitiva,
    Tarea,
    TOKEN_DE_CARACTER,
    formatear_trazado_dispositivo,
)
from brooder.entorno import (
    EntornoBrooder,
    Oraculo,
    R_LECTURA_ALMACEN_OK,
    R_ESCRITURA_OK,
)
from brooder.nucleo import NucleoBrooder, aviso_contrato
from brooder.primitivas.virtual import PCVirtual
from brooder.solicitudes import Solicitud

Q = TOKEN_DE_CARACTER["Q"]


def _maquina_montada() -> PCVirtual:
    m = PCVirtual()
    m.reiniciar_registros()
    m.conectar_dispositivo()
    assert m.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    return m


def _escribir_ranura(m: PCVirtual, ranura: int, valor: int) -> None:
    """Escribe un valor en la ranura del pendrive montado (vía bus)."""
    m.escribir_teclado([valor])
    m.leer_teclado()
    assert m.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, ranura)
    assert m.ejecutar(Primitiva.ESCRIBIR_DISPOSITIVO, ARG_BUS)


# ------------------------------------------------------------------
# el plan de datos exige montaje (como un USB real)
# ------------------------------------------------------------------
def test_las_tres_primitivas_exigen_dispositivo_montado():
    m = PCVirtual()
    m.conectar_dispositivo()  # presente pero SIN montar
    for primitiva, argumento in (
        (Primitiva.MOVER_PUNTERO_DISPOSITIVO, 3),
        (Primitiva.LEER_DISPOSITIVO, 0),
        (Primitiva.ESCRIBIR_DISPOSITIVO, ARG_BUS),
    ):
        assert not m.ejecutar(primitiva, argumento)
        assert "no montado" in m.instante().ultimo_error
    # sin dispositivo conectado, igual (y con causa visible)
    m.desconectar_dispositivo()
    assert not m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    assert "no montado" in m.instante().ultimo_error


def test_el_puntero_valida_el_rango_del_dispositivo():
    m = _maquina_montada()
    for d in range(N_RANURAS_DISPOSITIVO):
        assert m.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, d)
        assert m.instante().dispositivo_puntero == d
    # el pendrive solo tiene 8 ranuras (0..7): 8 y 9 son del disco
    assert not m.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, 8)
    assert "inválida" in m.instante().ultimo_error


def test_escribir_y_leer_por_el_bus():
    m = _maquina_montada()
    _escribir_ranura(m, 3, Q)
    instante = m.instante()
    assert instante.dispositivo_ranuras[3] == Q
    assert instante.escrituras_dispositivo == 1
    # la lectura deposita el token del medio en el bus
    assert m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    instante = m.instante()
    assert instante.bus_valido and instante.bus_valor == Q


# ------------------------------------------------------------------
# trazado I/O propio del dispositivo
# ------------------------------------------------------------------
def test_cada_operacion_deja_su_entrada_en_el_trazado():
    m = _maquina_montada()
    _escribir_ranura(m, 3, Q)
    m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    trazado = m.instante().dispositivo_trazado
    assert len(trazado) == 2
    # (paso, tipo, ranura, valor): escritura y luego lectura
    assert trazado[0][1] == "escritura" and trazado[0][2] == 3
    assert trazado[0][3] == Q
    assert trazado[1][1] == "lectura" and trazado[1][3] == Q
    # el panel formatea las entradas con su ranura y su carácter
    lineas = [l for l in m.panel_trazado_dispositivo() if l]
    assert any("ranura[3]" in l and "'Q'" in l for l in lineas)


def test_el_registro_del_sistema_no_se_contamina_con_la_io():
    """El trazado I/O es del DISPOSITIVO; el registro (dmesg) solo
    anota el ciclo de vida: montar/desmontar/extracción."""
    m = _maquina_montada()
    _escribir_ranura(m, 3, Q)
    m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    mensajes = [msg for _, msg in m.instante().registro]
    # solo el "dispositivo montado" del setup: ni lectura ni escritura
    assert 8 in mensajes  # MENSAJE_LOG_DISP_MONTADO
    assert MENSAJE_LOG_ESCRITURA not in mensajes
    assert MENSAJE_LOG_LECTURA not in mensajes


def test_el_trazado_formato_es_legible():
    linea = formatear_trazado_dispositivo((7, "escritura", 3, Q))
    assert linea == "[0007] E ranura[3] <- 'Q'"
    linea = formatear_trazado_dispositivo((9, "lectura", 3, Q))
    assert linea == "[0009] L ranura[3] -> 'Q'"


# ------------------------------------------------------------------
# persistencia real: los datos viven en el medio extraíble
# ------------------------------------------------------------------
def test_los_datos_sobreviven_a_desmontar_y_volver_a_montar():
    m = _maquina_montada()
    _escribir_ranura(m, 3, Q)
    assert m.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)
    assert m.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    m.reiniciar_registros()  # frontera de solicitud: puntero a cero
    assert m.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, 3)
    assert m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    assert m.instante().bus_valor == Q


def test_extraccion_segura_el_pendrive_recuerda():
    """Desmontar -> retirar -> reenchufar -> montar -> leer: el dato
    viaja CON EL MEDIO (la ranura vive en el pendrive, no en la
    máquina). Es la escena 'el pendrive recuerda' de la demo."""
    m = _maquina_montada()
    _escribir_ranura(m, 3, Q)
    assert m.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)
    assert m.desconectar_dispositivo()  # limpia: estaba desmontado
    m.conectar_dispositivo()
    assert m.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    m.reiniciar_registros()
    assert m.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, 3)
    assert m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    assert m.instante().bus_valor == Q


def test_extraccion_insegura_pierde_los_datos():
    """Retirar el pendrive MONTADO pierde el búfer sin sincronizar:
    el kernel registra el ERROR y las ranuras quedan a cero."""
    m = _maquina_montada()
    _escribir_ranura(m, 3, Q)
    assert not m.desconectar_dispositivo()  # insegura
    assert m.instante().dispositivo_ranuras[3] == 0
    assert not m.instante().dispositivo_trazado  # el anillo también se pierde
    m.conectar_dispositivo()
    assert m.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    assert m.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, 3)
    assert m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    assert m.instante().bus_valor == 0  # el dato se perdió DE VERDAD


def test_pendrive_con_contenido_de_fabrica():
    """El mundo puede enchufar un pendrive que ya trae datos."""
    m = PCVirtual()
    m.conectar_dispositivo(contenido={3: Q})
    assert m.instante().dispositivo_ranuras[3] == Q
    assert m.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    m.reiniciar_registros()
    assert m.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, 3)
    assert m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    assert m.instante().bus_valor == Q


# ------------------------------------------------------------------
# la tarea DISPOSITIVO: modos escribir y leer
# ------------------------------------------------------------------
def test_exito_de_escribir_exige_escritura_real():
    """Pantalla correcta pero ranura sin escribir: NO hay éxito. La
    política no puede 'adivinar' mostrando lo que tecleó el usuario."""
    m = _maquina_montada()
    s = Solicitud(
        Tarea.DISPOSITIVO, tokens=[3, Q, 3], esperado=[Q],
        datos={"modo": "escribir", "K": 3, "V": Q},
    )
    # pantalla bien (eco del valor), ranura vacía: falla
    m.escribir_teclado([Q])
    m.leer_teclado()
    m.ejecutar(Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)
    assert m.instante().pantalla == [Q]
    assert not s.exito(m.instante())
    # la escritura real completa el éxito
    m.reiniciar_registros()
    _escribir_ranura(m, 3, Q)
    m.reiniciar_registros()
    m.escribir_teclado([Q])
    m.leer_teclado()
    m.ejecutar(Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)
    assert s.exito(m.instante())


def test_exito_de_leer_exige_pantalla_correcta_y_montaje():
    m = PCVirtual()
    m.reiniciar_registros()
    # el pendrive llega con Q grabado de fábrica en la ranura 3
    m.conectar_dispositivo(contenido={3: Q})
    assert m.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    s = Solicitud(
        Tarea.DISPOSITIVO, tokens=[3], esperado=[Q],
        datos={"modo": "leer", "K": 3, "V": Q},
    )
    assert not s.exito(m.instante())  # pantalla vacía
    assert m.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, 3)
    assert m.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    assert m.ejecutar(Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)
    assert s.exito(m.instante())


def test_desde_texto_escribir_y_leer():
    s = Solicitud.desde_texto("escribir 3 P")
    assert s.tarea == Tarea.DISPOSITIVO
    assert s.datos == {"modo": "escribir", "K": 3, "V": TOKEN_DE_CARACTER["P"]}
    assert s.tokens == [3, TOKEN_DE_CARACTER["P"], 3]
    assert s.esperado == [TOKEN_DE_CARACTER["P"]]

    s = Solicitud.desde_texto("LEER 5 Z")
    assert s.datos == {"modo": "leer", "K": 5, "V": TOKEN_DE_CARACTER["Z"]}
    assert s.tokens == [5]  # el valor NO se teclea: vive en el medio
    assert s.esperado == [TOKEN_DE_CARACTER["Z"]]

    # el pendrive solo tiene ranuras 0..7: 8/9 no son direcciones válidas
    assert Solicitud.desde_texto("escribir 8 P") is None
    assert Solicitud.desde_texto("leer 9 P") is None


def test_desde_texto_pares_pegados():
    """Hotfix de campo: el dígito pegado a la letra es la MISMA
    solicitud que con espacio — 'leer 3P' = 'leer 3 P', como '3+5'
    ya lo era para la suma. Lo descubrió el dedo de un usuario real
    en sesión interactiva (dos intentos seguidos); el parser no
    debía castigar la forma compacta que la aritmética enseñó."""
    for pegada, canonica in (
        ("escribir 3P", "escribir 3 P"),
        ("leer 3P", "leer 3 P"),
        ("guardar 4G", "guardar 4 G"),
        ("recordar 2Z", "recordar 2 Z"),
    ):
        s = Solicitud.desde_texto(pegada)
        canon = Solicitud.desde_texto(canonica)
        assert s is not None, f"no parseó: {pegada!r}"
        assert s.tarea == canon.tarea
        assert s.datos == canon.datos
        assert s.tokens == canon.tokens
        assert s.esperado == canon.esperado


def test_desde_texto_pares_pegados_limites():
    # los límites siguen vigentes en la forma pegada: 8/9 no caben
    # en las 8 ranuras del pendrive (0..7)
    assert Solicitud.desde_texto("escribir 8P") is None
    assert Solicitud.desde_texto("leer 9P") is None
    # y lo que nunca fue un par, sigue sin serlo
    assert Solicitud.desde_texto("guardar X") is None
    assert Solicitud.desde_texto("leer P3") is None
    # el verbo pegado al argumento NO activa el par ni el eco: sin
    # espacio tras 'LEER', la línea no parsea y el intérprete muestra
    # el remedio (la línea de formatos) — mejor eso que encomendarle
    # al cerebro un eco con dígitos que nunca entrenó
    assert Solicitud.desde_texto("leer3P") is None


# ------------------------------------------------------------------
# el entorno: hot-plug con datos, recompensas y trazado
# ------------------------------------------------------------------
def _episodio_de_dispositivo(modo: str):
    entorno = EntornoBrooder(tareas_activas=[Tarea.DISPOSITIVO], semilla=13)
    for _ in range(400):
        entorno.reiniciar()
        if entorno.solicitud.datos.get("modo") == modo:
            return entorno
    raise AssertionError(f"no salió una solicitud de dispositivo en modo {modo}")


def test_el_entorno_prende_un_pendrive_con_datos_en_modo_leer():
    entorno = _episodio_de_dispositivo("leer")
    sol = entorno.solicitud
    instante = entorno.maquina.instante()
    assert instante.dispositivo_conectado and instante.dispositivo_montado
    # el valor esperado ya está grabado en el medio (contenido de fábrica)
    assert instante.dispositivo_ranuras[sol.datos["K"]] == sol.datos["V"]
    # y NO viaja por el teclado: la única fuente es el propio pendrive
    assert sol.datos["V"] not in sol.tokens


def test_el_entorno_sirve_el_pendrive_ya_montado_en_los_modos_de_datos():
    """Como en modo desmontar (lo montó la sesión anterior), los modos
    de almacenamiento reciben el medio YA montado: el plano de datos
    de la Fase 1.5 opera sobre el pendrive montado."""
    for modo in ("escribir", "leer", "desmontar"):
        entorno = _episodio_de_dispositivo(modo)
        instante = entorno.maquina.instante()
        assert instante.dispositivo_conectado
        assert instante.dispositivo_montado
        obs = entorno.observar()
        assert obs[21] == 1.0 and obs[22] == 1.0 and obs[23] == 1.0
    # el modo montar es el único que parte sin montar: decidirlo es
    # exactamente el trabajo de la política
    entorno = _episodio_de_dispositivo("montar")
    assert not entorno.maquina.instante().dispositivo_montado


def test_recompensa_de_escritura_y_lectura_del_dispositivo():
    # modo escribir (medio ya montado): la escritura en la ranura
    # pedida premia (el +1.0 del éxito llega con la recuperación)
    entorno = _episodio_de_dispositivo("escribir")
    sol = entorno.solicitud
    K, V = sol.datos["K"], sol.datos["V"]
    entorno.paso(Primitiva.LEER_TECLADO, 0)
    entorno.paso(Primitiva.MOVER_PUNTERO_DISPOSITIVO, ARG_BUS)
    entorno.paso(Primitiva.LEER_TECLADO, 0)
    _, recompensa, terminada, _ = entorno.paso(
        Primitiva.ESCRIBIR_DISPOSITIVO, ARG_BUS
    )
    assert recompensa >= R_ESCRITURA_OK
    assert not terminada  # falta recuperar el valor y mostrarlo

    # modo leer (medio ya montado, dato de fábrica): la lectura premia
    entorno = _episodio_de_dispositivo("leer")
    sol = entorno.solicitud
    K, V = sol.datos["K"], sol.datos["V"]
    entorno.paso(Primitiva.LEER_TECLADO, 0)
    entorno.paso(Primitiva.MOVER_PUNTERO_DISPOSITIVO, ARG_BUS)
    _, recompensa, _, _ = entorno.paso(Primitiva.LEER_DISPOSITIVO, 0)
    assert recompensa >= R_LECTURA_ALMACEN_OK


def test_el_oraculo_resuelve_escribir_y_leer():
    for modo in ("escribir", "leer"):
        entorno = _episodio_de_dispositivo(modo)
        oraculo = Oraculo(entorno.solicitud)
        terminada = False
        while not terminada:
            prim, arg = oraculo.accion()
            oraculo.avanzar()
            _, _, terminada, info = entorno.paso(prim, arg)
        assert info["exito"], f"el oráculo falló en modo {modo}"


def test_el_oraculo_traza_la_io_del_pendrive():
    """Referencia del 100 % de trazado: cada escritura/lectura del
    medio queda declarada con REGISTRAR_LOG (mensajes 2/1)."""
    entorno = _episodio_de_dispositivo("escribir")
    oraculo = Oraculo(entorno.solicitud)
    terminada = False
    while not terminada:
        prim, arg = oraculo.accion()
        oraculo.avanzar()
        _, _, terminada, _ = entorno.paso(prim, arg)
    tasa = entorno.tasa_trazado()
    assert tasa.get("DISPOSITIVO", (0.0, 0))[0] == 1.0
    assert tasa["DISPOSITIVO"][1] >= 2  # escribió Y leyó (trazable)


# ------------------------------------------------------------------
# el núcleo atiende la sesión completa (el pendrive recuerda)
# ------------------------------------------------------------------
def test_sesion_completa_escribir_retirar_volver_leer():
    """El ciclo de la demo, con la política perfecta: montar, escribir
    Q en la ranura 3, extracción segura, el mismo pendrive vuelve,
    la IA lo monta de nuevo y la lectura recupera Q. Entre
    solicitudes, la máquina conserva el medio (como hace
    arrancar/demo)."""
    from tests.test_trazado import _CerebroOraculo

    maquina = PCVirtual()
    maquina.conectar_dispositivo()

    montar = Solicitud(Tarea.DISPOSITIVO, datos={"modo": "montar"})
    nucleo = NucleoBrooder(
        maquina, _CerebroOraculo(montar), registro_eventos=False
    )
    r0 = nucleo.atender_solicitud(montar)
    assert r0.exito and maquina.dispositivo_montado

    escribir = Solicitud.desde_texto("escribir 3 Q")
    nucleo.cerebro = _CerebroOraculo(escribir)
    r1 = nucleo.atender_solicitud(escribir)
    assert r1.exito, r1.causa
    assert r1.pantalla == "Q"
    assert r1.trazos == [MENSAJE_LOG_ESCRITURA, MENSAJE_LOG_LECTURA]
    assert "pendrive[3]='Q'" in r1.detalle_dispositivos

    desmontar = Solicitud(Tarea.DISPOSITIVO, datos={"modo": "desmontar"})
    nucleo.cerebro = _CerebroOraculo(desmontar)
    r2 = nucleo.atender_solicitud(desmontar)
    assert r2.exito

    maquina.desconectar_dispositivo()  # extracción segura
    maquina.conectar_dispositivo()  # el mismo pendrive vuelve

    montar2 = Solicitud(Tarea.DISPOSITIVO, datos={"modo": "montar"})
    nucleo.cerebro = _CerebroOraculo(montar2)
    r3 = nucleo.atender_solicitud(montar2)
    assert r3.exito

    leer = Solicitud.desde_texto("leer 3 Q")
    nucleo.cerebro = _CerebroOraculo(leer)
    r4 = nucleo.atender_solicitud(leer)
    assert r4.exito, r4.causa
    assert r4.pantalla == "Q"  # EL PENDRIVE RECUERDA
    assert r4.trazos == [MENSAJE_LOG_LECTURA]


# ------------------------------------------------------------------
# PCReal: el almacenamiento es un archivo de verdad
# ------------------------------------------------------------------
def test_pcreal_el_pendrive_persiste_entre_instancias(tmp_path):
    from brooder.primitivas.reales import PCReal

    sandbox = tmp_path / "sb"
    m1 = PCReal(raiz_sandbox=sandbox)
    m1.conectar_dispositivo()
    assert m1.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    _escribir_ranura(m1, 3, Q)
    assert (sandbox / "pendrive.json").exists()

    # una INSTANCIA NUEVA (otro proceso, otro arranque) monta el mismo
    # pendrive: el dato sigue ahí — almacenamiento real en disco
    m2 = PCReal(raiz_sandbox=sandbox)
    assert m2.instante().dispositivo_ranuras[3] == Q
    m2.conectar_dispositivo()
    assert m2.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    m2.reiniciar_registros()
    assert m2.ejecutar(Primitiva.MOVER_PUNTERO_DISPOSITIVO, 3)
    assert m2.ejecutar(Primitiva.LEER_DISPOSITIVO, 0)
    assert m2.instante().bus_valor == Q


def test_pcreal_extraccion_insegura_borra_de_verdad(tmp_path):
    from brooder.primitivas.reales import PCReal

    sandbox = tmp_path / "sb"
    m1 = PCReal(raiz_sandbox=sandbox)
    m1.conectar_dispositivo()
    assert m1.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
    _escribir_ranura(m1, 3, Q)
    assert not m1.desconectar_dispositivo()  # retirado MONTADO: insegura

    m2 = PCReal(raiz_sandbox=sandbox)
    assert m2.instante().dispositivo_ranuras[3] == 0  # el borrado persistió


# ------------------------------------------------------------------
# el guardián de contrato generaliza (Fase 1 -> Fase 1.5)
# ------------------------------------------------------------------
def test_aviso_contrato_para_cerebro_fase1():
    """El cerebro de la Fase 1 (24x20) sobre este kernel (26x23)
    recibe el AVISO con las faltas exactas: las 3 primitivas del plan
    de datos y los 2 canales de percepción nuevos."""
    torch.manual_seed(11)
    cerebro_fase1 = CerebroBrooder(dim_entrada=24, n_primitivas=20)
    aviso = aviso_contrato(cerebro_fase1)
    assert aviso is not None
    for nombre in (
        "MOVER_PUNTERO_DISPOSITIVO",
        "LEER_DISPOSITIVO",
        "ESCRIBIR_DISPOSITIVO",
    ):
        assert nombre in aviso
    assert f"primitivas 20/{N_PRIMITIVAS}" in aviso
    assert f"observación 24/{OBS_DIM}" in aviso
    # y el remedio sigue siendo el mismo
    assert "contrato viejo" in aviso


def test_post_declara_el_contrato_fase15():
    torch.manual_seed(12)
    nucleo = NucleoBrooder(PCVirtual(), CerebroBrooder())
    linea = [c for c in nucleo.post() if c[0] == "Cerebro"]
    assert len(linea) == 1
    assert linea[0][1] == f"contrato {OBS_DIM}x{N_PRIMITIVAS}"
    assert linea[0][1] == "contrato 26x23"
