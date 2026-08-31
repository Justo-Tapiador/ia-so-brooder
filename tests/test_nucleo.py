"""Tests del núcleo de arranque (boot, SSD, recovery) y de la PC real."""
import json

import pytest

from brooder.cerebro import CerebroBrooder
from brooder.constantes import Tarea, tokens_a_texto
from brooder.nucleo import NucleoBrooder, exportar_ssd, montar_ssd
from brooder.primitivas.reales import PCReal
from brooder.primitivas.virtual import PCVirtual
from brooder.solicitudes import Solicitud


@pytest.fixture
def modelo(tmp_path):
    cerebro = CerebroBrooder()
    ruta = tmp_path / "mejor.pt"
    cerebro.guardar(ruta)
    return ruta


# ------------------------------------------------------------------
# SSD: exportar y montar
# ------------------------------------------------------------------
def test_ssd_ida_y_vuelta(modelo, tmp_path):
    salida = tmp_path / "ssd" / "brooder.img"
    exportar_ssd(modelo, salida, metricas={"exito_eval_final": {"ECO": 0.99}})
    assert salida.exists()

    cerebro, estado, manifiesto = montar_ssd(salida)
    assert isinstance(cerebro, CerebroBrooder)
    assert manifiesto["proyecto"] == "IA-SO Brooder"
    assert manifiesto["metricas"]["exito_eval_final"]["ECO"] == 0.99


def test_ssd_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError):
        montar_ssd(tmp_path / "no_existe.img")


# ------------------------------------------------------------------
# núcleo: POST y atención de solicitudes
# ------------------------------------------------------------------
def test_post_lista_dispositivos():
    nucleo = NucleoBrooder(PCVirtual(), CerebroBrooder())
    comprobaciones = nucleo.post()
    assert len(comprobaciones) >= 5
    nombres = " ".join(n for n, _ in comprobaciones)
    assert "CPU" in nombres and "Disco" in nombres


def test_atender_solicitud_con_cerebro_aleatorio_no_explota():
    """Con un cerebro sin entrenar, el núcleo debe seguir funcionando."""
    nucleo = NucleoBrooder(PCVirtual(), CerebroBrooder(), registro_eventos=False)
    resultado = nucleo.atender_solicitud(Solicitud.desde_texto("HOLA"))
    assert resultado.causa in ("exito", "presupuesto_agotado")
    assert resultado.ciclos == resultado.solicitud.presupuesto


def test_nucleo_con_oraculo_resuelve():
    """Sustituyendo el cerebro por el oráculo, el núcleo resuelve todo.

    Verifica que el ciclo percibir->decidir->actuar del núcleo es
    EXACTAMENTE el del entorno de entrenamiento.
    """
    from brooder.entorno import Oraculo

    class CerebroOraculo:
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

    for texto in ["HOLA", "3+5", "7+8", "guardar 4 G", "recordar 2 Z", "aviso A"]:
        solicitud = Solicitud.desde_texto(texto)
        nucleo = NucleoBrooder(
            PCVirtual(), CerebroOraculo(solicitud), registro_eventos=False
        )
        resultado = nucleo.atender_solicitud(solicitud)
        assert resultado.exito, (
            f"el núcleo no resolvió '{texto}': {resultado.causa} "
            f"pantalla='{resultado.pantalla}'"
        )


# ------------------------------------------------------------------
# estado persistente
# ------------------------------------------------------------------
def test_estado_se_acumula(modelo, tmp_path):
    nucleo = NucleoBrooder(PCVirtual(), CerebroBrooder())
    nucleo.estado.anotar_arranque()
    resultado = nucleo.atender_solicitud(Solicitud.desde_texto("HOLA"))
    nucleo.estado.anotar_arranque()
    assert nucleo.estado.arranques == 2
    assert nucleo.estado.solicitudes_atendidas == 1

    ruta = tmp_path / "estado.json"
    nucleo.estado.guardar(ruta)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert datos["solicitudes_atendidas"] == 1


# ------------------------------------------------------------------
# PC REAL (sandbox)
# ------------------------------------------------------------------
def test_pc_real_disco_persiste_entre_arranques(tmp_path):
    sandbox = tmp_path / "sandbox"
    pc1 = PCReal(raiz_sandbox=sandbox)
    pc1.reiniciar_registros()
    pc1.mover_cabezal_disco(4)
    pc1.escribir_teclado([15])  # 'F'
    pc1.leer_teclado()
    assert pc1.escribir_disco(38)  # ARG_BUS

    # apagar y volver a arrancar: el disco recuerda
    pc2 = PCReal(raiz_sandbox=sandbox)
    pc2.mover_cabezal_disco(4)
    assert pc2.leer_disco()
    assert pc2.instante().bus_valor == 15

    # y son archivos reales, confinados al sandbox
    archivo = sandbox / "disco" / "4.tok"
    assert archivo.exists()
    contenido = json.loads(archivo.read_text())
    assert contenido["token"] == 15


def test_pc_real_no_puede_escribir_fuera_del_sandbox(tmp_path):
    """El cabezal es un entero 0..9: no hay rutas arbitrarias."""
    pc = PCReal(raiz_sandbox=tmp_path / "sb")
    assert not pc.mover_cabezal_disco(99)
    assert not pc.mover_cabezal_disco(-1)
