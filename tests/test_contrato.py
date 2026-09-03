"""Tests del hotfix de contrato (Fase 1): el POST declara el cerebro.

Nació de un fallo real en campo: aplicar el parche de la Fase 1 sin
copiar su imagen SSD dejaba el cerebro 21x18 montado con un kernel que
habla 24x20. La compatibilidad de prefijo hace el arranque LEGAL, pero
'montar' fallaba en silencio (presupuesto agotado sin primitiva
posible). Estos tests fijan el contrato del remedio: el POST lo declara,
el aviso explica las faltas y splash_bios pinta niveles sin romper la
alineación.
"""
import argparse

import pytest

from brooder.cerebro import CerebroBrooder
from brooder.constantes import N_PRIMITIVAS, OBS_DIM
from brooder.nucleo import NucleoBrooder, aviso_contrato
from brooder.pantalla import splash_bios
from brooder.primitivas.virtual import PCVirtual


def _cerebro_viejo() -> CerebroBrooder:
    """El cerebro de la Fase 0.5: contrato 21x18 (sin dispositivo)."""
    return CerebroBrooder(dim_entrada=21, n_primitivas=18)


# ------------------------------------------------------------------
# POST: la línea "Cerebro" declara el contrato montado
# ------------------------------------------------------------------
def test_post_declara_contrato_moderno_sin_aviso():
    nucleo = NucleoBrooder(PCVirtual(), CerebroBrooder())
    comprobaciones = nucleo.post()
    linea = [c for c in comprobaciones if c[0] == "Cerebro"]
    assert len(linea) == 1
    nombre, detalle = linea[0][0], linea[0][1]
    assert detalle == f"contrato {OBS_DIM}x{N_PRIMITIVAS}"
    assert len(linea[0]) == 2  # sin nivel de aviso


def test_post_marca_imagen_antigua_con_aviso():
    nucleo = NucleoBrooder(PCVirtual(), _cerebro_viejo())
    comprobaciones = nucleo.post()
    linea = [c for c in comprobaciones if c[0] == "Cerebro"]
    assert len(linea) == 1
    assert linea[0][1] == "contrato 21x18 (imagen antigua)"
    assert linea[0][2] == "aviso"  # nivel amarillo


# ------------------------------------------------------------------
# aviso_contrato: describe el desfase y sus faltas exactas
# ------------------------------------------------------------------
def test_aviso_contrato_viejo_nombra_las_faltas():
    aviso = aviso_contrato(_cerebro_viejo())
    assert aviso is not None
    assert "MONTAR_DISPOSITIVO" in aviso
    assert "DESMONTAR_DISPOSITIVO" in aviso
    assert f"primitivas 18/{N_PRIMITIVAS}" in aviso
    assert f"observación 21/{OBS_DIM}" in aviso
    # el remedio menciona la imagen SSD reentrenada
    assert "contrato viejo" in aviso


def test_aviso_contrato_moderno_es_none():
    assert aviso_contrato(CerebroBrooder()) is None


# ------------------------------------------------------------------
# splash_bios: niveles de aviso sin romper la alineación
# ------------------------------------------------------------------
def test_splash_bios_alinea_ok_y_aviso(capsys):
    splash_bios(
        [
            ("CPU (acumulador)", "lista"),
            ("Cerebro", "contrato 21x18 (imagen antigua)", "aviso"),
        ],
        rapido=True,
    )
    salida = capsys.readouterr().out
    assert "[ OK    ]" in salida
    assert "[ AVISO ]" in salida
    lineas = [l for l in salida.splitlines() if l.startswith("  [ ")]
    # las dos etiquetas ocupan la misma columna (ancho fijo)
    columnas = {l.index("]") for l in lineas}
    assert len(columnas) == 1
    # el detalle del aviso también viaja en la línea
    assert "contrato 21x18 (imagen antigua)" in salida


# ------------------------------------------------------------------
# regresión rojo_local: diagnostico no crashea con tareas fallidas
# ------------------------------------------------------------------
def test_diagnostico_con_tarea_fallida_devuelve_1_sin_crashear(
    monkeypatch, capsys
):
    """El ✘ de cmd_diagnostico usaba rojo_local, que no existía.

    Un cerebro del contrato viejo deja DISPOSITIVO al 0 %: la rama
    fallida se ejecuta por fin y debe devolver 1, no NameError.
    """
    from brooder.cli import cmd_diagnostico

    # v0.4.0: evaluar creció con kwargs opcionales (semilla ya existía
    # como parámetro posicional y ahora llega 'estado_conector' en el
    # bloque de invarianza del diagnóstico); el fake los absorbe.
    def _evaluar_fallando(
        cerebro, tareas, n_solicitudes=2, con_trazado=False, **_
    ):
        return {"DISPOSITIVO": 0.0}, {}

    monkeypatch.setattr("brooder.incubadora.evaluar", _evaluar_fallando)
    args = argparse.Namespace(ssd=None, solicitudes=2)
    codigo = cmd_diagnostico(args)
    salida = capsys.readouterr().out
    assert codigo == 1
    assert "dominio parcial" in salida
    assert "✘" in salida  # la rama fallida se pintó, no explotó
