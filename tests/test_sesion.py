"""
Tests de la sesión compartida (Fase 2 — emulador web)
=====================================================

La clase clave del refactor: ``SesionInteractiva`` debe producir las
MISMAS líneas que la consola clásica. El test de equivalencia lo prueba
byte a byte: alimenta la misma sesión a ``cmd_arrancar`` (por stdin
scriptado) y a la sesión directa, y compara las salidas.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from brooder.sesion import encender

RUTA_SSD = Path("ssd") / "brooder.img"


def _encender_virtual():
    return encender(ssd=RUTA_SSD, maquina_real=False, rapido=True, capturar=True)


# ------------------------------------------------------------------
# arranque capturado (la vía del emulador web)
# ------------------------------------------------------------------
def test_arranque_capturado_muestra_post_y_banner():
    sesion, lineas = _encender_virtual()
    texto = "\n".join(lineas)
    assert "IA-SO Brooder BIOS v" in texto
    assert "contrato 26x23" in texto
    assert "IA-SO BROODER EN MARCHA" in texto
    assert "Máquina: VIRTUAL | Cerebro: SSD brooder.img" in texto
    assert "Sesión interactiva" in texto
    assert sesion.encendida


def test_arranque_sin_ssd_avisa_por_lineas(tmp_path):
    sesion, lineas = encender(
        ssd=tmp_path / "no-existe.img", maquina_real=False,
        rapido=True, capturar=True,
    )
    assert any("no hay imagen SSD" in linea for linea in lineas)
    assert any("cerebro sin entrenar" in linea for linea in lineas)


# ------------------------------------------------------------------
# equivalencia byte a byte: consola clásica vs sesión directa
# ------------------------------------------------------------------
GUIÓN = [
    "3+5",
    ":ayuda",
    "leer 3P",          # FALLO honesto: pendrive ausente
    ":pendrive",
    "montar",
    "escribir 3P",
    "leer 3P",
    ":salir",
]


def _quitar_ansi(linea: str) -> str:
    return linea.replace("\x1b", "").replace("[36m", "").replace("[0m", "")


def test_consola_web_igual_que_la_clasica(monkeypatch):
    """La MISMA ruta de código: cmd_arrancar (stdin scriptado) produce
    exactamente las mismas líneas que atender_linea de la sesión."""
    # 1) la vía clásica: cmd_arrancar con input scriptado y stdout capturado
    respuestas = list(GUIÓN)

    def input_falso(_prompt):
        if respuestas:
            return respuestas.pop(0)
        return ""

    monkeypatch.setattr("builtins.input", input_falso)
    args = argparse.Namespace(
        ssd=RUTA_SSD, maquina_real=False, sandbox=Path("brooder_sandbox"),
        rapido=True, detallado=False,
    )
    captura_cli = io.StringIO()
    from brooder.cli import cmd_arrancar

    with redirect_stdout(captura_cli):
        cmd_arrancar(args)
    lineas_cli = [
        _quitar_ansi(l) for l in captura_cli.getvalue().splitlines()
        if _quitar_ansi(l).strip() != "brooder> "
    ]

    # 2) la vía del emulador: encender + atender_linea
    sesion, lineas_arranque = _encender_virtual()
    lineas_web = [_quitar_ansi(l) for l in lineas_arranque]
    for entrada in GUIÓN:
        lineas_web += [_quitar_ansi(l) for l in sesion.atender_linea(entrada)]
        if not sesion.encendida:
            lineas_web += [_quitar_ansi(l) for l in sesion.apagar()]

    # 3) byte a byte (sin ANSI ni prompts, que en la web los dibuja el navegador)
    assert lineas_cli == lineas_web


# ------------------------------------------------------------------
# el ciclo del pendrive por líneas (el hotfix del parser heredado)
# ------------------------------------------------------------------
def test_ciclo_del_pendrive_por_sesion():
    sesion, _ = _encender_virtual()
    fallo = "\n".join(sesion.atender_linea("leer 3P"))
    assert "[FALLO]" in fallo and "pendrive=ausente" in fallo

    assert "pendrive conectado" in "\n".join(sesion.atender_linea(":pendrive"))
    assert "pendrive=montado" in "\n".join(sesion.atender_linea("montar"))

    escritura = "\n".join(sesion.atender_linea("escribir 3P"))
    assert "[ OK ]" in escritura and "pendrive[3]='P'" in escritura

    lectura = "\n".join(sesion.atender_linea("leer 3P"))
    assert "[ OK ]" in lectura and "pendrive[3]='P'" in lectura


# ------------------------------------------------------------------
# recovery por líneas (la vía de la consola web)
# ------------------------------------------------------------------
def test_recovery_por_lineas():
    sesion, _ = _encender_virtual()

    menu = "\n".join(sesion.atender_linea(":recovery"))
    assert "RECOVERY" in menu
    assert sesion.modo == "recovery"

    estado = "\n".join(sesion.atender_linea("E"))
    assert sesion.modo == "normal"          # vuelve al modo normal
    diag = json.loads(estado)               # la opción E imprime JSON válido
    assert "dispositivos" in diag and "pendrive" in diag["dispositivos"]

    # opción desconocida -> acción por defecto (estado), como el CLI
    "\n".join(sesion.atender_linea("X"))
    assert sesion.modo == "normal" and sesion.encendida

    sesion.atender_linea(":recovery")
    apagar = "\n".join(sesion.atender_linea("A"))
    assert "Apagando..." in apagar
    assert not sesion.encendida
    assert "Sesión cerrada." in "\n".join(sesion.apagar())


# ------------------------------------------------------------------
# apagado: acta de cierre + persistencia solo en máquina real
# ------------------------------------------------------------------
def test_apagar_persiste_solo_en_maquina_real(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = Path(__file__).resolve().parent.parent
    ssd = repo / "ssd" / "brooder.img"

    # máquina virtual: sin acta
    sesion, _ = encender(ssd=ssd, maquina_real=False, rapido=True, capturar=True)
    cierre = sesion.apagar()
    assert cierre == [f"Sesión cerrada. {sesion.nucleo.estado.resumen()}."]
    assert not (tmp_path / "estado.json").exists()

    # máquina real: acta + pendrive persistente
    sesion, _ = encender(
        ssd=ssd, maquina_real=True, sandbox=tmp_path / "sandbox",
        rapido=True, capturar=True,
    )
    sesion.atender_linea(":pendrive")
    sesion.atender_linea("montar")
    escritura = "\n".join(sesion.atender_linea("escribir 3P"))
    assert "pendrive[3]='P'" in escritura
    sesion.apagar()
    assert (tmp_path / "sandbox" / "estado.json").exists()
    datos = json.loads((tmp_path / "sandbox" / "pendrive.json").read_text())
    assert datos["ranuras"][3] != 0        # la 'P' grabada de verdad
