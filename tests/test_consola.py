"""Tests de robustez de consola: el CLI no debe romperse en cp1252.

Motivación (bug real, Windows 11 + PowerShell 5.1): PowerShell conecta la
salida de los procesos externos mediante una tubería, por lo que Python
codifica con la página de código ANSI (cp1252). El primer ``print`` con un
carácter de caja (``─``) lanzaba ``UnicodeEncodeError`` y mataba la demo.
"""
import io
import sys

from brooder.cli import _asegurar_consola


def test_degrada_a_reemplazo_en_cp1252():
    """Un flujo cp1252 puede imprimir '─' sin lanzar UnicodeEncodeError."""
    buffer = io.BytesIO()
    flujo = io.TextIOWrapper(buffer, encoding="cp1252")
    stdout_previo = sys.stdout
    sys.stdout = flujo
    contenido = b""
    try:
        _asegurar_consola()
        # antes del guard esto lanzaba: charmap codec can't encode '─'
        print("─" * 10)
        flujo.flush()
        contenido = buffer.getvalue()
    finally:
        sys.stdout = stdout_previo
        flujo.close()
    # lo impreso son '?' de cp1252 (0x3F): degradado, no roto
    assert contenido.rstrip(b"\r\n") == b"?" * 10


def test_no_toca_los_flujos_utf8():
    """Un flujo UTF-8 conserva sus caracteres y su modo de error estricto."""
    buffer = io.BytesIO()
    flujo = io.TextIOWrapper(buffer, encoding="utf-8")
    stdout_previo = sys.stdout
    sys.stdout = flujo
    contenido = b""
    try:
        _asegurar_consola()
        assert flujo.errors == "strict"
        print("─" * 3)
        flujo.flush()
        contenido = buffer.getvalue()
    finally:
        sys.stdout = stdout_previo
        flujo.close()
    # '─' UTF-8 completo (no degradado a '?')
    assert contenido.rstrip(b"\r\n") == b"\xe2\x94\x80" * 3


def test_tolera_flujos_sin_reconfigure():
    """Capturas tipo StringIO (sin .encoding/.reconfigure) no rompen el guard."""
    stdout_previo = sys.stdout
    sys.stdout = io.StringIO()
    try:
        _asegurar_consola()  # no debe lanzar AttributeError
    finally:
        sys.stdout = stdout_previo
