"""
Pantalla — la interfaz de terminal de IA-SO Brooder
====================================================

Renderiza con ANSI puro (sin dependencias):

* **BIOS/POST**: la secuencia de arranque del sistema.
* **Pantalla de Brooder**: lo que la IA "dibuja" (su salida).
* **Monitor del sistema**: panel lateral con el estado de la
  máquina (tarea, ciclo, bus, acumulador, cabezal del disco...).
* **Recovery**: el menú de emergencia independiente del cerebro.

Detecta si el terminal soporta ANSI y degrada con elegancia a
texto plano (útil para CI, tuberías y terminales antiguos).
"""
from __future__ import annotations

import os
import sys

from brooder.constantes import NOMBRE_PROYECTO, VERSION, tokens_a_texto

# ------------------------------------------------------------------
# soporte ANSI
# ------------------------------------------------------------------
_terminal_tty = sys.stdout.isatty()


def _color_activado() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not _terminal_tty:
        return False
    return os.environ.get("TERM", "dumb") != "dumb"


COLOR = _color_activado()


def c(texto: str, codigo: str) -> str:
    """Envuelve texto en color ANSI (si procede)."""
    if not COLOR:
        return texto
    return f"\033[{codigo}m{texto}\033[0m"


def negrita(t: str) -> str:
    return c(t, "1")


def cian(t: str) -> str:
    return c(t, "36")


def verde(t: str) -> str:
    return c(t, "32")


def rojo(t: str) -> str:
    return c(t, "31")


def amarillo(t: str) -> str:
    return c(t, "33")


def tenue(t: str) -> str:
    return c(t, "2")


def limpiar() -> None:
    if COLOR:
        print("\033[2J\033[H", end="", flush=True)


ANCHO_PANTALLA = 34   # columnas de la pantalla de Brooder
ALTO_PANTALLA = 6     # líneas visibles


# ------------------------------------------------------------------
# BIOS / POST
# ------------------------------------------------------------------
def splash_bios(comprobaciones, dispositivo: str = "CPU", rapido: bool = False):
    """Secuencia de arranque (POST + montaje del SSD)."""
    lineas = [
        negrita(cian(f"{NOMBRE_PROYECTO} BIOS v{VERSION}")),
        tenue("(c) Proyecto IA-SO Brooder — licencia MIT"),
        "",
        "POST (Power-On Self Test):",
    ]
    for nombre, detalle in comprobaciones:
        lineas.append(f"  [ {verde('OK')} ] {nombre:24s} {detalle}")
    lineas += [
        "",
        f"Dispositivo de inferencia: {negrita(dispositivo)}",
        "Montando imagen SSD ......................... " + verde("LISTO"),
        "Restaurando estado del sistema .............. " + verde("LISTO"),
        "Despertando al cerebro de Brooder ........... " + verde("LISTO"),
        "",
        negrita("IA-SO BROODER EN MARCHA"),
        "",
    ]
    for linea in lineas:
        print(linea, flush=True)
        if not rapido:
            import time

            time.sleep(0.05)


# ------------------------------------------------------------------
# pantalla de Brooder + monitor
# ------------------------------------------------------------------
def caja(titulo: str, lineas: list, ancho: int) -> list:
    """Dibuja una caja con título y contenido."""
    borde = "+" + "-" * (ancho - 2) + "+"
    salida = [cian(borde)]
    titulo_formateado = f"| {titulo}" + " " * max(0, ancho - 3 - len(titulo)) + "|"
    salida.append(cian(titulo_formateado))
    salida.append(cian(borde))
    for linea in lineas:
        cuerpo = linea[: ancho - 4]
        salida.append(cian("|") + f" {cuerpo}" + " " * max(0, ancho - 4 - len(cuerpo)) + cian("|"))
    salida.append(cian(borde))
    return salida


def render_pantalla_brooder(pantalla_tokens: list, ancho: int = ANCHO_PANTALLA) -> list:
    """La pantalla virtual de la IA, con envoltura de línea."""
    texto = tokens_a_texto(pantalla_tokens)
    lineas = []
    for i in range(0, max(len(texto), 1), ancho - 4):
        lineas.append(texto[i : i + ancho - 4])
        if len(lineas) >= ALTO_PANTALLA:
            break
    while len(lineas) < 3:
        lineas.append("")
    return caja("PANTALLA DE BROODER", lineas, ancho)


def render_monitor(
    tarea: str,
    ciclo: int,
    presupuesto: int,
    instante,
    estado_sistema: str = "",
) -> list:
    """El panel de monitor del sistema."""
    ancho = 44
    lineas = [
        f"Tarea actual      : {negrita(tarea)}",
        f"Ciclo             : {ciclo}/{presupuesto}",
        f"Bus de datos      : "
        f"{'-' if not instante.bus_valido else tokens_a_texto([instante.bus_valor])}"
        f"{' (valido)' if instante.bus_valido else ' (vacio)'}",
        f"CPU acumulador    : {instante.acumulador}",
        f"Disco cabezal     : {instante.disco_cabezal}",
        f"RAM puntero       : {instante.memoria_puntero}",
        f"Teclado pendiente : {instante.teclado_pendientes} tokens",
        f"Ultimo evento     : {instante.ultimo_evento[:26] or '-'}",
    ]
    if estado_sistema:
        lineas.append(tenue(estado_sistema[:38]))
    return caja("MONITOR DEL SISTEMA", lineas, ancho)


def mostrar_resultado_solicitud(resultado, detallado: bool = False) -> None:
    """Imprime el veredicto de una solicitud atendida."""
    simbolo = verde("[ OK ]") if resultado.exito else rojo("[FALLO]")
    esperado = tokens_a_texto(resultado.solicitud.esperado)
    print(
        f"  {simbolo} {resultado.solicitud.descripcion():26s} "
        f"-> pantalla='{resultado.pantalla}' "
        f"(esperado '{esperado}') "
        f"{resultado.detalle_dispositivos}"
    )
    if detallado:
        for nombre, evento in resultado.eventos:
            print(f"         {tenue('-')} {nombre:28s} {tenue(evento)}")


# ------------------------------------------------------------------
# RECOVERY — el "sistema nervioso autónomo"
# ------------------------------------------------------------------
MENU_RECOVERY = [
    ("R", "Reiniciar la IA", "reiniciar"),
    ("E", "Estado del sistema", "estado"),
    ("D", "Diagnostico completo", "diagnostico"),
    ("M", "Modo seguro (sin red, solo texto)", "modo_seguro"),
    ("A", "Apagar", "apagar"),
]


def menu_recovery() -> str:
    """Imprime el menú de recuperación y devuelve la opción elegida."""
    print()
    print(negrita(cian("╔════════════════════════════════════════╗")))
    print(negrita(cian("║        IA-SO BROODER  ·  RECOVERY       ║")))
    print(negrita(cian("╚════════════════════════════════════════╝")))
    print(tenue("  Entorno de emergencia independiente de la IA."))
    print()
    for tecla, etiqueta, _ in MENU_RECOVERY:
        print(f"   [{amarillo(tecla)}] {etiqueta}")
    print()
    eleccion = input("  Elige una opción> ").strip().upper()
    for tecla, _, accion in MENU_RECOVERY:
        if eleccion == tecla:
            return accion
    return "estado"


def ayuda_interactiva() -> None:
    """Ayuda del intérprete de arranque."""
    print()
    print(negrita("Cómo hablar con Brooder:"))
    print("  HOLA           -> eco: Brooder repetirá en pantalla lo que teclees")
    print("  3+5            -> suma: usará la CPU para calcular y mostrar el resultado")
    print("  guardar 4 G    -> escribirá G en la ranura 4 del disco y la recuperará")
    print("  recordar 2 Z   -> igual, pero en la RAM")
    print("  aviso A        -> mostrará A y pitará al leer la alarma")
    print(tenue("  Comandos: :ayuda  :recovery  :salir"))
    print()


def banner_sesion() -> None:
    print()
    print(
        negrita(
            "Sesión interactiva. Escribe una solicitud "
            "(o :ayuda para ver los formatos, :salir para apagar)."
        )
    )
    print()
