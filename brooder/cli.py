"""
CLI de IA-SO Brooder
====================

El punto de entrada del proyecto. Ciclo de vida completo:

    brooder incubar    -> entrenar a la IA en tu "incubadora"
    brooder exportar   -> empaquetar el cerebro como imagen SSD
    brooder arrancar   -> encender la IA-SO (sesión interactiva)
    brooder demo       -> demostración no interactiva verificada
    brooder diagnostico-> evaluar el modelo montado en el SSD
    brooder graficar   -> curvas de recompensa del entrenamiento
    brooder primitivas -> listar el contrato de hardware
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from brooder.constantes import (
    NOMBRE_PROYECTO,
    TABLA_PRIMITIVAS,
    Tarea,
    VERSION,
    Primitiva,
)
from brooder.solicitudes import Solicitud

RUTA_BASE = Path.cwd()
RUTA_SSD_DEFECTO = RUTA_BASE / "ssd" / "brooder.img"
RUTA_ENTRENAMIENTO_DEFECTO = RUTA_BASE / "entrenamiento"


# ------------------------------------------------------------------
# incubar
# ------------------------------------------------------------------
def cmd_incubar(args) -> int:
    from brooder.incubadora import ConfiguracionPPO, Incubadora

    cfg = ConfiguracionPPO(
        pasos_totales=args.pasos,
        semilla=args.semilla,
        parar_al_converger=not args.sin_parada,
    )
    incubadora = Incubadora(cfg=cfg, dir_salida=args.salida)
    resumen = incubadora.entrenar(reanudar=args.reanudar)

    print()
    print("─" * 52)
    print("  Incubación finalizada")
    print("─" * 52)
    for tarea, exito in resumen["exito_eval_final"].items():
        marca = "✔" if exito >= 0.85 else "✘"
        print(f"  {marca} {tarea:10s} {exito:.0%}")
    print(f"  Mejor cerebro: {args.salida}/mejor.pt")
    print(f"  Siguiente paso: brooder exportar --desde {args.salida}/mejor.pt")
    return 0


# ------------------------------------------------------------------
# exportar
# ------------------------------------------------------------------
def _cmd_exportar(args) -> int:
    from brooder.nucleo import exportar_ssd

    desde = Path(args.desde)
    if not desde.exists():
        print(f"No existe el modelo: {desde}", file=sys.stderr)
        return 1

    # adjuntar métricas si existen
    metricas = {}
    ruta_metricas = desde.parent / "metricas.jsonl"
    if ruta_metricas.exists():
        ultimas = [json.loads(l) for l in ruta_metricas.read_text().splitlines() if l.strip()]
        metricas = ultimas[-1] if ultimas else {}

    salida = exportar_ssd(desde, args.salida, metricas=metricas)
    tamano_kb = salida.stat().st_size / 1024
    print(f"Imagen SSD exportada: {salida} ({tamano_kb:.0f} KiB)")
    print(f"Arranca la IA-SO con: brooder arrancar --ssd {salida}")
    return 0


# ------------------------------------------------------------------
# arrancar (sesión interactiva)
# ------------------------------------------------------------------
def cmd_arrancar(args) -> int:
    from brooder import pantalla
    from brooder.nucleo import NucleoBrooder, montar_ssd
    from brooder.primitivas.reales import PCReal
    from brooder.primitivas.virtual import PCVirtual

    # 1. máquina (virtual por defecto, real con sandbox)
    if args.maquina_real:
        maquina = PCReal(raiz_sandbox=args.sandbox)
        tipo_maquina = "REAL (sandbox en disco)"
    else:
        maquina = PCVirtual()
        tipo_maquina = "VIRTUAL"

    # 2. cerebro: desde SSD o en blanco (para ver el arranque sin entrenar)
    estado_sistema = None
    if args.ssd and Path(args.ssd).exists():
        cerebro, estado_sistema, manifiesto = montar_ssd(args.ssd)
        origen = f"SSD {Path(args.ssd).name}"
    else:
        from brooder.cerebro import CerebroBrooder

        cerebro = CerebroBrooder()
        origen = "cerebro sin entrenar (aleatorio)"
        print(
            pantalla.amarillo(
                "Aviso: no hay imagen SSD; se arrancará un cerebro aleatorio."
            )
        )

    nucleo = NucleoBrooder(maquina, cerebro, estado=estado_sistema)
    estado = nucleo.estado
    estado.anotar_arranque()

    # 3. POST + bienvenida
    pantalla.splash_bios(nucleo.post(), rapido=args.rapido)
    print(f"Máquina: {tipo_maquina} | Cerebro: {origen}")
    print(f"Sistema: {estado.resumen()}")
    pantalla.banner_sesion()

    # 4. bucle de atención
    while True:
        try:
            linea = input(pantalla.cian("brooder> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Apagando IA-SO... hasta pronto.")
            break

        if not linea:
            continue
        comando = linea.lower()
        if comando in (":salir", ":apagar", "salir", "exit"):
            print("Apagando IA-SO... hasta pronto.")
            break
        if comando == ":ayuda" or comando == "ayuda":
            pantalla.ayuda_interactiva()
            continue
        if comando == ":recovery" or comando == "recovery":
            if _bucle_recovery(nucleo, args):
                break
            continue

        solicitud = Solicitud.desde_texto(linea)
        if solicitud is None:
            print(
                pantalla.amarillo(
                    "No entendí la solicitud. Prueba 'HOLA', '3+5', "
                    "'guardar 4 G', 'recordar 2 Z', 'aviso A' o :ayuda."
                )
            )
            continue

        resultado = nucleo.atender_solicitud(solicitud)
        pantalla.mostrar_resultado_solicitud(resultado, detallado=args.detallado)

    # 5. apagado: persistir el estado
    if args.maquina_real:
        estado.guardar(Path(args.sandbox) / "estado.json")
    print(f"Sesión cerrada. {estado.resumen()}.")
    return 0


def _bucle_recovery(nucleo, args) -> bool:
    """Menú de recuperación. Devuelve True si hay que apagar."""
    from brooder import pantalla

    accion = pantalla.menu_recovery()
    if accion == "reiniciar":
        print("Reiniciando el cerebro de Brooder...")
        nucleo.cerebro.eval()
        nucleo.estado.anotar_arranque()
        print(pantalla.verde("IA reiniciada."))
    elif accion == "estado":
        diag = nucleo.diagnostico()
        print(json.dumps(diag, ensure_ascii=False, indent=2))
    elif accion == "diagnostico":
        diag = nucleo.diagnostico()
        for k, v in diag.items():
            print(f"  {k}: {v}")
    elif accion == "modo_seguro":
        print("Modo seguro: la red permanece desactivada (política por defecto).")
    elif accion == "apagar":
        print("Apagando...")
        return True
    return False


# ------------------------------------------------------------------
# demo (no interactiva, verificada)
# ------------------------------------------------------------------
DEMO_SOLICITUDES = [
    ("eco HOLA", "ECO"),
    ("eco BROODER", "ECO"),
    ("3+5", "SUMA"),
    ("7+8", "SUMA"),       # resultado de dos dígitos
    ("guardar 4 G", "GUARDAR"),
    ("recordar 2 Z", "RECORDAR"),
    ("aviso A", "AVISO"),
]


def cmd_demo(args) -> int:
    from brooder import pantalla
    from brooder.nucleo import NucleoBrooder, montar_ssd
    from brooder.primitivas.reales import PCReal
    from brooder.primitivas.virtual import PCVirtual

    maquina = PCReal(raiz_sandbox=args.sandbox) if args.maquina_real else PCVirtual()
    cerebro, estado, manifiesto = montar_ssd(args.ssd)

    nucleo = NucleoBrooder(maquina, cerebro, estado=estado)
    estado.anotar_arranque()

    pantalla.splash_bios(nucleo.post(), rapido=True)
    print(f"Máquina: {'REAL (sandbox)' if args.maquina_real else 'VIRTUAL'}")
    if manifiesto.get("metricas", {}).get("exito_eval_final"):
        print(
            "Cerebro incubado: "
            + " | ".join(
                f"{t} {v:.0%}"
                for t, v in manifiesto["metricas"]["exito_eval_final"].items()
            )
        )
    print()
    print(negrita_local("DEMOSTRACIÓN: la IA-SO atiende solicitudes reales"))
    print("─" * 66)

    exitos = 0
    for texto, _tarea in DEMO_SOLICITUDES:
        solicitud = Solicitud.desde_texto(texto)
        resultado = nucleo.atender_solicitud(solicitud)
        exitos += resultado.exito
        pantalla.mostrar_resultado_solicitud(resultado, detallado=args.detallado)

    print("─" * 66)
    total = len(DEMO_SOLICITUDES)
    marca = verde_local if exitos == total else amarillo_local
    print(f"  Resultado de la demo: {marca(f'{exitos}/{total} solicitudes resueltas')}")
    print(f"  {estado.resumen()}")
    if args.maquina_real:
        print(
            f"  El disco de Brooder persiste en: "
            f"{Path(args.sandbox) / 'disco'} (revísalo: son archivos reales)"
        )
    return 0 if exitos == total else 1


def negrita_local(t):
    from brooder import pantalla

    return pantalla.negrita(t)


def verde_local(t):
    from brooder import pantalla

    return pantalla.verde(t)


def amarillo_local(t):
    from brooder import pantalla

    return pantalla.amarillo(t)


# ------------------------------------------------------------------
# diagnostico
# ------------------------------------------------------------------
def cmd_diagnostico(args) -> int:
    from brooder.incubadora import evaluar
    from brooder.nucleo import montar_ssd
    from brooder.cerebro import CerebroBrooder

    if args.ssd and Path(args.ssd).exists():
        cerebro, _, manifiesto = montar_ssd(args.ssd)
        print(f"Imagen SSD: {args.ssd}")
        print(f"Fecha de incubación: {manifiesto.get('fecha', '?')}")
        pasos = manifiesto.get("pasos_entrenamiento")
        if pasos:
            print(f"Pasos de entrenamiento: {pasos}")
    else:
        cerebro = CerebroBrooder()
        print("Sin SSD: evaluando un cerebro sin entrenar (referencia aleatoria)")

    print()
    print("Evaluación determinista por tarea "
          f"({args.solicitudes} solicitudes/tarea):")
    resultados = evaluar(cerebro, list(Tarea), n_solicitudes=args.solicitudes)
    todo_ok = True
    for tarea, exito in sorted(resultados.items()):
        marca = verde_local("✔") if exito >= 0.85 else rojo_local("✘")
        todo_ok &= exito >= 0.85
        print(f"  {marca} {tarea:10s} {exito:.0%}")
    print()
    print("Veredicto:", verde_local("DOMINIO COMPLETO") if todo_ok
          else amarillo_local("dominio parcial"))
    return 0 if todo_ok else 1


# ------------------------------------------------------------------
# graficar
# ------------------------------------------------------------------
def cmd_graficar(args) -> int:
    ruta_metricas = Path(args.metricas)
    if not ruta_metricas.exists():
        print(f"No hay métricas en {ruta_metricas}", file=sys.stderr)
        return 1
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no está instalado (pip install '.[graficas]')",
              file=sys.stderr)
        return 1

    registros = [
        json.loads(l)
        for l in ruta_metricas.read_text(encoding="utf-8").splitlines()
        if l.strip() and '"exito_eval"' in l
    ]
    if not registros:
        print("El fichero de métricas no tiene evaluaciones todavía.", file=sys.stderr)
        return 1

    pasos = [r["paso"] for r in registros]
    tareas = sorted({t for r in registros for t in r["exito_eval"]})
    colores = {
        "ECO": "#2a9d8f", "SUMA": "#e9c46a", "GUARDAR": "#f4a261",
        "RECORDAR": "#e76f51", "AVISO": "#9b5de5",
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
    )
    for tarea in tareas:
        xs, ys = [], []
        for r in registros:
            if tarea in r["exito_eval"]:
                xs.append(r["paso"])
                ys.append(r["exito_eval"][tarea])
        ax1.plot(xs, ys, label=tarea, color=colores.get(tarea), lw=2)
        ax2.plot(
            xs,
            [r["exito_medio_eval"] for r in registros if tarea in r["exito_eval"]][: len(xs)],
            color=colores.get(tarea), lw=0.8, alpha=0.0,
        )  # placeholder para mantener escala
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel("Éxito en evaluación")
    ax1.set_title("IA-SO Brooder — incubación por refuerzo (PPO)")
    ax1.legend(loc="lower right", frameon=False)
    ax1.grid(alpha=0.3)

    ax2.plot(
        pasos, [r["exito_medio_eval"] for r in registros],
        color="#264653", lw=2.5,
    )
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("Pasos de entrenamiento")
    ax2.set_ylabel("Éxito medio")
    ax2.grid(alpha=0.3)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f"Curva guardada en {salida}")
    return 0


# ------------------------------------------------------------------
# primitivas (documentación ejecutable)
# ------------------------------------------------------------------
def cmd_primitivas(_args) -> int:
    print(negrita_local("CONTRATO DE PRIMITIVAS DE IA-SO BROODER"))
    print("La IA nunca toca el hardware: solicita estas operaciones y el")
    print("núcleo las valida y ejecuta.")
    print("─" * 76)
    print(f"{'ID':>3}  {'PRIMITIVA':30} {'ARGUMENTO':10} DESCRIPCIÓN")
    print("─" * 76)
    for primitiva in Primitiva:
        info = TABLA_PRIMITIVAS[primitiva]
        print(f"{int(primitiva):>3}  {info.nombre:30} {info.tipo_argumento:10} {info.descripcion}")
    print("─" * 76)
    print("ARG_BUS = 38: 'toma el valor que hay ahora en el bus de datos'.")
    return 0


# ------------------------------------------------------------------
# parser
# ------------------------------------------------------------------
def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brooder",
        description=f"{NOMBRE_PROYECTO} v{VERSION} — un sistema operativo que se incuba, no se instala.",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("incubar", help="entrenar a Brooder en la incubadora (PPO)")
    p.add_argument("--pasos", type=int, default=800_000, help="pasos de entrenamiento (total absoluto)")
    p.add_argument("--semilla", type=int, default=1234)
    p.add_argument("--salida", type=Path, default=RUTA_ENTRENAMIENTO_DEFECTO)
    p.add_argument("--reanudar", action="store_true",
                   help="continuar desde el último checkpoint de --salida")
    p.add_argument("--sin-parada", action="store_true",
                   help="no detenerse al converger")
    p.set_defaults(func=cmd_incubar)

    p = sub.add_parser("exportar", help="empaquetar el cerebro como imagen SSD")
    p.add_argument("--desde", type=Path, default=RUTA_ENTRENAMIENTO_DEFECTO / "mejor.pt")
    p.add_argument("--salida", type=Path, default=RUTA_SSD_DEFECTO)
    p.set_defaults(func=_cmd_exportar)

    p = sub.add_parser("arrancar", help="encender la IA-SO (interactivo)")
    p.add_argument("--ssd", type=Path, default=RUTA_SSD_DEFECTO)
    p.add_argument("--maquina-real", action="store_true",
                   help="usar el PC real con sandbox (disco en ./brooder_sandbox)")
    p.add_argument("--sandbox", type=Path, default=Path("brooder_sandbox"))
    p.add_argument("--rapido", action="store_true", help="saltarse las pausas del POST")
    p.add_argument("--detallado", action="store_true", help="mostrar cada ciclo")
    p.set_defaults(func=cmd_arrancar)

    p = sub.add_parser("demo", help="demostración no interactiva verificada")
    p.add_argument("--ssd", type=Path, default=RUTA_SSD_DEFECTO)
    p.add_argument("--maquina-real", action="store_true")
    p.add_argument("--sandbox", type=Path, default=Path("brooder_sandbox"))
    p.add_argument("--detallado", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("diagnostico", help="evaluar el cerebro del SSD por tarea")
    p.add_argument("--ssd", type=Path, default=RUTA_SSD_DEFECTO)
    p.add_argument("--solicitudes", type=int, default=60)
    p.set_defaults(func=cmd_diagnostico)

    p = sub.add_parser("graficar", help="dibujar curvas de recompensa")
    p.add_argument("--metricas", type=Path,
                   default=RUTA_ENTRENAMIENTO_DEFECTO / "metricas.jsonl")
    p.add_argument("--salida", type=Path, default=RUTA_BASE / "img" / "curva_recompensa.png")
    p.set_defaults(func=cmd_graficar)

    p = sub.add_parser("primitivas", help="listar el contrato de hardware")
    p.set_defaults(func=cmd_primitivas)

    return parser


def principal(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # tubería cerrada (p. ej. `brooder primitivas | head`): salida limpia
        try:
            sys.stdout.close()
        except OSError:
            pass
        return 0
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130


if __name__ == "__main__":
    raise SystemExit(principal())
