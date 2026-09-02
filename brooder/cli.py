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
    MENSAJES_LOG,
    NOMBRE_PROYECTO,
    N_PRIMITIVAS,
    OBS_DIM,
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
# robustez de consola (Windows)
# ------------------------------------------------------------------
def _asegurar_consola() -> None:
    """Endurece stdout/stderr frente a consolas sin UTF-8.

    PowerShell 5.1 (Windows) conecta la salida de los procesos externos a
    través de una tubería, así que Python no usa la API Unicode de la consola
    sino la página de código ANSI (cp1252 en español). Ahí los caracteres de
    caja (``─``, ``│``, ``✔``…) no existen y el primer ``print`` que los
    emita lanza ``UnicodeEncodeError``.

    La salida de emergencia es degradar esos caracteres a ``?`` (modo
    ``errors="replace"``) en lugar de romper la ejecución. Las consolas UTF-8
    (Linux/macOS, Windows Terminal con ``PYTHONUTF8=1``…) no se tocan.
    """
    for flujo in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        try:
            if flujo is None:
                continue
            codificacion = (flujo.encoding or "").lower().replace("-", "")
            if codificacion and codificacion != "utf8":
                flujo.reconfigure(errors="replace")
        except Exception:
            # flujos sin reconfigure (StringIO capturado, tuberías raras…):
            # no es un problema de este guard, seguimos sin romper nada.
            pass


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
    for tarea, tasa in resumen.get("trazado_eval_final", {}).items():
        marca = "✔" if tasa >= 0.85 else "✘"
        print(f"  {marca} trazado {tarea:6s} {tasa:.0%}  (REGISTRAR_LOG tras I/O)")
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
        # la última línea suele ser el resumen final (métricas anidadas):
        # se desanida para que el manifiesto y la demo las muestren
        if "resumen_final" in metricas:
            metricas = metricas["resumen_final"]

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
    from brooder.nucleo import NucleoBrooder, aviso_contrato, montar_ssd
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
    # hotfix contrato: el desfase kernel/cerebro se explica ANTES de que
    # el usuario vea [FALLO]s mudos en 'montar' (fallo real en campo:
    # parche de Fase 1 aplicado sin copiar su imagen SSD).
    aviso = aviso_contrato(cerebro)
    if aviso:
        print(pantalla.amarillo(f"Aviso: {aviso}."))
        print(
            pantalla.amarillo(
                "  Copia la imagen SSD reentrenada que acompaña al parche "
                "sobre ssd/brooder.img."
            )
        )
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
        if comando in (":pendrive", "pendrive", "usb"):
            # Fase 1: hot-plug manual del monitor — el mundo exterior
            # enchufa o retira el pendrive del conector USB virtual.
            _toggle_pendrive(nucleo)
            continue

        solicitud = Solicitud.desde_texto(linea)
        if solicitud is None:
            print(
                pantalla.amarillo(
                    "No entendí la solicitud. Prueba 'HOLA', '3+5', "
                    "'guardar 4 G', 'recordar 2 Z', 'aviso A', 'montar', "
                    "'escribir 3 P', 'leer 3 P' o :ayuda."
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


def _toggle_pendrive(nucleo) -> None:
    """Fase 1: enchufa o retira el pendrive del conector (hot-plug)."""
    from brooder import pantalla

    maquina = nucleo.maquina
    instante = maquina.instante()
    if instante.dispositivo_conectado:
        limpia = maquina.desconectar_dispositivo()
        if limpia:
            print(
                "Conector USB: pendrive desconectado (estaba desmontado; "
                "sus datos viajan con él)."
            )
        else:
            print(
                pantalla.rojo(
                    "Conector USB: pendrive retirado MONTADO -> "
                    "extraccion insegura registrada por el kernel; "
                    "LOS DATOS DEL PENDRIVE SE PIERDEN."
                )
            )
    else:
        maquina.conectar_dispositivo()
        print(
            "Conector USB: pendrive conectado. Pide 'montar' a la IA "
            "(y luego 'escribir 3 P' / 'leer 3 P')."
        )


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
    metricas = manifiesto.get("metricas", {})
    if metricas.get("exito_eval_final"):
        print(
            "Cerebro incubado: "
            + " | ".join(
                f"{t} {v:.0%}"
                for t, v in metricas["exito_eval_final"].items()
            )
        )
    if metricas.get("trazado_eval_final"):
        print(
            "Trazado del registro: "
            + " | ".join(
                f"{t} {v:.0%}"
                for t, v in metricas["trazado_eval_final"].items()
            )
        )
    print()
    print(negrita_local("DEMOSTRACIÓN: la IA-SO atiende solicitudes reales"))
    print("─" * 66)

    exitos = 0
    resultados = []
    for texto, _tarea in DEMO_SOLICITUDES:
        solicitud = Solicitud.desde_texto(texto)
        resultado = nucleo.atender_solicitud(solicitud)
        exitos += resultado.exito
        resultados.append(resultado)
        pantalla.mostrar_resultado_solicitud(resultado, detallado=args.detallado)

    print("─" * 66)
    total = len(DEMO_SOLICITUDES)
    marca = verde_local if exitos == total else amarillo_local
    print(f"  Resultado de la demo: {marca(f'{exitos}/{total} solicitudes resueltas')}")
    print(f"  {estado.resumen()}")

    # --- macro-primitiva REGISTRAR_LOG (consola del kernel) -------
    # Fase 0.5: el cerebro reentrenado traza por sí mismo. Durante
    # las 7 solicitudes de arriba, cada REGISTRAR_LOG emitido por la
    # política neuronal quedó anotado en resultado.trazos y en el
    # anillo del kernel. Si el cerebro montado es del contrato viejo
    # (17 salidas) o no trazó, se muestra la vía sintética de la
    # Fase 0 (eventos emitidos por el propio núcleo).
    trazos_totales = sum(len(r.trazos) for r in resultados)
    print()
    if trazos_totales:
        print(
            negrita_local(
                "MACRO-PRIMITIVA: REGISTRAR_LOG — decisión propia del cerebro"
            )
        )
        print("─" * 66)
        for resultado in resultados:
            if not resultado.trazos:
                continue
            mensajes = ", ".join(MENSAJES_LOG[m][1] for m in resultado.trazos)
            print(
                f"  {resultado.solicitud.descripcion():26s}"
                f" {verde_local('trazó')}: {mensajes}"
            )
        print()
        for linea in pantalla.render_panel_registro(nucleo.maquina.panel_registro()):
            print(linea)
        print()
        print(
            pantalla.tenue(
                "  Eventos emitidos por la política neuronal (no por un "
                "guion): el cerebro declara su I/O de almacenamiento."
            )
        )
        print(
            pantalla.tenue(
                "  Anillo del kernel: retiene 8 entradas; el panel muestra "
                "las 4 últimas."
            )
        )
    else:
        contrato = (
            "el cerebro montado usa el contrato viejo (17 salidas): "
            "no puede emitir REGISTRAR_LOG"
            if cerebro.n_primitivas <= int(Primitiva.REGISTRAR_LOG)
            else "el cerebro decidió no trazar en esta pasada"
        )
        print(negrita_local("MACRO-PRIMITIVA: REGISTRAR_LOG (consola del kernel)"))
        print("─" * 66)
        print(pantalla.tenue(f"  Sin trazos propios: {contrato}."))
        print(pantalla.tenue("  Eventos sintéticos emitidos por el núcleo:"))
        for mensaje in (3, 1, 2, 5, 7):
            # 3 proceso iniciado | 1 lectura | 2 escritura | 5 aviso | 7 error
            ok = nucleo.maquina.ejecutar(Primitiva.REGISTRAR_LOG, mensaje)
            nucleo.maquina.avanzar_paso()
            simbolo = verde_local("[ OK ]") if ok else pantalla.rojo("[FALLO]")
            print(f"  {simbolo} registrar_log({mensaje})")
        print()
        for linea in pantalla.render_panel_registro(nucleo.maquina.panel_registro()):
            print(linea)
        print()
        print(
            pantalla.tenue(
                "  Anillo del kernel: retiene 8 entradas; el panel muestra "
                "las 4 últimas."
            )
        )
    # --- Fase 1: pendrive virtual (hot-plug USB) --------------------
    disp_ok = _demo_pendrive(nucleo, cerebro)
    if args.maquina_real:
        print(
            f"  El disco de Brooder persiste en: "
            f"{Path(args.sandbox) / 'disco'} (revísalo: son archivos reales)"
        )
        print(
            f"  El pendrive persiste en: "
            f"{Path(args.sandbox) / 'pendrive.json'} (ranuras reales, "
            "sobreviven a apagar y encender)"
        )
    return 0 if exitos == total and disp_ok else 1


def _demo_pendrive(nucleo, cerebro) -> bool:
    """Secciones Fase 1 y 1.5 de la demo: el pendrive virtual.

    Fase 1 — el mundo exterior enchufa un pendrive; el cerebro
    percibe la presencia en sus canales y decide montarlo. Después
    se pide una "extracción segura" y decide desmontarlo. Por último,
    una extracción FORZADA con el pendrive montado muestra la
    protección del kernel: "extraccion insegura" queda registrada
    como ERROR aunque la IA no haya reaccionado.

    Fase 1.5 — almacenamiento real: escribir un dato en el pendrive
    montado, retirarlo DESMONTADO (extracción segura), reenchufarlo y
    leer el dato: EL PENDRIVE RECUERDA. El trazado I/O propio del
    dispositivo cierra la escena.

    Devuelve True si el cerebro administró el dispositivo con éxito
    (los cerebros del contrato viejo no pueden: la sección muestra
    entonces la vía sintética del núcleo, como el REGISTRAR_LOG de la
    Fase 0, y no cuenta para el código de salida).
    """
    from brooder import pantalla
    from brooder.constantes import CARACTER_DE_TOKEN, TOKEN_DE_CARACTER

    print()
    print(
        negrita_local("DISPOSITIVO EXTERNO: el pendrive virtual (hot-plug)")
    )
    print("─" * 66)
    maquina = nucleo.maquina
    contrato_viejo = (
        cerebro.dim_entrada < OBS_DIM or cerebro.n_primitivas < N_PRIMITIVAS
    )

    def _atender(modo: str, K=None, V=None) -> bool:
        datos = {"modo": modo}
        tokens, esperado = [], []
        if modo in ("escribir", "leer"):
            datos["K"], datos["V"] = K, V
            if modo == "escribir":
                tokens, esperado = [K, V, K], [V]
            else:
                tokens, esperado = [K], [V]
        solicitud = Solicitud(
            Tarea.DISPOSITIVO, tokens=tokens, esperado=esperado, datos=datos
        )
        resultado = nucleo.atender_solicitud(solicitud)
        if resultado.exito:
            print(
                f"  {verde_local('[ OK ]')} {modo:10s} -> decisión del "
                f"cerebro ({resultado.ciclos} ciclos, "
                f"{resultado.detalle_dispositivos})"
            )
        else:
            print(
                f"  {amarillo_local('[FALLO]')} {modo:10s} -> "
                f"{resultado.causa} ({resultado.detalle_dispositivos})"
            )
        return resultado.exito

    if contrato_viejo:
        print(
            pantalla.amarillo(
                f"  Cerebro del contrato viejo ({cerebro.dim_entrada} entradas / "
                f"{cerebro.n_primitivas} primitivas): no percibe el conector "
                "USB. El ciclo lo ejecuta el núcleo."
            )
        )
        print(
            pantalla.amarillo(
                "  Remedio: copia la imagen SSD reentrenada que acompaña "
                "al parche sobre ssd/brooder.img."
            )
        )
        maquina.conectar_dispositivo()
        print("  [ > ] conector USB: pendrive conectado (evento externo)")
        maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
        maquina.avanzar_paso()
        print(f"  {verde_local('[ OK ]')} montado por el núcleo")
        # vía sintética del almacenamiento: el kernel demuestra el medio
        maquina.mover_puntero_dispositivo(3)
        maquina.escribir_teclado([TOKEN_DE_CARACTER["Q"]])
        maquina.leer_teclado()
        maquina.escribir_dispositivo(38)
        maquina.avanzar_paso()
        print(f"  {verde_local('[ OK ]')} escribir ranura 3 por el núcleo")
        # extracción SEGURA: desmontar antes de retirar (el dato viaja)
        maquina.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)
        maquina.avanzar_paso()
        maquina.desconectar_dispositivo()
        maquina.conectar_dispositivo()
        maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
        maquina.mover_puntero_dispositivo(3)
        ok_sintetico = maquina.leer_dispositivo()
        maquina.avanzar_paso()
        valor = maquina.instante().bus_valor
        print(
            f"  {verde_local('[ OK ]')} el pendrive recuerda: ranura 3 -> "
            f"'{CARACTER_DE_TOKEN.get(valor, '?')}'"
            if ok_sintetico
            else "  [FALLO] lectura sintética"
        )
        maquina.ejecutar(Primitiva.DESMONTAR_DISPOSITIVO, 0)
        maquina.avanzar_paso()
        print(f"  {verde_local('[ OK ]')} desmontado por el núcleo")
        maquina.desconectar_dispositivo()
        print("  [ > ] desconexión limpia")
        maquina.conectar_dispositivo()
        maquina.ejecutar(Primitiva.MONTAR_DISPOSITIVO, 0)
        maquina.avanzar_paso()
        maquina.desconectar_dispositivo()
        print("  [ ! ] extracción forzada con el pendrive montado (kernel)")
        return True

    ok = True
    # 1) hot-plug: el mundo exterior enchufa el pendrive
    maquina.conectar_dispositivo()
    print("  [ > ] conector USB: pendrive conectado (evento externo)")
    # 2) la política percibe la presencia y decide montar
    ok &= _atender("montar")
    # 3) el monitor pide una extracción segura
    print("  [ > ] el monitor pide extraer el pendrive de forma segura")
    ok &= _atender("desmontar")
    # 4) el mundo retira un pendrive ya desmontado: limpia
    maquina.desconectar_dispositivo()
    print(f"  {verde_local('[ OK ]')} desconexión limpia: se retiró desmontado")
    # 5) reconexión y extracción FORZADA: la protección del kernel
    maquina.conectar_dispositivo()
    print("  [ > ] el pendrive se vuelve a conectar")
    ok &= _atender("montar")
    print("  [ ! ] extracción FORZADA sin desmontar (el mundo lo retira)")
    maquina.desconectar_dispositivo()
    print(
        "      "
        + pantalla.rojo(
            "el kernel registra el ERROR: extraccion insegura "
            "(y el dato no sincronizado se pierde)"
        )
    )

    # --- Fase 1.5: almacenamiento real — el pendrive recuerda -------
    print()
    print(negrita_local("ALMACENAMIENTO REAL: el pendrive recuerda (Fase 1.5)"))
    print("─" * 66)
    # 6) la sesión pide montar el medio y luego escribir en él
    V = TOKEN_DE_CARACTER["Q"]
    maquina.conectar_dispositivo()
    print("  [ > ] conector USB: pendrive conectado")
    print("  [ > ] solicitud: montar (la IA acepta el medio)")
    ok &= _atender("montar")
    print("  [ > ] solicitud: escribir 3 Q (sobre el medio ya montado)")
    ok &= _atender("escribir", K=3, V=V)
    # 7) extracción segura: desmontar y retirar — el dato VIAJA en el medio
    ok &= _atender("desmontar")
    maquina.desconectar_dispositivo()
    print("  [ > ] el pendrive se retira DESMONTADO (extracción segura)")
    # 8) el mismo pendrive vuelve: la IA lo vuelve a montar y el dato
    #    sigue ahí
    maquina.conectar_dispositivo()
    print("  [ > ] el mismo pendrive vuelve al conector")
    ok &= _atender("montar")
    print("  [ > ] solicitud: leer 3 Q (el valor solo puede venir del medio)")
    ok &= _atender("leer", K=3, V=V)
    if ok:
        print(
            f"  {verde_local('[ OK ]')} EL PENDRIVE RECUERDA: Q sobrevivió "
            "al ciclo retirar-volver (las ranuras viven en el medio)"
        )
    # 9) trazado I/O propio del dispositivo (anillo del propio pendrive)
    print()
    print(pantalla.tenue("  Trazado I/O del dispositivo (anillo propio):"))
    for linea in maquina.panel_trazado_dispositivo():
        if linea:
            print(f"    {linea}")
    print()
    for linea in pantalla.render_panel_registro(maquina.panel_registro()):
        print(linea)
    print()
    print(
        pantalla.tenue(
            "  Montar/desmontar y el trazado de la propia I/O son decisiones"
        )
    )
    print(
        pantalla.tenue(
            "  de la política; el kernel valida cada primitiva, anota el"
        )
    )
    print(
        pantalla.tenue(
            "  ciclo de vida en su registro (dmesg) y el pendrive lleva su"
        )
    )
    print(
        pantalla.tenue(
            "  propio anillo de I/O. La extracción insegura pierde datos."
        )
    )
    return ok


def negrita_local(t):
    from brooder import pantalla

    return pantalla.negrita(t)


def verde_local(t):
    from brooder import pantalla

    return pantalla.verde(t)


def amarillo_local(t):
    from brooder import pantalla

    return pantalla.amarillo(t)


def rojo_local(t):
    # hotfix contrato: faltaba desde el primer commit — cmd_diagnostico
    # lanzaba NameError en cuanto una tarea caía por debajo del 85 %
    # (la rama ✘ nunca se había ejecutado: las evidencias siempre dieron
    # DOMINIO COMPLETO hasta el escenario de imagen antigua).
    from brooder import pantalla

    return pantalla.rojo(t)


# ------------------------------------------------------------------
# diagnostico
# ------------------------------------------------------------------
def cmd_diagnostico(args) -> int:
    from brooder.incubadora import evaluar
    from brooder.nucleo import aviso_contrato, montar_ssd
    from brooder.cerebro import CerebroBrooder
    from brooder import pantalla

    if args.ssd and Path(args.ssd).exists():
        cerebro, _, manifiesto = montar_ssd(args.ssd)
        print(f"Imagen SSD: {args.ssd}")
        # hotfix contrato: explicar por qué DISPOSITIVO va a fallar antes
        # de la evaluación (y no con un [FALLO] mudo).
        aviso = aviso_contrato(cerebro)
        if aviso:
            print(pantalla.amarillo(f"Aviso: {aviso}."))
            print(
                pantalla.amarillo(
                    "  Copia la imagen SSD reentrenada que acompaña al "
                    "parche sobre ssd/brooder.img."
                )
            )
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
    resultados, trazado = evaluar(
        cerebro, list(Tarea), n_solicitudes=args.solicitudes, con_trazado=True
    )
    todo_ok = True
    for tarea, exito in sorted(resultados.items()):
        marca = verde_local("✔") if exito >= 0.85 else rojo_local("✘")
        todo_ok &= exito >= 0.85
        extra = ""
        if tarea in trazado:
            extra = f"   trazado del registro: {trazado[tarea]:.0%}"
        print(f"  {marca} {tarea:10s} {exito:.0%}{extra}")
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
    _asegurar_consola()
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
