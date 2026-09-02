"""
Sesión interactiva compartida — una sola consola, dos teclados
================================================================

La sesión de IA-SO Brooder solía vivir dentro del bucle de ``cli.py``
(``input``/``print``). El emulador web (Fase 2) necesita atender la
MISMA sesión desde un navegador: mismas solicitudes, mismos veredictos,
misma secuencia de arranque.

Esta módulo extrae la sesión a un objeto reutilizable:

* **CLI** (``brooder arrancar``): imprime las líneas en vivo — su
  comportamiento es byte-idéntico al de la versión clásica (los 137
  tests anteriores lo certifican).
* **Servidor web** (``brooder servidor``): captura las mismas líneas
  (``redirect_stdout``) y las envía por JSON a la consola-web.

Una sola fuente de verdad: no puede existir una consola web «desincro-
nizada» de la real, porque ambas son la MISMA ruta de código.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from brooder.solicitudes import Solicitud


# ------------------------------------------------------------------
# construcción de la sesión (máquina + cerebro + POST + banner)
# ------------------------------------------------------------------
def encender(
    *,
    ssd,
    maquina_real: bool = False,
    sandbox="brooder_sandbox",
    detallado: bool = False,
    rapido: bool = True,
    capturar: bool = False,
):
    """Enciende la IA-SO y devuelve ``(sesion, lineas_arranque)``.

    Con ``capturar=False`` (CLI) todo se imprime en vivo y las líneas
    devueltas van vacías: el flujo visible es idéntico al clásico,
    pausas del POST incluidas. Con ``capturar=True`` (emulador web)
    nada se imprime: las líneas viajan por JSON y el navegador anima
    el POST con su propio retardo (``retardo_post_ms`` del perfil).
    """
    from brooder import pantalla
    from brooder.nucleo import NucleoBrooder, aviso_contrato, montar_ssd
    from brooder.primitivas.reales import PCReal
    from brooder.primitivas.virtual import PCVirtual

    buf = io.StringIO() if capturar else None
    contexto = (
        contextlib.redirect_stdout(buf) if capturar else contextlib.nullcontext()
    )
    with contexto:
        # 1. máquina (virtual por defecto, real con sandbox)
        if maquina_real:
            maquina = PCReal(raiz_sandbox=sandbox)
            tipo_maquina = "REAL (sandbox en disco)"
        else:
            maquina = PCVirtual()
            tipo_maquina = "VIRTUAL"

        # 2. cerebro: desde SSD o en blanco (para ver el arranque sin entrenar)
        estado_sistema = None
        manifiesto = {}
        if ssd and Path(ssd).exists():
            cerebro, estado_sistema, manifiesto = montar_ssd(ssd)
            origen = f"SSD {Path(ssd).name}"
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
        pantalla.splash_bios(nucleo.post(), rapido=rapido)
        # hotfix contrato: el desfase kernel/cerebro se explica ANTES de que
        # el usuario vea [FALLO]s mudos en 'montar'.
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

    sesion = SesionInteractiva(
        nucleo,
        tipo_maquina=tipo_maquina,
        origen_cerebro=origen,
        manifiesto=manifiesto,
        detallado=detallado,
        persistir=maquina_real,
        sandbox=sandbox,
    )
    lineas = buf.getvalue().splitlines() if capturar else []
    return sesion, lineas


# ------------------------------------------------------------------
# la sesión
# ------------------------------------------------------------------
class SesionInteractiva:
    """Una IA-SO encendida: atiende líneas y devuelve sus líneas de consola.

    El CLI imprime lo que devuelven los métodos; el servidor web lo
    envía por JSON. En ambos casos la salida es la misma porque la
    produce el mismo código.
    """

    def __init__(
        self,
        nucleo,
        *,
        tipo_maquina: str = "VIRTUAL",
        origen_cerebro: str = "",
        manifiesto: dict | None = None,
        detallado: bool = False,
        persistir: bool = False,
        sandbox="brooder_sandbox",
    ):
        self.nucleo = nucleo
        self.tipo_maquina = tipo_maquina
        self.origen_cerebro = origen_cerebro
        self.manifiesto = manifiesto or {}
        self.detallado = detallado
        self.persistir = persistir
        self.sandbox = sandbox
        self.encendida = True
        self.modo = "normal"  # "normal" | "recovery"

    # --------------------------------------------------
    # atención de una línea (comando o solicitud)
    # --------------------------------------------------
    def atender_linea(self, linea: str) -> list[str]:
        """Atiende UNA línea y devuelve las líneas que produce.

        El fin de sesión (``:salir``) se anuncia en ``encendida``: el
        llamador (CLI o servidor) debe llamar a ``apagar()`` después.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._procesar(linea)
        return buf.getvalue().splitlines()

    def _procesar(self, linea: str) -> None:
        from brooder import pantalla

        comando = linea.lower()
        if comando in (":salir", ":apagar", "salir", "exit"):
            print("Apagando IA-SO... hasta pronto.")
            self.encendida = False
            return
        if comando == ":ayuda" or comando == "ayuda":
            pantalla.ayuda_interactiva()
            return
        if comando in (":pendrive", "pendrive", "usb"):
            self._toggle_pendrive()
            return
        # modo recovery (consola web): la línea entrante es la opción
        # del menú; se ejecuta y se vuelve al modo normal.
        if self.modo == "recovery":
            self.modo = "normal"
            self.ejecutar_recovery(comando)
            return
        if comando == ":recovery" or comando == "recovery":
            self.modo = "recovery"
            # menú sin input: la SIGUIENTE línea será la opción elegida
            for linea_menu in pantalla.render_menu_recovery():
                print(linea_menu)
            print("  Elige una opción> ", end="")
            return

        solicitud = Solicitud.desde_texto(linea)
        if solicitud is None:
            print(
                pantalla.amarillo(
                    "No entendí la solicitud. Prueba 'HOLA', '3+5', "
                    "'guardar 4 G', 'recordar 2 Z', 'aviso A', 'montar', "
                    "'escribir 3 P', 'leer 3 P' o :ayuda."
                )
            )
            return

        resultado = self.nucleo.atender_solicitud(solicitud)
        pantalla.mostrar_resultado_solicitud(resultado, detallado=self.detallado)

    # --------------------------------------------------
    # recovery (la acción es compartida; el menú, según el teclado)
    # --------------------------------------------------
    def ejecutar_recovery(self, eleccion: str) -> bool:
        """Ejecuta una acción del menú recovery. True si hay que apagar.

        El CLI pasa la opción que leyó con ``input``; la consola web
        pasa la línea siguiente al menú. La acción (y su salida) es
        idéntica en ambos casos.
        """
        from brooder import pantalla

        eleccion = eleccion.strip().upper()
        accion = "estado"
        for tecla, _etiqueta, act in pantalla.MENU_RECOVERY:
            if eleccion == tecla:
                accion = act
                break

        if accion == "reiniciar":
            print("Reiniciando el cerebro de Brooder...")
            self.nucleo.cerebro.eval()
            self.nucleo.estado.anotar_arranque()
            print(pantalla.verde("IA reiniciada."))
        elif accion == "estado":
            diag = self.nucleo.diagnostico()
            print(json.dumps(diag, ensure_ascii=False, indent=2))
        elif accion == "diagnostico":
            diag = self.nucleo.diagnostico()
            for k, v in diag.items():
                print(f"  {k}: {v}")
        elif accion == "modo_seguro":
            print("Modo seguro: la red permanece desactivada (política por defecto).")
        elif accion == "apagar":
            print("Apagando...")
            self.encendida = False
        return not self.encendida

    # --------------------------------------------------
    # apagado (persistencia + acta de cierre)
    # --------------------------------------------------
    def apagar(self) -> list[str]:
        """Persiste el estado (máquina real) y devuelve la acta de cierre.

        Igual que en la Fase 1.5: ``estado.json`` es un acta forense de
        la sesión (write-only); el estado de arranque siempre viene del
        manifiesto congelado dentro de la imagen SSD.
        """
        if self.persistir:
            self.nucleo.estado.guardar(Path(self.sandbox) / "estado.json")
        return [f"Sesión cerrada. {self.nucleo.estado.resumen()}."]

    # --------------------------------------------------
    # hot-plug del pendrive (Fase 1)
    # --------------------------------------------------
    def _toggle_pendrive(self) -> None:
        """Enchufa o retira el pendrive del conector (hot-plug)."""
        from brooder import pantalla

        maquina = self.nucleo.maquina
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
