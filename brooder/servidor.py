"""
Servidor web-emulador — la consola de IA-SO Brooder en el navegador
====================================================================

Un teclado y una pantalla remotos sobre la MISMA máquina:

    navegador ──HTTP/JSON──> ServidorEmulador ──> SesionInteractiva
                                                 (el mismo núcleo y la
                                                  misma imagen SSD que
                                                  usa `brooder arrancar`)

Decisiones de diseño (Fase 2):

* **100 % stdlib** (``http.server`` + ``json`` + ``threading``): cero
  dependencias nuevas, la filosofía del repo intacta.
* **Sesión única**: una consola = una máquina física. El ``lock`` del
  emulador serializa el acceso (la inferencia de torch no se compite).
* **Loopback por defecto** (``127.0.0.1``), coherente con «Red
  desactivada por seguridad»: exponerlo a la LAN es decisión explícita
  del config.json y se anuncia con aviso.
* **Sin ejecución arbitraria**: el endpoint acepta una línea de texto
  que pasa por ``Solicitud.desde_texto`` igual que el teclado real.
  Cero exec, cero shell, cero archivos servidos fuera de la consola.

La API:

    GET  /              -> la consola web (brooder/web/consola.html)
    GET  /api/config    -> perfil, tamaño del monitor, tema, avisos
    GET  /api/estado    -> máquina encendida + panel del monitor
    POST /api/arrancar  -> enciende la IA-SO (POST + banner)
    POST /api/linea     -> atiende una línea (comando o solicitud)
    POST /api/apagar    -> botón de apagado
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from brooder.constantes import VERSION

RUTA_CONSOLA = Path(__file__).parent / "web" / "consola.html"


class ServidorEmulador:
    """El estado del emulador: una sola IA-SO encendida (o ninguna)."""

    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()
        self._sesion = None  # la máquina encendida

    # --------------------------------------------------
    # helpers (todos bajo lock)
    # --------------------------------------------------
    def _respuesta_base(self) -> dict:
        respuesta = {
            "encendida": False,
            "modo": "normal",
            "resumen": None,
            "panel": None,
        }
        sesion = self._sesion
        if sesion is not None and sesion.encendida:
            respuesta["encendida"] = True
            respuesta["modo"] = sesion.modo
            respuesta["resumen"] = sesion.nucleo.estado.resumen()
            respuesta["panel"] = sesion.nucleo.diagnostico()
        return respuesta

    # --------------------------------------------------
    # API
    # --------------------------------------------------
    def arrancar(self) -> dict:
        """Enciende la IA-SO (o rechaza si ya está encendida)."""
        with self._lock:
            if self._sesion is not None and self._sesion.encendida:
                return {
                    "ok": True,
                    "lineas": ["La IA-SO ya está encendida. Escribe :salir "
                               "para apagarla."],
                    **self._respuesta_base(),
                }
            from brooder.sesion import encender

            sesion, lineas = encender(
                ssd=self.config.ssd,
                maquina_real=self.config.maquina_real,
                sandbox=self.config.sandbox,
                rapido=True,       # el navegador anima el POST con su retardo
                capturar=True,     # líneas por JSON, no por stdout
            )
            self._sesion = sesion
            return {"ok": True, "lineas": lineas, **self._respuesta_base()}

    def atender(self, texto: str) -> dict:
        """Atiende una línea de la consola web."""
        with self._lock:
            sesion = self._sesion
            if sesion is None or not sesion.encendida:
                return {
                    "ok": True,
                    "lineas": ["La IA-SO está apagada. Pulsa ARRANCAR para "
                               "encenderla."],
                    **self._respuesta_base(),
                }
            lineas = sesion.atender_linea(texto)
            if not sesion.encendida:
                # :salir o recovery-A: cierre honesto con acta
                lineas = list(lineas) + sesion.apagar()
                self._sesion = None
            return {"ok": True, "lineas": lineas, **self._respuesta_base()}

    def apagar(self) -> dict:
        """Botón de apagado (equivalente a :salir)."""
        with self._lock:
            sesion = self._sesion
            if sesion is None or not sesion.encendida:
                return {"ok": True, "lineas": [], **self._respuesta_base()}
            lineas = ["Apagando IA-SO... hasta pronto."] + sesion.apagar()
            self._sesion = None
            return {"ok": True, "lineas": lineas, **self._respuesta_base()}

    def estado(self) -> dict:
        """Panel del monitor + métricas del cerebro (perfil C)."""
        with self._lock:
            respuesta = self._respuesta_base()
        respuesta["maquina"] = self.config.maquina
        respuesta["version"] = VERSION
        respuesta["perfil"] = self.config.perfil
        sesion = self._sesion
        if self.config.metricas and sesion is not None:
            metricas = sesion.manifiesto.get("metricas", {})
            respuesta["metricas"] = {
                "exito": metricas.get("exito_eval_final", {}),
                "trazado": metricas.get("trazado_eval_final", {}),
            }
        return respuesta

    def config_json(self) -> dict:
        """Lo que la consola necesita para dibujarse."""
        return {
            "perfil": self.config.perfil,
            "nombre_perfil": self.config.nombre_perfil,
            "maquina": self.config.maquina,
            "ssd": str(self.config.ssd),
            "version": VERSION,
            "consola": {
                "columnas": self.config.columnas,
                "filas": self.config.filas,
                "tema": self.config.tema,
                "retardo_post_ms": self.config.retardo_post_ms,
                "panel": self.config.panel,
                "metricas": self.config.metricas,
            },
            "avisos": list(self.config.avisos),
        }


# ------------------------------------------------------------------
# handler HTTP
# ------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    """Rutas del emulador. El emulador vive en self.server.emulador."""

    # silencio en el log de acceso (la consola es la interfaz, no el log)
    def log_message(self, formato, *args):  # pragma: no cover
        pass

    # --------------------------------------------------
    # GET
    # --------------------------------------------------
    def do_GET(self):  # noqa: N802 (nombre impuesto por http.server)
        emulador = self.server.emulador
        if self.path == "/" or self.path.startswith("/index"):
            self._servir_consola()
        elif self.path == "/api/config":
            self._json(emulador.config_json())
        elif self.path == "/api/estado":
            self._json(emulador.estado())
        else:
            self._json({"ok": False, "error": "ruta desconocida"}, codigo=404)

    # --------------------------------------------------
    # POST
    # --------------------------------------------------
    def do_POST(self):  # noqa: N802
        emulador = self.server.emulador
        cuerpo = self._leer_cuerpo()
        if cuerpo is None:
            return
        if self.path == "/api/arrancar":
            self._json(emulador.arrancar())
        elif self.path == "/api/linea":
            texto = cuerpo.get("texto")
            if not isinstance(texto, str):
                self._json(
                    {"ok": False, "error": "falta 'texto' (línea de consola)"},
                    codigo=400,
                )
                return
            self._json(emulador.atender(texto))
        elif self.path == "/api/apagar":
            self._json(emulador.apagar())
        else:
            self._json({"ok": False, "error": "ruta desconocida"}, codigo=404)

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    def _servir_consola(self) -> None:
        try:
            contenido = RUTA_CONSOLA.read_bytes()
        except OSError:
            self._json(
                {"ok": False, "error": "consola.html no encontrada"}, codigo=500
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(contenido)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(contenido)

    def _leer_cuerpo(self) -> dict | None:
        try:
            longitud = int(self.headers.get("Content-Length", "0"))
            crudo = self.rfile.read(longitud) if longitud else b"{}"
            datos = json.loads(crudo.decode("utf-8")) if crudo else {}
            if not isinstance(datos, dict):
                raise ValueError("el cuerpo debe ser un objeto JSON")
            return datos
        except (ValueError, UnicodeDecodeError) as exc:
            self._json({"ok": False, "error": f"cuerpo inválido: {exc}"}, codigo=400)
            return None

    def _json(self, datos: dict, codigo: int = 200) -> None:
        contenido = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(contenido)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(contenido)


# ------------------------------------------------------------------
# arranque del servidor
# ------------------------------------------------------------------
def crear_httpd(config) -> ThreadingHTTPServer:
    """Crea el servidor listo para serve_forever (host/puerto del config)."""
    emulador = ServidorEmulador(config)
    httpd = ThreadingHTTPServer((config.host, config.puerto), _Handler)
    httpd.daemon_threads = True
    httpd.emulador = emulador
    return httpd
