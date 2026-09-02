"""
Configuración del emulador web — config.json
=============================================

El emulador (``brooder servidor``) lee ``config.json`` de la raíz del
repo. La validación es HONESTA, con la misma convención que
``pendrive.json``: un campo inválido se sustituye por su valor por
defecto y el cambio se anuncia en ``avisos`` (nunca un crash, nunca un
silencio).

Los perfiles replican la tabla de hardware del README:

    A · PC reciclado          monitor 80x25, POST perezoso (se siente)
    B · Sobremesa equilibrado monitor 100x30, POST corto
    C · Incubadora            monitor 120x32, sin retardo + métricas
    D · SSD viajero           el perfil natural del servidor: cualquier
                              PC con Python 3.9+ lo hospeda y cualquier
                              navegador lo usa (móvil incluido)

El perfil fija el CARÁCTER de la máquina emulada (monitor, cadencia del
POST, panel de métricas), no el rendimiento real: la inferencia corre
donde corra el servidor. La sección ``consola`` puede sobrescribir
campo a campo cualquier valor del perfil.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PERFILES = {
    "A": {
        "nombre": "PC reciclado",
        "columnas": 80,
        "filas": 25,
        "retardo_post_ms": 120,
        "panel": True,
        "metricas": False,
    },
    "B": {
        "nombre": "Sobremesa equilibrado",
        "columnas": 100,
        "filas": 30,
        "retardo_post_ms": 40,
        "panel": True,
        "metricas": False,
    },
    "C": {
        "nombre": "Incubadora",
        "columnas": 120,
        "filas": 32,
        "retardo_post_ms": 0,
        "panel": True,
        "metricas": True,
    },
    "D": {
        "nombre": "SSD viajero",
        "columnas": 100,
        "filas": 30,
        "retardo_post_ms": 60,
        "panel": True,
        "metricas": False,
    },
}

TEMAS = ("cian", "verde", "ambar")

RUTA_CONFIG_DEFECTO = Path("config.json")


@dataclass
class ConfigServidor:
    """Configuración resuelta (perfil + overrides + validación)."""

    perfil: str = "D"
    nombre_perfil: str = PERFILES["D"]["nombre"]
    maquina: str = "virtual"  # "virtual" | "real"
    ssd: Path = Path("ssd/brooder.img")
    sandbox: Path = Path("brooder_sandbox")
    host: str = "127.0.0.1"
    puerto: int = 7800
    columnas: int = 100
    filas: int = 30
    tema: str = "cian"
    retardo_post_ms: int = 60
    panel: bool = True
    metricas: bool = False
    avisos: list = field(default_factory=list)

    @property
    def maquina_real(self) -> bool:
        return self.maquina == "real"


def _clamp(valor, minimo, maximo, defecto, nombre, avisos):
    """Entero en rango o defecto con aviso honesto."""
    try:
        valor = int(valor)
    except (TypeError, ValueError):
        avisos.append(f"config.json: {nombre} inválido ({valor!r}); usado {defecto}")
        return defecto
    if not minimo <= valor <= maximo:
        avisos.append(
            f"config.json: {nombre}={valor} fuera de rango "
            f"({minimo}..{maximo}); usado {defecto}"
        )
        return defecto
    return valor


def cargar_config(ruta=RUTA_CONFIG_DEFECTO) -> ConfigServidor:
    """Carga config.json con validación honesta campo a campo.

    Un archivo inexistente no es error: se usan los valores por defecto
    (perfil D) y se anuncia — el emulador arranca igual.
    """
    ruta = Path(ruta)
    config = ConfigServidor()
    avisos = config.avisos

    datos: dict = {}
    if ruta.exists():
        try:
            texto = ruta.read_text(encoding="utf-8")
            cargado = json.loads(texto)
            if isinstance(cargado, dict):
                datos = cargado
            else:
                avisos.append(
                    f"config.json: la raíz no es un objeto; usados los "
                    f"valores por defecto"
                )
        except (ValueError, OSError) as exc:
            avisos.append(f"config.json: ilegible ({exc}); valores por defecto")
    else:
        avisos.append(
            f"config.json: no existe ({ruta}); usados los valores por defecto "
            f"(perfil D)"
        )

    # --- perfil ---------------------------------------------------
    perfil = datos.get("perfil", config.perfil)
    if perfil in PERFILES:
        config.perfil = perfil
        base = PERFILES[perfil]
        config.nombre_perfil = base["nombre"]
        config.columnas = base["columnas"]
        config.filas = base["filas"]
        config.retardo_post_ms = base["retardo_post_ms"]
        config.panel = base["panel"]
        config.metricas = base["metricas"]
    else:
        avisos.append(
            f"config.json: perfil {perfil!r} desconocido (A/B/C/D); usado D"
        )

    # --- máquina / ssd / sandbox ----------------------------------
    maquina = datos.get("maquina", config.maquina)
    if maquina in ("virtual", "real"):
        config.maquina = maquina
    else:
        avisos.append(
            f"config.json: maquina {maquina!r} inválida (virtual|real); "
            f"usada 'virtual'"
        )

    for campo in ("ssd", "sandbox"):
        valor = datos.get(campo)
        if valor is not None:
            if isinstance(valor, str) and valor.strip():
                setattr(config, campo, Path(valor.strip()))
            else:
                avisos.append(f"config.json: {campo} inválido ({valor!r}); "
                              f"usado {getattr(config, campo)}")

    if config.ssd and not Path(config.ssd).exists():
        avisos.append(
            f"config.json: la imagen SSD '{config.ssd}' no existe; al arrancar "
            f"se usará un cerebro sin entrenar (aviso del POST)"
        )

    # --- red ------------------------------------------------------
    red = datos.get("red", {})
    if not isinstance(red, dict):
        avisos.append("config.json: red no es un objeto; ignorada")
        red = {}
    host = red.get("host")
    if host is not None:
        if isinstance(host, str) and host.strip():
            config.host = host.strip()
        else:
            avisos.append(f"config.json: red.host inválido ({host!r}); "
                          f"usado {config.host}")
    puerto = red.get("puerto")
    if puerto is not None:
        config.puerto = _clamp(puerto, 1, 65535, config.puerto, "red.puerto", avisos)

    # --- consola (overrides del perfil) ----------------------------
    consola = datos.get("consola", {})
    if not isinstance(consola, dict):
        avisos.append("config.json: consola no es un objeto; ignorada")
        consola = {}
    if "columnas" in consola:
        config.columnas = _clamp(
            consola["columnas"], 40, 200, config.columnas, "consola.columnas", avisos
        )
    if "filas" in consola:
        config.filas = _clamp(
            consola["filas"], 12, 60, config.filas, "consola.filas", avisos
        )
    if "retardo_post_ms" in consola:
        config.retardo_post_ms = _clamp(
            consola["retardo_post_ms"], 0, 2000, config.retardo_post_ms,
            "consola.retardo_post_ms", avisos,
        )
    if "tema" in consola:
        tema = consola["tema"]
        if tema in TEMAS:
            config.tema = tema
        else:
            avisos.append(
                f"config.json: consola.tema {tema!r} desconocido "
                f"({'/'.join(TEMAS)}); usado '{config.tema}'"
            )
    if "panel" in consola:
        if isinstance(consola["panel"], bool):
            config.panel = consola["panel"]
        else:
            avisos.append(
                f"config.json: consola.panel debe ser true/false; "
                f"usado {config.panel}"
            )

    return config
