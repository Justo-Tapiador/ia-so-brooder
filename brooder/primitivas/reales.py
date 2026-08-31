"""
PC Real — la misma máquina, respaldada por tu sistema operativo
================================================================

Modo "instalación real": el disco de Brooder son archivos de verdad
dentro de un directorio sandbox (`./brooder_sandbox/disco/`), el
audio suena con la campana del terminal y la GPU refresca el frame
compuesto en tu terminal.

Límites de seguridad (deliberados e innegociables):

* El disco solo puede leer/escribir los archivos `0.tok` .. `9.tok`
  dentro del sandbox. No hay rutas arbitrarias: el "cabezal" es un
  entero 0..9 validado antes de tocar nada.
* No existe primitiva de ejecución de código, de proceso o de
  sistema de archivos arbitrario.
* La red está desactivada: `leer_red()` devuelve error controlado.

La IA-SO nunca ve rutas, nombres de archivo ni funciones del host:
percibe la máquina a través de `InstanteMaquina`, igual que en el
entorno virtual de entrenamiento.
"""
from __future__ import annotations

import json
from pathlib import Path

from brooder.constantes import N_RANURAS_DISCO, N_TOKENS
from brooder.primitivas.base import MaquinaBase


class PCReal(MaquinaBase):
    """Máquina con respaldo real y confinado en un sandbox."""

    def __init__(self, raiz_sandbox: str | Path = "brooder_sandbox") -> None:
        super().__init__()
        self.raiz = Path(raiz_sandbox)
        self.dir_disco = self.raiz / "disco"
        self.dir_disco.mkdir(parents=True, exist_ok=True)
        self._montar_disco()

    # --------------------------------------------------
    # montaje del disco real
    # --------------------------------------------------
    def _montar_disco(self) -> None:
        """Carga el estado persistido del sandbox (o lo formatea)."""
        self._disco = []
        for ranura in range(N_RANURAS_DISCO):
            archivo = self.dir_disco / f"{ranura}.tok"
            if archivo.exists():
                try:
                    datos = json.loads(archivo.read_text(encoding="utf-8"))
                    token = int(datos.get("token", 0))
                    self._disco.append(token if 0 <= token < N_TOKENS else 0)
                    continue
                except (ValueError, OSError):
                    pass
            # ranura nueva o corrupta: formatear a 0
            self._disco.append(0)
            self._persistir_ranura(ranura, 0)

    def _persistir_ranura(self, ranura: int, token: int) -> None:
        """ESCRITURA CONFINADA: siempre dentro de dir_disco."""
        archivo = self.dir_disco / f"{ranura}.tok"
        archivo.write_text(
            json.dumps({"ranura": ranura, "token": token}, ensure_ascii=True),
            encoding="utf-8",
        )

    # --------------------------------------------------
    # persistencia del disco
    # --------------------------------------------------
    def reiniciar(self) -> None:
        # reinicio en caliente: el disco real NO se formatea,
        # se vuelve a montar tal como quedó.
        estado_disco = list(self._disco)
        super().reiniciar()
        self._disco = estado_disco

    def _escribir_disco_interno(self, direccion: int, token: int) -> bool:
        try:
            self._persistir_ranura(direccion, token)
        except OSError:
            return False
        self._disco[direccion] = token
        return True

    # --------------------------------------------------
    # audio real
    # --------------------------------------------------
    def reproducir_audio(self, frecuencia: int) -> bool:
        from brooder.constantes import ARG_BUS

        if frecuencia == ARG_BUS and not self._bus_valido:
            self._error("reproducir_audio: bus vacío")
            return False
        ok = super().reproducir_audio(frecuencia)
        if ok:
            # campana del terminal + descripción en el monitor
            print("\a", end="", flush=True)
        return ok
