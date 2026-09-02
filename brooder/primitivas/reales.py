"""
PC Real — la misma máquina, respaldada por tu sistema operativo
================================================================

Modo "instalación real": el disco de Brooder son archivos de verdad
dentro de un directorio sandbox (`./brooder_sandbox/disco/`), el
pendrive persiste en `./brooder_sandbox/pendrive.json`, el audio
suena con la campana del terminal y la GPU refresca el frame
compuesto en tu terminal.

Límites de seguridad (deliberados e innegociables):

* El disco solo puede leer/escribir los archivos `0.tok` .. `9.tok`
  dentro del sandbox. No hay rutas arbitrarias: el "cabezal" es un
  entero 0..9 validado antes de tocar nada.
* El pendrive solo puede leer/escribir `pendrive.json` (sus ranuras
  de datos) dentro del mismo sandbox: la ranura se valida contra
  0..7 antes de tocar nada. Sin rutas, sin nombres de archivo, sin
  sistema de archivos arbitrario — exactamente la misma política
  de confinamiento del disco.
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

from brooder.constantes import (
    N_RANURAS_DISCO,
    N_RANURAS_DISPOSITIVO,
    N_TOKENS,
)
from brooder.primitivas.base import MaquinaBase


class PCReal(MaquinaBase):
    """Máquina con respaldo real y confinado en un sandbox."""

    def __init__(self, raiz_sandbox: str | Path = "brooder_sandbox") -> None:
        super().__init__()
        self.raiz = Path(raiz_sandbox)
        self.dir_disco = self.raiz / "disco"
        self.dir_disco.mkdir(parents=True, exist_ok=True)
        self._montar_disco()
        self._montar_pendrive()

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
    # persistencia del pendrive (Fase 1.5: almacenamiento REAL)
    # --------------------------------------------------
    def _montar_pendrive(self) -> None:
        """Carga el pendrive persistido del sandbox (o lo formatea).

        Un archivo JSON de propiedades fijas (solo ints y listas de
        ints), validado campo a campo: un pendrive.json manipulado
        se ignora y el dispositivo arranca formateado. Igual que el
        disco, la ESCRITURA queda confinada a este único archivo del
        sandbox — el medio extraíble de verdad: lo que se escribe
        aquí sobrevive a apagar la IA-SO y volver a encenderla.
        """
        self.archivo_pendrive = self.raiz / "pendrive.json"
        self._disp_ranuras = [0] * N_RANURAS_DISPOSITIVO
        self._disp_trazado.clear()
        if self.archivo_pendrive.exists():
            try:
                datos = json.loads(
                    self.archivo_pendrive.read_text(encoding="utf-8")
                )
                ranuras = datos.get("ranuras", [])
                if isinstance(ranuras, list):
                    for i, token in enumerate(ranuras[:N_RANURAS_DISPOSITIVO]):
                        token = int(token)
                        self._disp_ranuras[i] = (
                            token if 0 <= token < N_TOKENS else 0
                        )
            except (ValueError, OSError, TypeError):
                pass  # pendrive corrupto: formateado en frío

    def _persistir_pendrive(self) -> bool:
        """ESCRITURA CONFINADA: siempre el mismo archivo del sandbox.

        El trazado I/O NO persiste (es un anillo de observabilidad
        del proceso, no datos del usuario): solo viajan las ranuras.
        """
        try:
            self.archivo_pendrive.write_text(
                json.dumps(
                    {"ranuras": list(self._disp_ranuras)}, ensure_ascii=True
                ),
                encoding="utf-8",
            )
        except OSError:
            return False
        return True

    def _escribir_dispositivo_interno(self, direccion: int, token: int) -> bool:
        self._disp_ranuras[direccion] = token
        if not self._persistir_pendrive():
            return False  # el medio rechaza la escritura (solo-lectura)
        return True

    def desconectar_dispositivo(self) -> bool:
        """La extracción del pendrive real también persiste.

        La extracción INSEGURA formatea las ranuras en memoria (ver
        MaquinaBase): aquí además se sincroniza el borrado con el
        archivo del sandbox — los datos perdidos lo están DE VERDAD.
        """
        estaba_montado = self._disp_montado
        limpia = super().desconectar_dispositivo()
        if estaba_montado:
            self._persistir_pendrive()
        return limpia

    # --------------------------------------------------
    # reinicio: el disco y el pendrive NO se formatean
    # --------------------------------------------------
    def reiniciar(self) -> None:
        # reinicio en caliente: el disco y el pendrive reales NO se
        # formatean: el disco se vuelve a montar tal como quedó y el
        # pendrive (las ranuras) se conservan igual — es hardware.
        estado_disco = list(self._disco)
        estado_pendrive = list(self._disp_ranuras)
        super().reiniciar()
        self._disco = estado_disco
        self._disp_ranuras = estado_pendrive
        # tras __init__ el puntero al archivo del sandbox se recreó
        self._montar_pendrive()
        self._disp_ranuras = estado_pendrive

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
