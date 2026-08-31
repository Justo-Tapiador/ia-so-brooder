"""
Estado de Brooder — la memoria persistente del sistema
======================================================

Registro episódico de lo que la IA-SO ha vivido: solicitudes
atendidas, éxitos, fallos, ciclos consumidos y primitivas usadas.
Viaja DENTRO de la imagen SSD (se restaura al arrancar) y también
se persiste en el sandbox de la máquina real.

Es la evolución del `BrooderState` del prototipo original: ya no es
un diccionario suelto, es un registro estructurado con historial,
estadísticas y versionado.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from brooder.constantes import VERSION


@dataclass
class RegistroSolicitud:
    """Una solicitud atendida por la IA-SO."""

    tarea: str
    descripcion: str
    exito: bool
    ciclos: int
    causa: str
    pantalla_final: str
    primitivas_usadas: int


@dataclass
class EstadoBrooder:
    """Estado persistente del sistema (se guarda y restaura)."""

    version: str = VERSION
    arranques: int = 0
    solicitudes_atendidas: int = 0
    solicitudes_ok: int = 0
    ciclos_totales: int = 0
    primera_vez: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    ultimo_arranque: str = ""
    historial: list = field(default_factory=list)  # últimos RegistroSolicitud

    # --------------------------------------------------
    # registro
    # --------------------------------------------------
    def anotar_arranque(self) -> None:
        self.arranques += 1
        self.ultimo_arranque = time.strftime("%Y-%m-%d %H:%M:%S")

    def anotar_solicitud(self, registro: RegistroSolicitud) -> None:
        self.solicitudes_atendidas += 1
        self.solicitudes_ok += int(registro.exito)
        self.ciclos_totales += registro.ciclos
        self.historial.append(asdict(registro))
        # conservar solo los últimos 50 (el registro completo va al log)
        if len(self.historial) > 50:
            self.historial = self.historial[-50:]

    # --------------------------------------------------
    # persistencia
    # --------------------------------------------------
    def guardar(self, ruta) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def cargar(ruta) -> "EstadoBrooder":
        ruta = Path(ruta)
        if not ruta.exists():
            return EstadoBrooder()
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            estado = EstadoBrooder(**{
                k: v for k, v in datos.items() if k in EstadoBrooder.__dataclass_fields__
            })
            return estado
        except (ValueError, OSError):
            return EstadoBrooder()

    # --------------------------------------------------
    # resumen para el monitor
    # --------------------------------------------------
    def resumen(self) -> str:
        if self.solicitudes_atendidas == 0:
            return "sin solicitudes atendidas todavía"
        tasa = self.solicitudes_ok / self.solicitudes_atendidas
        return (
            f"{self.solicitudes_atendidas} solicitudes atendidas "
            f"({tasa:.0%} resueltas) en {self.arranques} arranques"
        )
