"""
Percepción — el vector con el que Brooder "ve" el mundo
=======================================================

Esta función es EL ÚNICO lugar donde se construye la observación.
La usan, sin excepción:

* `brooder.entorno.EntornoBrooder` (entrenamiento en la incubadora)
* `brooder.nucleo.NucleoBrooder` (ejecución en el PC de nacimiento)

Gracias a eso, la política entrenada percibe exactamente lo mismo
que la política desplegada: cero desajuste de distribución.

Diseño del vector (OBS_DIM = 21):

  [0..4]    tarea actual, one-hot (canal de control del cargador)
  [5]       tokens pendientes en el teclado / 8
  [6]       1.0 si hay datos en el teclado
  [7]       valor del bus de datos / 37
  [8]       1.0 si el bus contiene una lectura válida
  [9]       acumulador de la CPU / 18 (recortado)
  [10]      1.0 si acumulador >= 10 (resultado de dos dígitos)
  [11]      longitud de la pantalla / 16
  [12]      último token mostrado / 37
  [13]      comparador de consola: lo mostrado va bien (prefijo correcto)
  [14]      cabezal del disco / 9
  [15]      puntero de la RAM / 9
  [16]      ciclos restantes de la solicitud / presupuesto máximo
  [17]      1.0 si el último ciclo terminó en error
  [18]      1.0 si ya se emitió algún pitido en esta solicitud
  [19]      1.0 si ya se escribió en el disco en esta solicitud
  [20]      1.0 si ya se escribió en la RAM en esta solicitud
"""
from __future__ import annotations

from brooder.constantes import (
    N_TAREAS,
    Tarea,
)
from brooder.primitivas.base import InstanteMaquina
from brooder.solicitudes import PRESUPUESTO_MAX, Solicitud


def construir_observacion(
    instante: InstanteMaquina,
    tarea: Tarea,
    solicitud: Solicitud,
    ciclos_restantes: int,
) -> list:
    """Fotografía el estado de la máquina como vector de percepción."""
    obs = [0.0] * N_TAREAS
    obs[int(tarea)] = 1.0

    obs.append(min(instante.teclado_pendientes, 8) / 8.0)        # [5]
    obs.append(1.0 if instante.teclado_hay_datos else 0.0)       # [6]
    obs.append(instante.bus_valor / 37.0)                        # [7]
    obs.append(1.0 if instante.bus_valido else 0.0)              # [8]
    obs.append(max(0, min(instante.acumulador, 18)) / 18.0)      # [9]
    obs.append(1.0 if instante.acumulador >= 10 else 0.0)        # [10]
    obs.append(min(len(instante.pantalla), 16) / 16.0)           # [11]
    obs.append(
        (instante.pantalla[-1] / 37.0) if instante.pantalla else 0.0
    )                                                            # [12]
    obs.append(1.0 if solicitud.pantalla_coincide(instante) else 0.0)  # [13]
    obs.append(instante.disco_cabezal / 9.0)                     # [14]
    obs.append(instante.memoria_puntero / 9.0)                   # [15]
    obs.append(max(0, ciclos_restantes) / float(PRESUPUESTO_MAX))  # [16]
    obs.append(1.0 if instante.ultimo_error else 0.0)            # [17]
    obs.append(1.0 if instante.pitidos else 0.0)                 # [18]
    obs.append(1.0 if instante.escrituras_disco else 0.0)        # [19]
    obs.append(1.0 if instante.escrituras_memoria else 0.0)      # [20]
    return obs


def nombre_de_canales() -> list:
    """Nombres legibles de cada dimensión (para depuración/TUI)."""
    nombres = [f"tarea_{t.name.lower()}" for t in Tarea]
    nombres += [
        "teclado_pend",
        "teclado_hay",
        "bus_valor",
        "bus_valido",
        "acumulador",
        "acc_ge_10",
        "pantalla_len",
        "pantalla_ultimo",
        "pantalla_coincide",
        "disco_cabezal",
        "memoria_puntero",
        "ciclos_restantes",
        "ultimo_error",
        "pitido_hecho",
        "escrituras_disco",
        "escrituras_memoria",
    ]
    return nombres
