"""
Percepción — el vector con el que Brooder "ve" el mundo
=======================================================

Esta función es EL ÚNICO lugar donde se construye la observación.
La usan, sin excepción:

* `brooder.entorno.EntornoBrooder` (entrenamiento en la incubadora)
* `brooder.nucleo.NucleoBrooder` (ejecución en el PC de nacimiento)

Gracias a eso, la política entrenada percibe exactamente lo mismo
que la política desplegada: cero desajuste de distribución.

Diseño del vector (OBS_DIM = 26):

  [0..4]    tarea clásica, one-hot (canal de control del cargador).
            La tarea DISPOSITIVO NO usa el one-hot: su canal vive al
            final (posición 21) para que las 21 primeras posiciones
            sean BIT A BIT el contrato viejo (compatibilidad de
            prefijo: los cerebros con dim_entrada=21 siguen montando;
            el núcleo recorta la observación a su dim_entrada).
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
  [21]      1.0 si la solicitud es de dispositivo (Fase 1)
  [22]      1.0 si hay pendrive en el conector USB
  [23]      1.0 si el pendrive está montado
  [24]      puntero del pendrive / 7 (Fase 1.5: direccionamiento del
            almacenamiento del dispositivo; la ranura se normaliza
            por N_RANURAS_DISPOSITIVO-1, como cabezal/puntero)
  [25]      1.0 si ya se escribió en el pendrive en esta solicitud
            (Fase 1.5: espejo de los canales 19/20 para el medio
            extraíble)
"""
from __future__ import annotations

from brooder.constantes import (
    N_RANURAS_DISPOSITIVO,
    N_TAREAS_CLASICAS,
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
    # one-hot SOLO de las tareas clásicas: la tarea DISPOSITIVO usa su
    # canal escalar propio al final (compatibilidad de prefijo, ver
    # docstring del módulo y constantes.OBS_DIM).
    obs = [0.0] * N_TAREAS_CLASICAS
    if tarea != Tarea.DISPOSITIVO:
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
    # --- Fase 1: canales del dispositivo externo (pendrive) ---------
    obs.append(1.0 if tarea == Tarea.DISPOSITIVO else 0.0)        # [21]
    obs.append(1.0 if instante.dispositivo_conectado else 0.0)   # [22]
    obs.append(1.0 if instante.dispositivo_montado else 0.0)     # [23]
    # --- Fase 1.5: almacenamiento del dispositivo --------------------
    obs.append(
        instante.dispositivo_puntero / (N_RANURAS_DISPOSITIVO - 1)
    )                                                            # [24]
    obs.append(1.0 if instante.escrituras_dispositivo else 0.0)  # [25]
    return obs


def nombre_de_canales() -> list:
    """Nombres legibles de cada dimensión (para depuración/TUI)."""
    nombres = [f"tarea_{t.name.lower()}" for t in list(Tarea)[:N_TAREAS_CLASICAS]]
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
        "disp_tarea",
        "disp_conectado",
        "disp_montado",
        "disp_puntero",
        "disp_escrituras",
    ]
    return nombres
