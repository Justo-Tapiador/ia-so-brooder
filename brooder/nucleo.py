"""
Núcleo de Brooder — el arranque de la IA-SO
===========================================

El núcleo es el componente DE CONFIANZA del sistema. Su
responsabilidad:

1. **POST/BIOS**: comprobar la máquina y montar la imagen SSD.
2. **Atender solicitudes**: para cada solicitud del usuario,
   ejecutar el ciclo  percibir -> decidir -> actuar  de Brooder.
3. **Mediar SIEMPRE**: la IA nunca toca el hardware; el núcleo
   valida cada solicitud de primitiva y la ejecuta contra la
   máquina (contrato de tipos incluido).
4. **Vigilar el presupuesto**: cada solicitud tiene un máximo de
   ciclos; agotarlo cierra la atención con veredicto.
5. **Recovery**: si el cerebro falla (excepción, modelo corrupto),
   el núcleo sigue vivo y ofrece el menú de recuperación — como el
   "sistema nervioso autónomo" de la conversación original.

El ciclo de atención reproduce EXACTAMENTE la semántica del
entorno de entrenamiento (misma percepción, mismos presupuestos,
mismo reinicio de registros por solicitud), así que la política
incubada se comporta igual en el PC de nacimiento.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from brooder.cerebro import CerebroBrooder
from brooder.constantes import (
    N_PRIMITIVAS,
    OBS_DIM,
    TABLA_PRIMITIVAS,
    Tarea,
    Primitiva,
    tokens_a_texto,
)
from brooder.estado import EstadoBrooder, RegistroSolicitud
from brooder.percepcion import construir_observacion
from brooder.primitivas.base import InterfazPrimitivas
from brooder.solicitudes import Solicitud


@dataclass
class ResultadoSolicitud:
    """Veredicto de atender una solicitud."""

    solicitud: Solicitud
    exito: bool
    causa: str
    ciclos: int
    pantalla: str = ""
    eventos: list = field(default_factory=list)
    primitivas_usadas: int = 0
    detalle_dispositivos: str = ""
    # Fase 0.5: ids de mensaje REGISTRAR_LOG que el CEREBRO decidió
    # emitir durante la atención (vacío si no trazó). Es la evidencia
    # de que la política reentrenada usa su consola de kernel.
    trazos: list = field(default_factory=list)


class NucleoBrooder:
    """El "kernel" de confianza que aloja al cerebro de Brooder."""

    def __init__(
        self,
        maquina: InterfazPrimitivas,
        cerebro: CerebroBrooder,
        estado: EstadoBrooder | None = None,
        registro_eventos: bool = True,
    ):
        self.maquina = maquina
        self.cerebro = cerebro
        self.estado = estado or EstadoBrooder()
        self.registro_eventos = registro_eventos
        self.cerebro.eval()

    # --------------------------------------------------
    # POST / BIOS
    # --------------------------------------------------
    def post(self) -> list:
        """Power-On Self Test: comprobaciones de la máquina."""
        instante = self.maquina.instante()
        if instante.dispositivo_montado:
            pendrive = "conectado y montado"
        elif instante.dispositivo_conectado:
            pendrive = "conectado (sin montar)"
        else:
            pendrive = "vacío (hot-plug)"
        comprobaciones = [
            ("CPU (acumulador)", "lista"),
            ("RAM", f"{len(instante.memoria_contenido)} ranuras"),
            ("Disco", f"{len(instante.disco_contenido)} ranuras"),
            ("Pantalla", "conectada"),
            ("Teclado", "conectado"),
            ("Audio", "conectado"),
            ("Red", "desactivada por seguridad"),
            ("Conector USB", pendrive),
        ]
        # hotfix contrato (Fase 1): el POST declara qué contrato habla el
        # cerebro montado. Arrancar un cerebro del contrato viejo es
        # LEGAL (compatibilidad de prefijo: sus tareas clásicas siguen
        # al 100 %), pero el hardware nuevo le es invisible y la sesión
        # interactiva devolvería [FALLO]s sin explicación. Nació de un
        # fallo real en campo: parche aplicado sin copiar la imagen SSD.
        salidas = getattr(self.cerebro, "n_primitivas", N_PRIMITIVAS)
        entradas = getattr(self.cerebro, "dim_entrada", OBS_DIM)
        if aviso_contrato(self.cerebro) is None:
            comprobaciones.append(("Cerebro", f"contrato {entradas}x{salidas}"))
        else:
            comprobaciones.append(
                ("Cerebro", f"contrato {entradas}x{salidas} (imagen antigua)", "aviso")
            )
        return comprobaciones

    # --------------------------------------------------
    # ciclo percibir -> decidir -> actuar
    # --------------------------------------------------
    def atender_solicitud(self, solicitud: Solicitud) -> ResultadoSolicitud:
        """Atiende una solicitud completa con el ciclo de Brooder."""
        # como en el entorno: cada solicitud nace con registros limpios
        self.maquina.reiniciar_registros()
        self.maquina.escribir_teclado(solicitud.tokens)

        ciclos_restantes = solicitud.presupuesto
        eventos = []
        trazos = []
        primitivas_usadas = 0
        causa = "presupuesto_agotado"

        # estado del cerebro nuevo para cada solicitud (idéntico al
        # entrenamiento: cada solicitud es un "proceso" fresco)
        h, M = self.cerebro.estado_inicial()
        # compatibilidad de prefijo (Fase 1): un cerebro montado del
        # contrato viejo conoce solo las primeras dim_entrada
        # posiciones de la observación — que son bit a bit las del
        # contrato con el que se incubó. Las tareas clásicas le
        # llegan intactas; la de dispositivo le es invisible.
        dim_obs = getattr(self.cerebro, "dim_entrada", OBS_DIM)

        while ciclos_restantes > 0:
            instante = self.maquina.instante()
            obs = construir_observacion(
                instante, solicitud.tarea, solicitud, ciclos_restantes
            )
            # (el getattr cubre cerebros de prueba sin el atributo)
            obs_t = torch.tensor(
                obs[:dim_obs], dtype=torch.float32
            )

            try:
                prim, arg, _, h, M = self.cerebro.decidir(
                    obs_t, h, M, determinista=True
                )
            except Exception as exc:  # el cerebro falla -> recovery
                causa = f"fallo_del_cerebro: {exc}"
                eventos.append(("KERNEL", causa))
                break

            self.maquina.ejecutar(prim, arg)
            self.maquina.avanzar_paso()
            ciclos_restantes -= 1
            primitivas_usadas += 1

            # Fase 0.5: contabiliza el trazado decidido por el cerebro
            if prim == Primitiva.REGISTRAR_LOG:
                trazos.append(arg)

            if self.registro_eventos:
                eventos.append(
                    (TABLA_PRIMITIVAS[prim].nombre, self.maquina.instante().ultimo_evento)
                )

            instante = self.maquina.instante()
            if solicitud.exito(instante):
                causa = "exito"
                break

        instante = self.maquina.instante()
        exito = causa == "exito"
        resultado = ResultadoSolicitud(
            solicitud=solicitud,
            exito=exito,
            causa=causa,
            ciclos=solicitud.presupuesto - max(0, ciclos_restantes),
            pantalla=tokens_a_texto(instante.pantalla),
            eventos=eventos,
            primitivas_usadas=primitivas_usadas,
            trazos=trazos,
        )
        self._anotar(resultado, instante)
        return resultado

    def _anotar(self, resultado: ResultadoSolicitud, instante) -> None:
        detalle = []
        if resultado.solicitud.tarea == Tarea.SUMA:
            detalle.append(f"acumulador={instante.acumulador}")
        if resultado.solicitud.tarea in (Tarea.GUARDAR, Tarea.RECORDAR):
            K = resultado.solicitud.datos["K"]
            if resultado.solicitud.tarea == Tarea.GUARDAR:
                valor = instante.disco_contenido[K]
                detalle.append(f"disco[{K}]='{tokens_a_texto([valor])}'")
            else:
                valor = instante.memoria_contenido[K]
                detalle.append(f"ram[{K}]='{tokens_a_texto([valor])}'")
        if resultado.solicitud.tarea == Tarea.AVISO:
            detalle.append(f"pitidos={len(instante.pitidos)}")
        if resultado.solicitud.tarea == Tarea.DISPOSITIVO:
            if instante.dispositivo_montado:
                detalle.append("pendrive=montado")
            elif instante.dispositivo_conectado:
                detalle.append("pendrive=conectado sin montar")
            else:
                detalle.append("pendrive=ausente")
        resultado.detalle_dispositivos = " ".join(detalle)

        self.estado.anotar_solicitud(
            RegistroSolicitud(
                tarea=resultado.solicitud.tarea.name,
                descripcion=resultado.solicitud.descripcion(),
                exito=resultado.exito,
                ciclos=resultado.ciclos,
                causa=resultado.causa,
                pantalla_final=resultado.pantalla,
                primitivas_usadas=resultado.primitivas_usadas,
            )
        )

    # --------------------------------------------------
    # recovery (independiente del cerebro)
    # --------------------------------------------------
    def diagnostico(self) -> dict:
        """Chequeo del sistema sin involucrar a la red neuronal."""
        instante = self.maquina.instante()
        return {
            "estado_sistema": self.estado.resumen(),
            "dispositivos": {
                "acumulador": instante.acumulador,
                "pantalla": tokens_a_texto(instante.pantalla) or "(vacía)",
                "disco": tokens_a_texto(instante.disco_contenido),
                "ram": tokens_a_texto(instante.memoria_contenido),
                "teclado_pendiente": instante.teclado_pendientes,
                "ultimo_error": instante.ultimo_error or "(ninguno)",
            },
            "arranques": self.estado.arranques,
            "version": self.estado.version,
        }


# ------------------------------------------------------------------
# desfase kernel/cerebro (hotfix contrato, Fase 1)
# ------------------------------------------------------------------
def aviso_contrato(cerebro) -> str | None:
    """Describe el desfase entre este kernel y el cerebro montado.

    Un cerebro incubado con el contrato viejo (menos primitivas o
    menos canales de percepción que el kernel actual) arranca sin
    problemas gracias a la compatibilidad de prefijo, pero no puede
    decidir sobre el hardware nuevo: sus cabezas nunca emiten ids
    >= n_primitivas y los canales del dispositivo quedan fuera de su
    ventana de percepción. ``arrancar``, ``demo`` y ``diagnostico``
    usan este aviso para explicar el desfase y su remedio ANTES de
    que el usuario vea [FALLO]s sin causa aparente.
    """
    salidas = getattr(cerebro, "n_primitivas", N_PRIMITIVAS)
    entradas = getattr(cerebro, "dim_entrada", OBS_DIM)
    faltan = [p.name for p in Primitiva if int(p) >= salidas]
    if not faltan and entradas >= OBS_DIM:
        return None
    partes = []
    if faltan:
        partes.append(
            f"no puede emitir {', '.join(faltan)} "
            f"(primitivas {salidas}/{N_PRIMITIVAS})"
        )
    if entradas < OBS_DIM:
        partes.append(
            f"no percibe los canales del dispositivo "
            f"(observación {entradas}/{OBS_DIM})"
        )
    return "el cerebro montado habla un contrato viejo: " + " y ".join(partes)


# ------------------------------------------------------------------
# carga de la imagen SSD
# ------------------------------------------------------------------
def montar_ssd(ruta_ssd: str | Path):
    """Monta una imagen SSD exportada y devuelve (cerebro, estado).

    La imagen es un ZIP con:
      * brooder.pt    -> pesos del cerebro (config incluida)
      * manifiesto.json -> métricas de incubación y estado del sistema
    """
    import io
    import json
    from zipfile import ZipFile

    ruta_ssd = Path(ruta_ssd)
    if not ruta_ssd.exists():
        raise FileNotFoundError(
            f"No se encuentra la imagen SSD: {ruta_ssd}. "
            "Ejecuta 'brooder exportar' o incuba un modelo primero."
        )

    with ZipFile(ruta_ssd) as imagen:
        nombres = imagen.namelist()
        if "brooder.pt" not in nombres:
            raise ValueError("La imagen SSD no contiene brooder.pt")
        with imagen.open("brooder.pt") as f:
            # weights_only=True: la imagen SSD viaja por redes y discos
            # no confiables. Un .pt es un pickle: cargado sin
            # restricciones ejecutaría cualquier código embebido en el
            # PC de nacimiento. En modo restringido solo se aceptan
            # tensores y tipos básicos; una imagen manipulada puede,
            # como mucho, fallar al cargar.
            paquete = torch.load(
                io.BytesIO(f.read()), map_location="cpu", weights_only=True
            )
        cerebro = CerebroBrooder(**paquete["config"])
        cerebro.load_state_dict(paquete["estado"])
        cerebro.eval()

        manifiesto = {}
        estado = EstadoBrooder()
        if "manifiesto.json" in nombres:
            with imagen.open("manifiesto.json") as f:
                manifiesto = json.loads(f.read().decode("utf-8"))
            datos_estado = manifiesto.get("estado", {})
            campos = set(EstadoBrooder.__dataclass_fields__)
            estado = EstadoBrooder(
                **{k: v for k, v in datos_estado.items() if k in campos}
            )
    return cerebro, estado, manifiesto


# ------------------------------------------------------------------
# exportación de la imagen SSD
# ------------------------------------------------------------------
def exportar_ssd(
    ruta_modelo: str | Path,
    ruta_salida: str | Path,
    metricas: dict | None = None,
) -> Path:
    """Empaqueta el modelo entrenado como imagen SSD (ZIP)."""
    import json
    from zipfile import ZipFile, ZIP_DEFLATED

    ruta_modelo = Path(ruta_modelo)
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    # weights_only=True también al leer: la misma política de carga en
    # todos los puntos de entrada deja un único patrón que auditar.
    paquete = torch.load(ruta_modelo, map_location="cpu", weights_only=True)
    # La imagen SSD solo empaqueta lo que el PC de nacimiento necesita:
    # config y pesos. El estado del optimizador NO viaja (imagen más
    # ligera y sin datos de entrenamiento innecesarios en el "SSD").
    pesos = {
        "config": paquete["config"],
        "estado": paquete["estado"],
    }
    manifiesto = {
        "proyecto": "IA-SO Brooder",
        "descripcion": (
            "Cerebro incubado de la IA-SO Brooder. Arranca con: "
            "brooder arrancar --ssd <imagen>"
        ),
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pasos_entrenamiento": paquete.get("paso", None),
        "etapa_curriculo": paquete.get("etapa", None),
        "metricas": metricas or {},
        "config_cerebro": paquete.get("config", {}),
    }

    temporal = ruta_salida.with_suffix(".tmp")
    with ZipFile(temporal, "w", compression=ZIP_DEFLATED) as imagen:
        imagen.writestr(
            "brooder.pt",
            _torch_a_bytes(pesos),
        )
        imagen.writestr(
            "manifiesto.json",
            json.dumps(manifiesto, ensure_ascii=False, indent=2),
        )
    temporal.replace(ruta_salida)
    return ruta_salida


def _torch_a_bytes(paquete: dict) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(paquete, buffer)
    return buffer.getvalue()
