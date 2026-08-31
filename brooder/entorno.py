"""
Entorno de entrenamiento — la cuna virtual de Brooder
=====================================================

`EntornoBrooder` presenta a la IA-SO miles de solicitudes generadas
aleatoriamente y le devuelve recompensas o penalizaciones según cómo
las atiende. Es el "mundo" durante la incubación.

Diseño de recompensas (denso, para acelerar el aprendizaje):

  -0.01   cada ciclo consumido           (presión de eficiencia)
  +0.08   carácter correcto en pantalla  (progreso verificado)
  +0.15   uso correcto de la CPU (suma alcanzada)
  +0.25   escritura correcta en disco/RAM
  +0.15   lectura correcta del dato almacenado
  +0.30   pitido válido (tras leer ALARMA)
  -0.15   escritura en el dispositivo equivocada
  -0.30   pitido prematuro o repetido
  +1.00   solicitud resuelta (condiciones de éxito completas)
  -0.50   solicitud fallida o carácter erróneo en pantalla
          (fallo temprano: la pantalla es solo-append)

El entorno también expone `Oraculo`: la política *ideal* escrita a
mano. Sirve para dos cosas: (1) verificar en los tests que el
entorno es resoluble al 100 %, y (2) comparar qué tan lejos está la
IA del programa perfecto.
"""
from __future__ import annotations

import random

from brooder.constantes import (
    ARG_BUS,
    TOKEN_ALARMA,
    Primitiva,
    Tarea,
)
from brooder.percepcion import construir_observacion
from brooder.primitivas.virtual import PCVirtual
from brooder.solicitudes import Solicitud

# ------------------------------------------------------------------
# Recompensas (constantes nombradas para poder ajustarlas)
# ------------------------------------------------------------------
R_CICLO = -0.01
R_LECTURA = 0.02       # consumir datos de entrada es intrínsecamente bueno
R_SALIDA = 0.02        # ...y producir salida también (flujo de datos vivo)
R_CARACTER_OK = 0.12
R_CARACTER_MAL = -0.05 # error recuperable: usar_gpu() permite redibujar
R_SUMA_OK = 0.15
R_DIRECCION_OK = 0.06  # posicionar el cabezal/puntero en la ranura pedida
R_ESCRITURA_OK = 0.25
R_LECTURA_ALMACEN_OK = 0.15
R_PITIDO_VALIDO = 0.30
R_PITIDO_MAL = -0.30   # pitido prematuro o duplicado (solo en AVISO)
R_EXITO = 1.00
R_FALLO = -0.50
R_NOVEDAD = 0.02      # curiosidad: probar capacidades poco usadas

# La curiosidad es una recompensa de exploración clásica (count-based
# bonus): probar una primitiva que apenas se ha usado recibe un
# pequeño premio 1/sqrt(n) que se extingue solo. Sin ella, una
# política que ya resuelve ECO/SUMA deja de muestrear escribir_disco
# y jamás descubriría GUARDAR: la recompensa del disco existe, pero
# es inalcanzable sin exploración.


class EntornoBrooder:
    """Entorno estilo Gymnasium sobre la máquina virtual."""

    def __init__(self, tareas_activas=None, semilla: int | None = None):
        self.maquina = PCVirtual()
        self.rng = random.Random(semilla)
        self.tareas_activas = list(tareas_activas or [Tarea.ECO])
        self.solicitud: Solicitud | None = None
        self.ciclos_restantes = 0
        # banderas de moldeado (se resetean por solicitud)
        self._suma_moldeada = False
        self._escritura_moldeada = False
        self._lectura_moldeada = False
        self._pitido_moldeado = False
        self._prefijo_max = 0
        self._direccion_moldeada = False
        # métricas
        self.resultados = []  # (tarea, exito) por solicitud
        # curiosidad: conteos de uso por primitiva (con decaimiento)
        self._conteos_novedad = {}

    # --------------------------------------------------
    # ciclo de vida
    # --------------------------------------------------
    def fijar_tareas(self, tareas) -> None:
        """Cambia el repertorio de tareas (lo usa el currículo)."""
        self.tareas_activas = list(tareas)

    def reiniciar(self) -> list:
        """Comienza una solicitud nueva. Devuelve la observación."""
        tarea = self.rng.choice(self.tareas_activas)
        self.solicitud = Solicitud.aleatoria(tarea, self.rng)
        # cada solicitud nace con la máquina limpia: el disco y la
        # RAM se restauran para que la lectura dependa de la
        # escritura realizada en ESTA solicitud.
        self.maquina.reiniciar()
        self.maquina.escribir_teclado(self.solicitud.tokens)
        self.ciclos_restantes = self.solicitud.presupuesto
        self._suma_moldeada = False
        self._escritura_moldeada = False
        self._lectura_moldeada = False
        self._pitido_moldeado = False
        self._prefijo_max = 0
        self._direccion_moldeada = False
        return self.observar()

    def observar(self) -> list:
        return construir_observacion(
            self.maquina.instante(),
            self.solicitud.tarea,
            self.solicitud,
            self.ciclos_restantes,
        )

    # --------------------------------------------------
    # un ciclo de la IA-SO
    # --------------------------------------------------
    def paso(self, primitiva: Primitiva, argumento: int):
        """Ejecuta la solicitud de primitiva y devuelve
        (observación, recompensa, terminada, info)."""
        assert self.solicitud is not None, "llama a reiniciar() primero"

        instante_antes = self.maquina.instante()
        exito_ejecucion = self.maquina.ejecutar(primitiva, argumento)
        self.maquina.avanzar_paso()
        instante = self.maquina.instante()
        self.ciclos_restantes -= 1

        recompensa = R_CICLO
        terminada = False
        causa = "ciclo"

        # --- curiosidad: bonus de novedad ------------------------
        # el decaimiento deja un pequeño suelo de curiosidad: las
        # capacidades raramente usadas siguen siendo un poco más
        # atractivas que las sobreexplotadas (punto fijo ~200 usos)
        if primitiva != Primitiva.NADA:
            conteo = self._conteos_novedad.get(primitiva, 0.0)
            recompensa += R_NOVEDAD / (1.0 + conteo) ** 0.5
            self._conteos_novedad[primitiva] = conteo * 0.995 + 1.0

        sol = self.solicitud

        # --- moldeado: escritura en pantalla -------------------
        # La pantalla es recuperable: usar_gpu() compone un frame
        # nuevo y se puede redibujar. El moldeado premia el PROGRESO
        # (nuevo máximo prefijo correcto), no la repetición, para
        # que no se pueda "cultivar" recompensa borrando y
        # redibujando lo mismo.
        if len(instante.pantalla) > len(instante_antes.pantalla):
            recompensa += R_SALIDA  # hubo flujo bus -> pantalla
            if sol.pantalla_coincide(instante):
                if len(instante.pantalla) > self._prefijo_max:
                    recompensa += R_CARACTER_OK
                    self._prefijo_max = len(instante.pantalla)
            else:
                recompensa += R_CARACTER_MAL

        # --- moldeado: lectura de entrada -----------------------
        if primitiva == Primitiva.LEER_TECLADO and instante_antes.teclado_hay_datos:
            recompensa += R_LECTURA

        # --- moldeado: CPU --------------------------------------
        if primitiva == Primitiva.CPU_SUMAR and not self._suma_moldeada:
            if sol.tarea == Tarea.SUMA and instante.acumulador == sol.datos["suma"]:
                recompensa += R_SUMA_OK
                self._suma_moldeada = True

        # --- moldeado: direccionamiento de disco/RAM -------------
        # acorta el abismo de descubrimiento de GUARDAR/RECORDAR:
        # leer la clave y posicionar el dispositivo en su ranura
        # ya es progreso verificable.
        if primitiva == Primitiva.MOVER_CABEZAL_DISCO and not self._direccion_moldeada:
            if sol.tarea == Tarea.GUARDAR and instante.disco_cabezal == sol.datos["K"]:
                recompensa += R_DIRECCION_OK
                self._direccion_moldeada = True

        if primitiva == Primitiva.MOVER_PUNTERO_MEMORIA and not self._direccion_moldeada:
            if sol.tarea == Tarea.RECORDAR and instante.memoria_puntero == sol.datos["K"]:
                recompensa += R_DIRECCION_OK
                self._direccion_moldeada = True

        # --- moldeado: disco y RAM ------------------------------
        # Las consecuencias de un dispositivo solo se evalúan en las
        # tareas donde ese dispositivo importa, y solo en positivo:
        # castigar la escritura "equivocada" crea una barrera contra
        # la propia exploración que necesita el aprendizaje (los
        # intentos imperfectos ya pagan su ciclo; el acierto se
        # premia con R_ESCRITURA_OK).
        if instante.escrituras_disco > instante_antes.escrituras_disco:
            if (
                sol.tarea == Tarea.GUARDAR
                and instante.disco_cabezal == sol.datos["K"]
                and instante.disco_contenido[sol.datos["K"]] == sol.datos["V"]
                and not self._escritura_moldeada
            ):
                recompensa += R_ESCRITURA_OK
                self._escritura_moldeada = True

        if instante.escrituras_memoria > instante_antes.escrituras_memoria:
            if (
                sol.tarea == Tarea.RECORDAR
                and instante.memoria_puntero == sol.datos["K"]
                and instante.memoria_contenido[sol.datos["K"]] == sol.datos["V"]
                and not self._escritura_moldeada
            ):
                recompensa += R_ESCRITURA_OK
                self._escritura_moldeada = True

        if primitiva == Primitiva.LEER_DISCO and not self._lectura_moldeada:
            if (
                sol.tarea == Tarea.GUARDAR
                and instante.disco_cabezal == sol.datos["K"]
                and instante.disco_contenido[sol.datos["K"]] == sol.datos["V"]
            ):
                recompensa += R_LECTURA_ALMACEN_OK
                self._lectura_moldeada = True

        if primitiva == Primitiva.LEER_MEMORIA and not self._lectura_moldeada:
            if (
                sol.tarea == Tarea.RECORDAR
                and instante.memoria_puntero == sol.datos["K"]
                and instante.memoria_contenido[sol.datos["K"]] == sol.datos["V"]
            ):
                recompensa += R_LECTURA_ALMACEN_OK
                self._lectura_moldeada = True

        # --- moldeado: audio -------------------------------------
        # (solo en AVISO: ver comentario de disco/RAM)
        if primitiva == Primitiva.REPRODUCIR_AUDIO and sol.tarea == Tarea.AVISO:
            alarma_leida = any(
                tok == TOKEN_ALARMA for _, tok in instante.teclado_leidos
            )
            if alarma_leida and not self._pitido_moldeado:
                recompensa += R_PITIDO_VALIDO
                self._pitido_moldeado = True
            else:
                recompensa += R_PITIDO_MAL

        # --- condiciones terminales -----------------------------
        if sol.exito(instante):
            terminada, causa = True, "exito"
            self._registrar(True)
            return self._cerrar(recompensa + R_EXITO, True, causa)

        if self.ciclos_restantes <= 0:
            terminada, causa = True, "presupuesto_agotado"
            self._registrar(False)
            return self._cerrar(recompensa + R_FALLO, True, causa)

        return self.observar(), recompensa, terminada, {
            "tarea": sol.tarea.name,
            "exito": False,
            "causa": causa,
            "ejecucion": exito_ejecucion,
            "evento": instante.ultimo_evento,
        }

    def _cerrar(self, recompensa: float, terminada: bool, causa: str):
        info = {
            "tarea": self.solicitud.tarea.name,
            "exito": causa == "exito",
            "causa": causa,
            "evento": self.maquina.instante().ultimo_evento,
        }
        return self.observar(), recompensa, terminada, info

    def _registrar(self, exito: bool) -> None:
        self.resultados.append((self.solicitud.tarea.name, exito))

    # --------------------------------------------------
    # métricas
    # --------------------------------------------------
    def tasa_exito(self) -> dict:
        """Tasa de éxito por tarea desde el último reinicio de métricas."""
        conteo: dict = {}
        for tarea, exito in self.resultados:
            aciertos, total = conteo.get(tarea, (0, 0))
            conteo[tarea] = (
                aciertos + int(exito),
                total + 1,
            )
        return {
            tarea: (aciertos / total if total else 0.0, total)
            for tarea, (aciertos, total) in conteo.items()
        }

    def limpiar_metricas(self) -> None:
        self.resultados = []


# ------------------------------------------------------------------
# ORÁCULO: el programa ideal, escrito a mano
# ------------------------------------------------------------------
class Oraculo:
    """Política perfecta de referencia.

    No es la IA: es el "manual de servicio" contra el que se mide.
    Los tests exigen que alcance el 100 % de éxito; si el entorno
    estuviera mal construido, el oráculo fallaría y el test lo
    delataría.
    """

    def __init__(self, solicitud: Solicitud):
        self.solicitud = solicitud
        self.indice = 0
        self.programa = self._compilar(solicitud)

    @staticmethod
    def _compilar(s: Solicitud):
        p = []
        if s.tarea == Tarea.ECO:
            for _ in s.tokens:
                p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]

        elif s.tarea == Tarea.SUMA:
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.CPU_SUMAR, ARG_BUS)]  # a
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.NADA, 0)]             # '+' se descarta
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.CPU_SUMAR, ARG_BUS)]  # b
            if s.datos["suma"] >= 10:
                p += [(Primitiva.CPU_COCIENTE, 0), (Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]
                p += [(Primitiva.CPU_RESTO, 0), (Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]
            else:
                p += [(Primitiva.LEER_CPU, 0), (Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]

        elif s.tarea == Tarea.GUARDAR:
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.MOVER_CABEZAL_DISCO, ARG_BUS)]
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.ESCRIBIR_DISCO, ARG_BUS)]
            p += [(Primitiva.LEER_DISCO, 0), (Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]

        elif s.tarea == Tarea.RECORDAR:
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.MOVER_PUNTERO_MEMORIA, ARG_BUS)]
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.ESCRIBIR_MEMORIA, ARG_BUS)]
            p += [(Primitiva.LEER_MEMORIA, 0), (Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]

        elif s.tarea == Tarea.AVISO:
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.REPRODUCIR_AUDIO, ARG_BUS)]

        # margen de reposo hasta que el núcleo declare el éxito
        p += [(Primitiva.NADA, 0)] * 6
        return p

    def accion(self):
        return self.programa[min(self.indice, len(self.programa) - 1)]

    def avanzar(self) -> None:
        self.indice += 1
