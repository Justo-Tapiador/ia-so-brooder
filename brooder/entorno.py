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
  +0.25   escritura correcta en disco/RAM/pendrive
  +0.15   lectura correcta del dato almacenado
  +0.30   pitido válido (tras leer ALARMA)
  -0.15   escritura en el dispositivo equivocada
  -0.30   pitido prematuro o repetido
  +0.10   trazar con REGISTRAR_LOG la operación de
          almacenamiento correcta, en tareas de almacenamiento
          (GUARDAR/RECORDAR/DISPOSITIVO-escribir/leer;
          ≤ TRAZO_VENTANA ciclos después;
          máximo 1 premio por tipo/solicitud)
  -0.03   REGISTRAR_LOG con un mensaje ya emitido en la
          solicitud (suprime el spam de repeticiones)
  +0.25   montar/desmontar el pendrive en la dirección
          que pide la solicitud (Fase 1, solo en DISPOSITIVO)
  +1.00   solicitud resuelta (condiciones de éxito completas)
  -0.50   solicitud fallida o carácter erróneo en pantalla
          (fallo temprano: la pantalla es solo-append)

El entorno también expone `Oraculo`: la política *ideal* escrita a
mano. Sirve para dos cosas: (1) verificar en los tests que el
entorno es resoluble al 100 %, y (2) comparar qué tan lejos está la
IA del programa perfecto. Desde la Fase 0.5 el oráculo también
traza: registra la escritura y la lectura de almacenamiento justo
después de ejecutarlas, y sirve de referencia del 100 % de trazado.
Desde la Fase 1.5 traza igual la I/O del pendrive montado.
"""
from __future__ import annotations

import random

from brooder.constantes import (
    ARG_BUS,
    LETRAS_ENTRENAMIENTO,
    MENSAJE_LOG_LECTURA,
    MENSAJE_LOG_ESCRITURA,
    N_RANURAS_DISPOSITIVO,
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

# --- trazado del registro (Fase 0.5) ------------------------------
# Trazar es declarar una operación sensible en la consola del kernel
# JUSTO después de ejecutarla. El premio es acotado por diseño: como
# máximo 1 por tipo de operación y solicitud, en una ventana corta.
# Así no se puede "cultivar" recompensa intercalando logs (el
# log-spam además paga su ciclo), y la política óptima sigue siendo
# resolver la solicitud: el trazado es un extra, no un sustituto.
R_TRAZO_VALIDO = 0.10
R_TRAZO_RUIDO = -0.03  # mensaje ya emitido en esta solicitud
TRAZO_VENTANA = 2      # ciclos máx. entre la operación y su traza
# el trazado vive en las tareas de ALMACENAMIENTO: en ECO/SUMA/AVISO
# no hay operaciones que declarar, y permitir el premio ahí crearía
# un señuelo (escribir en RAM por escribir para luego trazarlo) que
# distrae del aprendizaje de la tarea (bug real de la primera
# incubación de la Fase 0.5: SUMA se estancó al 45 %).
# Fase 1.5: la tarea DISPOSITIVO se une a la familia — sus modos de
# almacenamiento (escribir/leer el pendrive) son I/O declarable con
# los mismos mensajes 1/2; en los modos de ciclo de vida (montar/
# desmontar) no hay operaciones trazables, así que no cuentan.
TRAZO_TAREAS = (Tarea.GUARDAR, Tarea.RECORDAR, Tarea.DISPOSITIVO)
# operaciones de almacenamiento que el oráculo/la IA pueden trazar
OPERACIONES_ESCRITURA = (
    Primitiva.ESCRIBIR_DISCO,
    Primitiva.ESCRIBIR_MEMORIA,
    Primitiva.ESCRIBIR_DISPOSITIVO,
)
OPERACIONES_LECTURA = (
    Primitiva.LEER_DISCO,
    Primitiva.LEER_MEMORIA,
    Primitiva.LEER_DISPOSITIVO,
)
_MENSAJE_POR_TIPO = {
    "lectura": MENSAJE_LOG_LECTURA,
    "escritura": MENSAJE_LOG_ESCRITURA,
}

# --- dispositivo externo (Fase 1) ---------------------------------
# Administrar el pendrive se premia SOLO en la tarea DISPOSITIVO: la
# misma lección anti-señuelo del trazado. En el resto de tareas el
# conector está vacío (el hot-plug lo aplica el entorno únicamente en
# las solicitudes de dispositivo) y MONTAR/DESMONTAR fracasan en la
# máquina sin más recompensa que su ciclo: no se puede "cultivar"
# premio enchufando hardware en mitad de un ECO.
R_DISP_OK = 0.25
TAREAS_DISPOSITIVO = (Tarea.DISPOSITIVO,)

# --- variabilidad del conector (v0.4.0, fix OOD) --------------------
# EL AGUJERO: hasta v0.3.0 las clásicas solo existían con el conector
# VACÍO (el hot-plug era exclusivo de DISPOSITIVO), así que el cerebro
# incubado jamás vio ECO/SUMA/GUARDAR/RECORDAR/AVISO con los canales
# [22]-[25] activos: fuera de distribución (OOD). El cerebro publicado
# v0.3.0 resolvía las clásicas al ~99 % con el conector vacío y al
# 45 % con un pendrive montado con datos residuales.
# EL FIX: el conector de las clásicas nace ahora en uno de los tres
# estados legales en campo — vacío (mayoría), conectado sin montar o
# montado con restos de una "sesión anterior" (1-3 ranuras A-Z y el
# cursor donde lo dejó). El pendrive sigue SIN dar recompensa en las
# clásicas (anti-señuelo): solo cambia la distribución de entradas;
# las condiciones de éxito de las clásicas no tocan el dispositivo
# (cirugía limpia: el oráculo sigue resolviendo el 100 %, ver tests).
# En los episodios DISPOSITIVO el comportamiento es el histórico bit
# a bit: el pendrive llega por contrato, no por azar.
ESTADOS_CONECTOR = ("vacio", "conectado", "montado")
ESTADO_CONECTOR_ALEATORIO = "aleatorio"
P_CONECTOR_MONTADO = 0.30    # clásicas que nacen con el pendrive montado
P_CONECTOR_CONECTADO = 0.10  # enchufado pero sin montar (hot-plug a medias)
RANURAS_RESIDUALES_MIN = 1   # datos que "recuerda" el medio reenchufado
RANURAS_RESIDUALES_MAX = 3

# La curiosidad es una recompensa de exploración clásica (count-based
# bonus): probar una primitiva que apenas se ha usado recibe un
# pequeño premio 1/sqrt(n) que se extingue solo. Sin ella, una
# política que ya resuelve ECO/SUMA deja de muestrear escribir_disco
# y jamás descubriría GUARDAR: la recompensa del disco existe, pero
# es inalcanzable sin exploración.


class EntornoBrooder:
    """Entorno estilo Gymnasium sobre la máquina virtual.

    ``estado_conector`` (v0.4.0, fix OOD) decide cómo nace el
    conector USB en las solicitudes CLÁSICAS:

    * ``None``      — comportamiento histórico: siempre vacío.
    * ``"aleatorio"`` — sortea vacío/conectado/montado (60/10/30) con
      datos residuales; el estado de ENTRENAMIENTO del fix.
    * un estado concreto — lo fuerza; es el estado de EVALUACIÓN
      (mide cada distribución por separado).

    Los episodios DISPOSITIVO ignoran este parámetro: el pendrive
    llega por contrato de la solicitud, no por azar.
    """

    def __init__(
        self,
        tareas_activas=None,
        semilla: int | None = None,
        estado_conector: str | None = None,
    ):
        if estado_conector is not None and estado_conector not in (
            ESTADO_CONECTOR_ALEATORIO,
            *ESTADOS_CONECTOR,
        ):
            raise ValueError(
                "estado_conector debe ser None, 'aleatorio' o uno de "
                f"{ESTADOS_CONECTOR}; recibido: {estado_conector!r}"
            )
        self.maquina = PCVirtual()
        self.rng = random.Random(semilla)
        self.tareas_activas = list(tareas_activas or [Tarea.ECO])
        self.estado_conector = estado_conector
        self.solicitud: Solicitud | None = None
        self.ciclos_restantes = 0
        # banderas de moldeado (se resetean por solicitud)
        self._suma_moldeada = False
        self._escritura_moldeada = False
        self._lectura_moldeada = False
        self._pitido_moldeado = False
        self._prefijo_max = 0
        self._direccion_moldeada = False
        # trazado (Fase 0.5): operación pendiente de traza, tipos ya
        # trazados y mensajes ya emitidos en esta solicitud
        self._trazo_pendiente = None        # (tipo, ciclo) o None
        self._tipos_trazados = set()
        self._mensajes_emitidos = set()
        self._ciclo_solicitud = 0
        # dispositivo (Fase 1): premio de administración ya concedido
        self._disp_moldeado = False
        # métricas
        self.resultados = []  # (tarea, exito) por solicitud
        # trazado: (aciertos, oportunidades) por tarea y ventana móvil
        self.trazos_detalle = {}
        self.trazos_recientes = []  # (tarea, trazos, ops) por solicitud
        # curiosidad: conteos de uso por primitiva (con decaimiento)
        self._conteos_novedad = {}
        # contadores de trazado del episodio en curso
        self._ops_episodio = 0
        self._trazos_episodio = 0

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
        # Fase 1: en las solicitudes de dispositivo, el mundo exterior
        # enchufa el pendrive (hot-plug del kernel). En modo "desmontar"
        # lo deja ADEMÁS montado — lo montó la sesión anterior — para que
        # el trabajo de la política sea liberarlo.
        # Fase 1.5: los modos de ALMACENAMIENTO también reciben el medio
        # YA MONTADO (misma convención que desmontar: lo montó la
        # solicitud 'montar' previa — el plano de datos de la Fase 1.5
        # opera sobre el pendrive montado, no re-decide el montaje). En
        # modo "leer" el pendrive llega además CON DATOS (los grabó la
        # sesión anterior): el valor esperado nunca pasa por el teclado,
        # la única fuente es el propio dispositivo.
        if tarea in TAREAS_DISPOSITIVO:
            modo = self.solicitud.datos.get("modo")
            contenido = None
            if modo == "leer":
                contenido = {
                    self.solicitud.datos["K"]: self.solicitud.datos["V"]
                }
            self.maquina.conectar_dispositivo(contenido=contenido)
            if modo != "montar":
                self.maquina.montar_dispositivo()
        elif self.estado_conector is not None:
            # v0.4.0, fix OOD: las clásicas también conocen el
            # conector — el mundo exterior puede haber dejado un
            # pendrive enchufado (o montado, con sus datos) de la
            # sesión anterior. Con estado_conector=None este bloque
            # no se ejecuta NI consume azar: el comportamiento es el
            # histórico bit a bit (mismo flujo del rng, mismos
            # episodios con semilla fija).
            self._enchufar_conector_clasica()
        self.ciclos_restantes = self.solicitud.presupuesto
        self._suma_moldeada = False
        self._escritura_moldeada = False
        self._lectura_moldeada = False
        self._pitido_moldeado = False
        self._prefijo_max = 0
        self._direccion_moldeada = False
        self._trazo_pendiente = None
        self._tipos_trazados = set()
        self._mensajes_emitidos = set()
        self._ciclo_solicitud = 0
        self._ops_episodio = 0
        self._trazos_episodio = 0
        self._disp_moldeado = False
        return self.observar()

    def _enchufar_conector_clasica(self) -> None:
        """Deja el conector en el estado sorteado/forzado (clásicas).

        "montado" — el pendrive de una sesión anterior, reenchufado y
        montado, con 1-3 ranuras de datos A-Z y el cursor donde lo
        dejaron; "conectado" — hot-plug a medias: presente pero sin
        montar (el medio viaja vacío); "vacio" — nada que enchufar.
        """
        estado = self.estado_conector
        if estado == ESTADO_CONECTOR_ALEATORIO:
            dado = self.rng.random()
            if dado < P_CONECTOR_MONTADO:
                estado = "montado"
            elif dado < P_CONECTOR_MONTADO + P_CONECTOR_CONECTADO:
                estado = "conectado"
            else:
                estado = "vacio"
        if estado == "vacio":
            return
        contenido = None
        puntero = None
        if estado == "montado":
            n = self.rng.randint(RANURAS_RESIDUALES_MIN, RANURAS_RESIDUALES_MAX)
            contenido = {
                self.rng.randrange(N_RANURAS_DISPOSITIVO): self.rng.choice(
                    LETRAS_ENTRENAMIENTO
                )
                for _ in range(n)
            }
            # el cursor NO se rebobina: la política debe leer [24]
            puntero = self.rng.randrange(N_RANURAS_DISPOSITIVO)
        self.maquina.conectar_dispositivo(contenido=contenido, puntero=puntero)
        if estado == "montado":
            self.maquina.montar_dispositivo()

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
        ciclo = self._ciclo_solicitud
        self._ciclo_solicitud += 1

        # --- trazado (Fase 0.5): operación sensible recién ejecutada
        # (una escritura/lectura de almacenamiento exitosa queda
        # "pendiente de traza" durante TRAZO_VENTANA ciclos; solo en
        # las tareas de almacenamiento, ver TRAZO_TAREAS)
        if exito_ejecucion and primitiva in OPERACIONES_ESCRITURA:
            if self.solicitud.tarea in TRAZO_TAREAS:
                self._trazo_pendiente = ("escritura", ciclo)
                self._anotar_trazable()
        elif exito_ejecucion and primitiva in OPERACIONES_LECTURA:
            if self.solicitud.tarea in TRAZO_TAREAS:
                self._trazo_pendiente = ("lectura", ciclo)
                self._anotar_trazable()

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

        # Fase 1.5: lo mismo contra el pendrive — posicionar el
        # puntero del dispositivo en la ranura pedida ya es progreso
        if (
            primitiva == Primitiva.MOVER_PUNTERO_DISPOSITIVO
            and not self._direccion_moldeada
        ):
            if (
                sol.tarea in TAREAS_DISPOSITIVO
                and sol.datos.get("modo") in ("escribir", "leer")
                and instante.dispositivo_puntero == sol.datos["K"]
            ):
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

        # Fase 1.5: escritura en el pendrive (modo escribir) — el dato
        # aterriza en la ranura pedida del medio extraíble
        if (
            instante.escrituras_dispositivo
            > instante_antes.escrituras_dispositivo
        ):
            if (
                sol.tarea in TAREAS_DISPOSITIVO
                and sol.datos.get("modo") == "escribir"
                and instante.dispositivo_puntero == sol.datos["K"]
                and instante.dispositivo_ranuras[sol.datos["K"]] == sol.datos["V"]
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

        # Fase 1.5: lectura del pendrive (modo leer) — el valor que
        # solo existía en el medio extraíble llega al bus
        if (
            primitiva == Primitiva.LEER_DISPOSITIVO
            and not self._lectura_moldeada
        ):
            if (
                sol.tarea in TAREAS_DISPOSITIVO
                and sol.datos.get("modo") == "leer"
                and instante.dispositivo_puntero == sol.datos["K"]
                and instante.bus_valor == sol.datos["V"]
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

        # --- moldeado: trazado del registro (Fase 0.5) -------------
        # REGISTRAR_LOG con el mensaje QUE TOCA, en el ciclo que toca:
        # la consola del kernel refleja la operación recién ejecutada.
        # Un mensaje equivocado (o una traza tardía) no castiga: paga
        # su ciclo como cualquier acción no productiva. Repetir un
        # mensaje ya emitido en la solicitud sí penaliza un poco: sin
        # esto, el argmax puede caer en un bucle de log(1) inofensivo
        # pero ruidoso (patología vista en la incubación real).
        if primitiva == Primitiva.REGISTRAR_LOG and exito_ejecucion:
            if argumento in self._mensajes_emitidos:
                recompensa += R_TRAZO_RUIDO
            else:
                self._mensajes_emitidos.add(argumento)
            if self._trazo_pendiente is not None:
                tipo, ciclo_op = self._trazo_pendiente
                if (
                    ciclo - ciclo_op <= TRAZO_VENTANA
                    and argumento == _MENSAJE_POR_TIPO[tipo]
                    and tipo not in self._tipos_trazados
                ):
                    recompensa += R_TRAZO_VALIDO
                    self._tipos_trazados.add(tipo)
                    self._anotar_trazo()
                    # la operación queda declarada: la próxima traza
                    # válida exigirá una operación nueva
                    self._trazo_pendiente = None

        # --- moldeado: dispositivo externo (Fase 1) ------------------
        # montar/desmontar EN LA DIRECCIÓN que pide la solicitud, solo
        # en la tarea DISPOSITIVO (anti-señuelo). La dirección equivoca
        # (p. ej. desmontar cuando se pide montar) fracasa en la
        # máquina: paga su ciclo, sin castigo extra — igual que el
        # resto del moldeado de dispositivos. Fase 1.5: los modos de
        # almacenamiento reciben el medio YA montado (como desmontar),
        # así que su premio vive en el moldeado del plano de datos
        # (dirección/escritura/lectura), no aquí.
        if (
            sol.tarea in TAREAS_DISPOSITIVO
            and exito_ejecucion
            and not self._disp_moldeado
        ):
            modo = sol.datos.get("modo")
            if (
                (modo == "montar" and primitiva == Primitiva.MONTAR_DISPOSITIVO)
                or (modo == "desmontar" and primitiva == Primitiva.DESMONTAR_DISPOSITIVO)
            ):
                recompensa += R_DISP_OK
                self._disp_moldeado = True

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
        self.trazos_recientes.append(
            (self.solicitud.tarea.name, self._trazos_episodio, self._ops_episodio)
        )

    def _anotar_trazable(self) -> None:
        """Cuenta una operación de almacenamiento que podía trazarse."""
        tarea = self.solicitud.tarea.name
        aciertos, oportunidades = self.trazos_detalle.get(tarea, (0, 0))
        self.trazos_detalle[tarea] = (aciertos, oportunidades + 1)
        self._ops_episodio += 1

    def _anotar_trazo(self) -> None:
        """Cuenta una traza correcta (mensaje y momento adecuados)."""
        tarea = self.solicitud.tarea.name
        aciertos, oportunidades = self.trazos_detalle.get(tarea, (0, 0))
        self.trazos_detalle[tarea] = (aciertos + 1, oportunidades)
        self._trazos_episodio += 1

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

    def tasa_trazado(self) -> dict:
        """Trazado correcto por tarea: trazas válidas / operaciones trazables.

        El 100 % significa: cada escritura/lectura de almacenamiento
        quedó declarada en el registro con el mensaje correcto y en
        el momento oportuno. Es la métrica que la Fase 0.5 quiere
        subir de 0 (cerebro viejo: no puede trazar) a la referencia
        del oráculo.
        """
        return {
            tarea: (aciertos / oportunidades, oportunidades)
            for tarea, (aciertos, oportunidades) in self.trazos_detalle.items()
            if oportunidades
        }

    def tasa_trazado_ventana(self, ventana: int = 200) -> float:
        """Trazado global en la ventana móvil de solicitudes.

        Es la medida ESTOCÁSTICA del trazado (la que ve el muestreo de
        entrenamiento, no el argmax). La incubadora la registra como
        métrica de observabilidad; la decisión de explorar/consolidar
        se toma sobre la eval determinista, porque la entropía diluye
        el muestreo y esta ventana subestima lo que el argmax ya sabe
        hacer (lección de la Fase 0.5).
        """
        recientes = self.trazos_recientes[-ventana:]
        ops = sum(o for _, _, o in recientes)
        trazos = sum(t for _, t, _ in recientes)
        return (trazos / ops) if ops else 0.0

    def limpiar_metricas(self) -> None:
        self.resultados = []
        self.trazos_detalle = {}
        self.trazos_recientes = []


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
            # Fase 0.5: el programa ideal declara sus operaciones de
            # almacenamiento en la consola del kernel (trazado)
            p += [(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)]
            p += [(Primitiva.LEER_DISCO, 0), (Primitiva.REGISTRAR_LOG, MENSAJE_LOG_LECTURA)]
            p += [(Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]

        elif s.tarea == Tarea.RECORDAR:
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.MOVER_PUNTERO_MEMORIA, ARG_BUS)]
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.ESCRIBIR_MEMORIA, ARG_BUS)]
            p += [(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)]
            p += [(Primitiva.LEER_MEMORIA, 0), (Primitiva.REGISTRAR_LOG, MENSAJE_LOG_LECTURA)]
            p += [(Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]

        elif s.tarea == Tarea.AVISO:
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]
            p += [(Primitiva.LEER_TECLADO, 0), (Primitiva.REPRODUCIR_AUDIO, ARG_BUS)]

        elif s.tarea == Tarea.DISPOSITIVO:
            # Fase 1: el programa ideal lee el estado del conector en
            # los canales de percepción y decide: pendrive presente y
            # sin montar -> montar; montado -> desmontar (extracción
            # segura). El kernel anota ambas operaciones en su
            # registro (dmesg) sin intervención del oráculo.
            modo = s.datos.get("modo")
            if modo == "desmontar":
                p += [(Primitiva.DESMONTAR_DISPOSITIVO, 0)]
            elif modo == "montar":
                p += [(Primitiva.MONTAR_DISPOSITIVO, 0)]
            elif modo == "escribir":
                # Fase 1.5: el arco de una escritura sobre el medio YA
                # montado (el montaje lo decidió la solicitud 'montar'
                # previa): leer la ranura y el valor del teclado,
                # escribir y declararlo (trazado), leer de vuelta y
                # declararlo, y mostrar el valor recuperado como prueba.
                p += [(Primitiva.LEER_TECLADO, 0),
                      (Primitiva.MOVER_PUNTERO_DISPOSITIVO, ARG_BUS)]
                p += [(Primitiva.LEER_TECLADO, 0),
                      (Primitiva.ESCRIBIR_DISPOSITIVO, ARG_BUS)]
                p += [(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_ESCRITURA)]
                p += [(Primitiva.LEER_DISPOSITIVO, 0),
                      (Primitiva.REGISTRAR_LOG, MENSAJE_LOG_LECTURA)]
                p += [(Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]
            elif modo == "leer":
                # Fase 1.5: el valor vive en el pendrive montado —
                # posicionar la ranura tecleada, leer el medio,
                # declarar la lectura y mostrarla.
                p += [(Primitiva.LEER_TECLADO, 0),
                      (Primitiva.MOVER_PUNTERO_DISPOSITIVO, ARG_BUS)]
                p += [(Primitiva.LEER_DISPOSITIVO, 0)]
                p += [(Primitiva.REGISTRAR_LOG, MENSAJE_LOG_LECTURA)]
                p += [(Primitiva.MOSTRAR_EN_PANTALLA, ARG_BUS)]

        # margen de reposo hasta que el núcleo declare el éxito
        p += [(Primitiva.NADA, 0)] * 6
        return p

    def accion(self):
        return self.programa[min(self.indice, len(self.programa) - 1)]

    def avanzar(self) -> None:
        self.indice += 1
