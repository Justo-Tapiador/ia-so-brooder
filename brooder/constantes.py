"""
IA-SO Brooder — constantes globales
===================================

Este módulo define el "universo simbólico" compartido por todas las
piezas del sistema: el vocabulario de tokens, las tareas que Brooder
aprende a atender, la tabla de primitivas de hardware y las
dimensiones de observación/acción.

Mantener TODO aquí garantiza que el entrenamiento (incubadora) y la
ejecución (núcleo de arranque) hablen exactamente el mismo idioma.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# ============================================================
# 1. VOCABULARIO DE TOKENS
# ============================================================
# El mundo exterior (teclado, disco, red) se comunica con Brooder
# mediante tokens enteros. Es el "juego de caracteres" del sistema.
#
#   0 .. 9    -> dígitos '0'..'9'
#  10 .. 35   -> letras  'A'..'Z'
#  36         -> '+'      (separador de suma)
#  37         -> '!'      (token ALARMA)
#
# ARG_BUS (38) NO es un token: es un valor especial del espacio de
# ARGUMENTOS que significa "usa lo que hay ahora mismo en el bus de
# datos" (la última lectura realizada). Es el mecanismo mediante el
# cual Brooder mueve datos entre dispositivos sin tocar el hardware.

TOKEN_0 = 0
TOKEN_9 = 9
TOKEN_A = 10
TOKEN_Z = 35
TOKEN_MAS = 36      # '+'
TOKEN_ALARMA = 37   # '!'

N_TOKENS = 38
ARG_BUS = 38
N_ARGUMENTOS = 39   # tokens 0..37 + el argumento especial BUS

CARACTER_DE_TOKEN = (
    {t: chr(ord("0") + t) for t in range(10)}
    | {t: chr(ord("A") + t - 10) for t in range(10, 36)}
    | {TOKEN_MAS: "+", TOKEN_ALARMA: "!"}
)
TOKEN_DE_CARACTER = {v: k for k, v in CARACTER_DE_TOKEN.items()}


def tokens_a_texto(tokens) -> str:
    """Convierte una secuencia de tokens en texto imprimible."""
    return "".join(CARACTER_DE_TOKEN.get(int(t), "?") for t in tokens)


def texto_a_tokens(texto: str):
    """Convierte texto (mayúsculas) en tokens. Ignora espacios."""
    return [TOKEN_DE_CARACTER[c] for c in texto.upper() if c in TOKEN_DE_CARACTER]


# ============================================================
# 2. DISPOSITIVOS
# ============================================================
N_RANURAS_DISCO = 10     # el disco tiene 10 ranuras direccionables (0..9)
N_RANURAS_MEMORIA = 10   # la RAM tiene 10 ranuras direccionables (0..9)
# Fase 1.5: el pendrive es un dispositivo pequeño (8 ranuras, 0..7):
# menos capacidad que el disco interno — es un medio extraíble, no
# un reemplazo del almacenamiento de la máquina.
N_RANURAS_DISPOSITIVO = 8
PANTALLA_MAX = 32        # capacidad máxima de la pantalla (en tokens)

# Presupuesto de ciclos por solicitud. El núcleo corta la atención
# de una solicitud cuando se agota, exactamente igual en
# entrenamiento y en ejecución.
PRESUPUESTO_CICLOS = {
    "ECO": 24,
    "SUMA": 34,
    "GUARDAR": 26,
    "RECORDAR": 26,
    "AVISO": 20,
    # Fase 1: administrar el pendrive es una decisión de 1 ciclo; el
    # presupuesto daba margen para explorar sin ser infinito.
    # Fase 1.5: los modos de ALMACENAMIENTO (escribir/leer datos en
    # el pendrive montado) necesitan el ciclo completo de una tarea
    # de almacenamiento — montar + direccionar + escribir + trazar +
    # leer + mostrar, el mismo arco que GUARDAR —, así que el
    # presupuesto de la tarea DISPOSITIVO se iguala al del disco (los
    # modos de solo montar/desmontar siguen resolviéndose en 1-7
    # ciclos: la presión de -0.01/ciclo premia la eficiencia).
    "DISPOSITIVO": 26,
}


# ============================================================
# 3. TAREAS
# ============================================================
class Tarea(IntEnum):
    """Solicitudes que el "usuario" plantea a la IA-SO.

    La tarea se anuncia en la observación (canal de control del
    cargador, equivalente a una interrupción): Brooder no tiene que
    adivinar qué se le pide, pero SÍ tiene que descubrir qué
    primitivas usar y en qué orden para resolverla.
    """

    ECO = 0        # repetir en pantalla lo tecleado
    SUMA = 1       # sumar dos dígitos con la CPU y mostrar el resultado
    GUARDAR = 2    # almacenar un valor en el disco y recuperarlo
    RECORDAR = 3   # igual que GUARDAR pero en la RAM
    AVISO = 4      # mostrar un carácter y pitar al leer ALARMA
    DISPOSITIVO = 5  # Fase 1: atender el pendrive virtual del conector


N_TAREAS = len(Tarea)
NOMBRES_TAREAS = [t.name for t in Tarea]

# Currículo de la incubadora: cada etapa añade tareas nuevas sin
# abandonar las anteriores (retención). La etapa 5 (Fase 1) añade la
# administración del dispositivo externo: el pendrive se conecta por
# hot-plug y la política debe decidir montarlo/desmontarlo.
CURRICULO = [
    [Tarea.ECO],
    [Tarea.ECO, Tarea.SUMA],
    [Tarea.ECO, Tarea.SUMA, Tarea.GUARDAR, Tarea.RECORDAR],
    [Tarea.ECO, Tarea.SUMA, Tarea.GUARDAR, Tarea.RECORDAR, Tarea.AVISO],
    [Tarea.ECO, Tarea.SUMA, Tarea.GUARDAR, Tarea.RECORDAR, Tarea.AVISO,
     Tarea.DISPOSITIVO],
]

# Alfabeto de entrenamiento: TODO el vocabulario de letras (A..Z).
# El diseño original usaba solo 10 letras para acelerar el
# aprendizaje y confiaba en que las tareas son agnósticas al valor
# (el dato viaja por el bus). La Fase 0.5 demostró que esa apuesta
# se rompe cuando la política aprende comportamientos nuevos: el
# cerebro con trazado fallaba o spameaba con valores fuera de A..J
# (p. ej. 'guardar 3 Z'). Entrenar con las 26 letras cierra el hueco
# de generalización sin coste apreciable.
LETRAS_ENTRENAMIENTO = list(range(TOKEN_A, TOKEN_Z + 1))  # 'A'..'Z'


# ============================================================
# 4. PRIMITIVAS DE HARDWARE
# ============================================================
class Primitiva(IntEnum):
    """Las ÚNICAS operaciones que Brooder puede pedir al hardware.

    La IA no ejecuta nada directamente: emite una *solicitud de
    primitiva* (id + argumento) y el núcleo —código de confianza—
    la valida y la ejecuta contra la máquina. Este es el principio
    de seguridad central del proyecto.
    """

    NADA = 0                 # ciclo en vacío (idle)
    LEER_TECLADO = 1         # bus <- siguiente token del teclado
    MOSTRAR_EN_PANTALLA = 2  # añade un carácter a la pantalla
    CPU_PONER = 3            # acumulador <- valor
    CPU_SUMAR = 4            # acumulador += valor
    CPU_COCIENTE = 5         # bus <- acumulador // 10
    CPU_RESTO = 6            # bus <- acumulador % 10
    LEER_CPU = 7             # bus <- acumulador
    MOVER_CABEZAL_DISCO = 8  # cabezal del disco <- dirección
    LEER_DISCO = 9           # bus <- disco[cabezal]
    ESCRIBIR_DISCO = 10      # disco[cabezal] <- valor
    MOVER_PUNTERO_MEMORIA = 11  # puntero de RAM <- dirección
    LEER_MEMORIA = 12        # bus <- RAM[puntero]
    ESCRIBIR_MEMORIA = 13    # RAM[puntero] <- valor
    REPRODUCIR_AUDIO = 14    # emite un pitido
    USAR_GPU = 15            # refresca el frame compuesto (reservado)
    LEER_RED = 16            # bus <- siguiente paquete de red

    # --- primera macro-primitiva ("syscall") ------------------------
    # Las primitivas 0..16 son operaciones MICRO de un solo ciclo.
    # REGISTRAR_LOG inaugura la familia de macro-primitivas: acciones
    # de nivel de sistema que el agente decide y el núcleo ejecuta
    # como código de confianza. Su valor se añade AL FINAL del enum
    # para no renumerar los ids existentes: los cerebros incubados
    # con el contrato viejo (17 salidas) siguen montando sin cambios
    # —simplemente no pueden emitir esta primitiva hasta reentrenar.
    REGISTRAR_LOG = 17       # añade una entrada al registro del sistema

    # --- Fase 1: ciclo de vida del dispositivo externo ---------------
    # Segunda y tercera macro-primitivas: montar/desmontar el pendrive
    # del conector USB virtual. Igual que REGISTRAR_LOG, se añaden AL
    # FINAL del enum: los cerebros con el contrato viejo (18 salidas)
    # siguen montando (sus cabezas no pueden emitir ids >= 18) hasta
    # reentrenar/migrar el contrato.
    # NOTA: la conexión/desconexión física (hot-plug) NO es una
    # primitiva: es un evento del mundo exterior que el kernel aplica
    # directamente — la IA solo decide qué hacer con lo que hay.
    MONTAR_DISPOSITIVO = 18    # monta el pendrive presente en el conector
    DESMONTAR_DISPOSITIVO = 19  # desmonta limpio (libera el pendrive)

    # --- Fase 1.5: almacenamiento real en el pendrive montado --------
    # El plan de datos del dispositivo, espejo del par de disco/RAM
    # (mover/leer/escribir) pero contra las ranuras del pendrive y
    # SOLO accesible con el dispositivo montado: el kernel rechaza
    # las tres si no hay un pendrive montado en el conector. Igual
    # que las macro-primitivas anteriores, se añaden AL FINAL del
    # enum para no renumerar nada: los cerebros del contrato 24x20
    # siguen montando (compatibilidad de prefijo) hasta migrar.
    MOVER_PUNTERO_DISPOSITIVO = 20  # puntero del pendrive <- dirección (0..7)
    LEER_DISPOSITIVO = 21           # bus <- ranura del pendrive
    ESCRIBIR_DISPOSITIVO = 22       # ranura del pendrive <- valor del bus


N_PRIMITIVAS = len(Primitiva)


@dataclass(frozen=True)
class InfoPrimitiva:
    """Metadatos de una primitiva (para documentación y TUI)."""

    nombre: str          # firma legible, p. ej. "escribir_disco(v)"
    usa_argumento: bool  # si el argumento aporta algo
    descripcion: str
    tipo_argumento: str = "ninguno"  # bus | direccion | libre | ninguno


# Contrato de argumentos que el NÚCLEO valida antes de ejecutar
# (como un syscall real: argumentos de tipo incorrecto -> rechazo):
#   * primitivas de DATOS -> solo ARG_BUS (el bus es el único canal
#     de datos de la máquina; no se puede "mostrar" lo no leído)
#   * primitivas de DIRECCIONAMIENTO -> literal (0..9) o ARG_BUS
#   * audio -> cualquier valor (frecuencia)
PRIMITIVAS_DATO_BUS = {
    Primitiva.MOSTRAR_EN_PANTALLA,
    Primitiva.CPU_PONER,
    Primitiva.CPU_SUMAR,
    Primitiva.ESCRIBIR_DISCO,
    Primitiva.ESCRIBIR_MEMORIA,
    Primitiva.ESCRIBIR_DISPOSITIVO,
}
PRIMITIVAS_DIRECCION = {
    Primitiva.MOVER_CABEZAL_DISCO,
    Primitiva.MOVER_PUNTERO_MEMORIA,
    Primitiva.MOVER_PUNTERO_DISPOSITIVO,
}


TABLA_PRIMITIVAS = {
    Primitiva.NADA: InfoPrimitiva(
        "nada()", False, "Un ciclo en vacío. No consume dispositivos."
    ),
    Primitiva.LEER_TECLADO: InfoPrimitiva(
        "leer_teclado()", False,
        "Extrae el siguiente token del búfer de teclado y lo deposita en el bus de datos.",
    ),
    Primitiva.MOSTRAR_EN_PANTALLA: InfoPrimitiva(
        "mostrar_en_pantalla(bus)", True,
        "Escribe en pantalla el carácter que hay en el bus de datos.", "bus"
    ),
    Primitiva.CPU_PONER: InfoPrimitiva(
        "cpu_poner(bus)", True,
        "Carga en el acumulador el valor que hay en el bus.", "bus"
    ),
    Primitiva.CPU_SUMAR: InfoPrimitiva(
        "cpu_sumar(bus)", True,
        "Suma al acumulador el valor que hay en el bus.", "bus"
    ),
    Primitiva.CPU_COCIENTE: InfoPrimitiva(
        "cpu_cociente()", False,
        "Deposita en el bus la decena del acumulador (acumulador // 10)."
    ),
    Primitiva.CPU_RESTO: InfoPrimitiva(
        "cpu_resto()", False,
        "Deposita en el bus la unidad del acumulador (acumulador % 10)."
    ),
    Primitiva.LEER_CPU: InfoPrimitiva(
        "leer_cpu()", False,
        "Deposita el acumulador en el bus de datos."
    ),
    Primitiva.MOVER_CABEZAL_DISCO: InfoPrimitiva(
        "mover_cabezal_disco(d)", True,
        "Posiciona el cabezal del disco en la ranura d (0..9).", "direccion"
    ),
    Primitiva.LEER_DISCO: InfoPrimitiva(
        "leer_disco()", False,
        "Lee el token bajo el cabezal y lo deposita en el bus."
    ),
    Primitiva.ESCRIBIR_DISCO: InfoPrimitiva(
        "escribir_disco(bus)", True,
        "Escribe en la ranura bajo el cabezal el valor del bus.", "bus"
    ),
    Primitiva.MOVER_PUNTERO_MEMORIA: InfoPrimitiva(
        "mover_puntero_memoria(d)", True,
        "Posiciona el puntero de la RAM en la ranura d (0..9).", "direccion"
    ),
    Primitiva.LEER_MEMORIA: InfoPrimitiva(
        "leer_memoria()", False,
        "Lee el token bajo el puntero de RAM y lo deposita en el bus."
    ),
    Primitiva.ESCRIBIR_MEMORIA: InfoPrimitiva(
        "escribir_memoria(bus)", True,
        "Escribe en la ranura bajo el puntero el valor del bus.", "bus"
    ),
    Primitiva.REPRODUCIR_AUDIO: InfoPrimitiva(
        "reproducir_audio(f)", True,
        "Emite un pitido con la frecuencia f.", "libre"
    ),
    Primitiva.USAR_GPU: InfoPrimitiva(
        "usar_gpu()", False,
        "Compone un frame nuevo: vacía la pantalla para volver a dibujarla."
    ),
    Primitiva.LEER_RED: InfoPrimitiva(
        "leer_red()", False,
        "Lee el siguiente paquete de red (desactivado en v1: devuelve error)."
    ),
    Primitiva.REGISTRAR_LOG: InfoPrimitiva(
        "registrar_log(m)", True,
        "Añade una entrada al registro del sistema (m = id de MENSAJES_LOG). "
        "El panel de registro es la consola del kernel.", "mensaje"
    ),
    Primitiva.MONTAR_DISPOSITIVO: InfoPrimitiva(
        "montar_dispositivo()", False,
        "Monta el pendrive del conector USB. Solo el cerebro decide "
        "cuándo; el kernel valida que haya dispositivo y no esté ya "
        "montado. El montaje queda anotado en el registro (dmesg).",
    ),
    Primitiva.DESMONTAR_DISPOSITIVO: InfoPrimitiva(
        "desmontar_dispositivo()", False,
        "Desmonta el pendrive de forma limpia (extracción segura). "
        "El desmontaje queda anotado en el registro del sistema.",
    ),
    # --- Fase 1.5: plan de datos del pendrive (requiere montaje) -----
    Primitiva.MOVER_PUNTERO_DISPOSITIVO: InfoPrimitiva(
        "mover_puntero_dispositivo(d)", True,
        "Posiciona el puntero del pendrive en la ranura d (0..7). "
        "Requiere dispositivo montado.", "direccion"
    ),
    Primitiva.LEER_DISPOSITIVO: InfoPrimitiva(
        "leer_dispositivo()", False,
        "Lee el token bajo el puntero del pendrive y lo deposita en el "
        "bus. Requiere dispositivo montado. La lectura queda anotada "
        "en el trazado I/O del dispositivo.",
    ),
    Primitiva.ESCRIBIR_DISPOSITIVO: InfoPrimitiva(
        "escribir_dispositivo(bus)", True,
        "Escribe en la ranura del puntero el valor del bus. Requiere "
        "dispositivo montado. La escritura queda anotada en el "
        "trazado I/O del dispositivo.", "bus"
    ),
}


# ------------------------------------------------------------------
# REGISTRO DEL SISTEMA (consola del kernel)
# ------------------------------------------------------------------
# La primera macro-primitiva necesita un vocabulario cerrado de
# mensajes: el agente emite un id, el núcleo ejecuta el registro.
# Mantener el vocabulario en una tabla (y no en cadenas libres) es la
# misma política de seguridad del resto del contrato: nada de lo que
# la IA emite se interpreta como texto arbitrario.
MENSAJES_LOG = (
    ("INFO", "listo"),
    ("INFO", "lectura completada"),
    ("INFO", "escritura completada"),
    ("INFO", "proceso iniciado"),
    ("AVISO", "disco ocupado"),
    ("AVISO", "bus saturado"),
    ("ERROR", "direccion invalida"),
    ("ERROR", "dispositivo no listo"),
    # --- Fase 1: ciclo de vida del pendrive -------------------------
    # El kernel anota automáticamente estas líneas al ejecutar
    # montar/desmontar y al detectar una extracción insegura: son la
    # traza "dmesg" del dispositivo, EMITIDA POR EL KERNEL (no hace
    # falta que la IA las declare; su REGISTRAR_LOG sigue siendo para
    # declarar su propia I/O). Sin acentos, como el resto del
    # vocabulario, por seguridad de codificación en consolas.
    ("INFO", "dispositivo montado"),
    ("INFO", "dispositivo desmontado"),
    ("ERROR", "extraccion insegura"),
)
N_MENSAJES_LOG = len(MENSAJES_LOG)

# Ids canónicos para el TRAZADO de operaciones de almacenamiento
# (Fase 0.5): tras escribir en disco/RAM el mensaje esperado es el 2
# ("escritura completada") y tras leer, el 1 ("lectura completada").
# El entorno de entrenamiento premia exactamente esa correspondencia:
# la política aprendida no solo emite REGISTRAR_LOG, sino que declara
# el evento QUE TOCA. Los ids son posiciones de MENSAJES_LOG.
MENSAJE_LOG_LECTURA = 1
MENSAJE_LOG_ESCRITURA = 2

# Ids del ciclo de vida del pendrive (Fase 1), anotados por el KERNEL
MENSAJE_LOG_DISP_MONTADO = 8
MENSAJE_LOG_DISP_DESMONTADO = 9
MENSAJE_LOG_EXTRACCION_INSEGURA = 10

# El registro es un anillo a lo dmesg: las entradas viejas se pierden
# cuando llega una nueva. Persiste entre solicitudes (es la consola
# del sistema, no un registro volátil del proceso) y solo se vacía
# en un arranque en frío.
REGISTRO_CAPACIDAD = 8      # entradas retenidas en el anillo
REGISTRO_PANEL_LINEAS = 4   # líneas visibles del panel en la TUI

# Fase 1.5: trazado I/O PROPIO del pendrive. Cada leer/escribir sobre
# el dispositivo montado deja una entrada en el anillo del propio
# dispositivo (paso, tipo, ranura, valor): es el "smart-log" del
# medio extraíble, separado del registro del sistema (que solo anota
# el ciclo de vida: montar/desmontar/extracción). Se vacía en frío.
TRAZADO_DISPOSITIVO_CAPACIDAD = 12  # entradas retenidas en el anillo
TRAZADO_DISPOSITIVO_PANEL = 4       # líneas visibles en monitor/demo


def formatear_trazado_dispositivo(entrada: tuple) -> str:
    """Entrada (paso, tipo, ranura, valor) -> línea legible del panel.

    tipo: "lectura" | "escritura". Sin acentos en las etiquetas cortas
    por la misma política de codificación del resto del vocabulario.
    """
    paso, tipo, ranura, valor = entrada
    letra = "E" if tipo == "escritura" else "L"
    flecha = "<-" if tipo == "escritura" else "->"
    ch = CARACTER_DE_TOKEN.get(valor, "?")
    return f"[{paso:04d}] {letra} ranura[{ranura}] {flecha} '{ch}'"


def formatear_registro(paso: int, mensaje: int) -> str:
    """Entrada (paso, id_mensaje) -> línea legible del panel."""
    nivel, texto = MENSAJES_LOG[mensaje]
    return f"[{paso:04d}] {nivel:>5}| {texto}"


# ------------------------------------------------------------------
# MÁSCARAS DE ACCIÓN (contrato de tipos del núcleo)
# ------------------------------------------------------------------
# El espacio de acciones de la política respeta la misma firma de
# tipos que valida el kernel: no se puede PROponer un argumento de
# tipo incorrecto, igual que un programa no puede pasar un puntero
# donde un syscall espera un entero. Esto elimina de raíz el ruido
# de exploración sobre combinaciones que siempre serían rechazadas.
def mascara_argumentos() -> list:
    """Máscara booleana [n_primitivas][n_argumentos]."""
    mascaras = []
    for p in Primitiva:
        tipo = TABLA_PRIMITIVAS[p].tipo_argumento
        m = [False] * N_ARGUMENTOS
        if tipo == "bus":
            m[ARG_BUS] = True
        elif tipo == "direccion":
            for d in range(10):
                m[d] = True
            m[ARG_BUS] = True
        elif tipo == "libre":
            m = [True] * N_ARGUMENTOS
        elif tipo == "mensaje":
            # REGISTRAR_LOG: solo ids de la tabla de mensajes; el resto
            # de tokens siempre sería rechazado por la máquina
            for i in range(N_MENSAJES_LOG):
                m[i] = True
        else:  # "ninguno": el argumento se ignora; una sola opción
            m[0] = True
        mascaras.append(m)
    return mascaras


MASCARA_ARGUMENTOS = mascara_argumentos()


# ============================================================
# 5. DIMENSIONES DE OBSERVACIÓN Y ACCIÓN
# ============================================================
# Vector de percepción (ver brooder/percepcion.py):
#   one-hot de las tareas CLÁSICAS (5) + 16 señales escalares
#   + 5 canales del dispositivo externo (Fase 1: tarea/conectado/
#   montado; Fase 1.5: puntero y escrituras).
#
# La tarea DISPOSITIVO NO entra en el one-hot: usa su canal escalar
# propio AL FINAL del vector (disp_tarea). Motivo: el contrato de
# percepción se EXTIENDE POR EL FINAL — las primeras posiciones son
# bit a bit las del contrato viejo, así que un cerebro incubado con
# OBS_DIM menor percibe exactamente lo que percibía: el núcleo
# recorta la observación a dim_entrada del cerebro montado (ver
# nucleo.NucleoBrooder e incubadora.evaluar). Es la misma política
# de compatibilidad que añadir primitivas AL FINAL del enum.
N_TAREAS_CLASICAS = 5      # ECO..AVISO (la tarea DISPOSITIVO va aparte)
N_CANALES_DISPOSITIVO = 5  # disp_tarea | disp_conectado | disp_montado
#                          # | disp_puntero | disp_escrituras (Fase 1.5)
OBS_DIM = N_TAREAS_CLASICAS + 16 + N_CANALES_DISPOSITIVO

# Espacio de acción factorizado:
#   acción = (primitiva, argumento)
ACCION_DIM_PRIMITIVA = N_PRIMITIVAS
ACCION_DIM_ARGUMENTO = N_ARGUMENTOS

# ============================================================
# 6. IDENTIDAD DEL PROYECTO
# ============================================================
NOMBRE_PROYECTO = "IA-SO Brooder"
VERSION = "0.3.0"
ESLOGAN = "un sistema operativo que se incuba, no se instala"
