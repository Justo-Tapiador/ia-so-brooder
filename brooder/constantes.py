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


N_TAREAS = len(Tarea)
NOMBRES_TAREAS = [t.name for t in Tarea]

# Currículo de la incubadora: cada etapa añade tareas nuevas sin
# abandonar las anteriores (retención).
CURRICULO = [
    [Tarea.ECO],
    [Tarea.ECO, Tarea.SUMA],
    [Tarea.ECO, Tarea.SUMA, Tarea.GUARDAR, Tarea.RECORDAR],
    [Tarea.ECO, Tarea.SUMA, Tarea.GUARDAR, Tarea.RECORDAR, Tarea.AVISO],
]

# Alfabeto reducido para generalizar más rápido (10 letras).
LETRAS_ENTRENAMIENTO = list(range(TOKEN_A, TOKEN_A + 10))  # 'A'..'J'


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
}
PRIMITIVAS_DIRECCION = {
    Primitiva.MOVER_CABEZAL_DISCO,
    Primitiva.MOVER_PUNTERO_MEMORIA,
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
}


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
        else:  # "ninguno": el argumento se ignora; una sola opción
            m[0] = True
        mascaras.append(m)
    return mascaras


MASCARA_ARGUMENTOS = mascara_argumentos()


# ============================================================
# 5. DIMENSIONES DE OBSERVACIÓN Y ACCIÓN
# ============================================================
# Vector de percepción (ver brooder/percepcion.py):
#   tarea one-hot (5) + 16 señales escalares.
OBS_DIM = N_TAREAS + 16

# Espacio de acción factorizado:
#   acción = (primitiva, argumento)
ACCION_DIM_PRIMITIVA = N_PRIMITIVAS
ACCION_DIM_ARGUMENTO = N_ARGUMENTOS

# ============================================================
# 6. IDENTIDAD DEL PROYECTO
# ============================================================
NOMBRE_PROYECTO = "IA-SO Brooder"
VERSION = "0.1.0"
ESLOGAN = "un sistema operativo que se incuba, no se instala"
