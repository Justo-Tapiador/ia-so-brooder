"""
Primitivas de hardware — contrato base
======================================

La idea central de seguridad de IA-SO Brooder:

    La IA NUNCA tiene acceso directo al hardware.
    Solo puede *solicitar* la ejecución de primitivas.

Toda máquina que quiera alojar a Brooder (la simulada por defecto o
la que opera sobre el host real, en modo sandbox) debe implementar
`InterfazPrimitivas`. El núcleo (código de confianza) media SIEMPRE
entre la decisión de la IA y la máquina.

La IA percibe la máquina únicamente a través de `InstanteMaquina`,
una fotografía inocua de su estado: sin punteros, sin rutas de
archivo, sin capacidad de ejecutar nada.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from brooder.constantes import (
    ARG_BUS,
    MENSAJES_LOG,
    MENSAJE_LOG_DISP_DESMONTADO,
    MENSAJE_LOG_DISP_MONTADO,
    MENSAJE_LOG_EXTRACCION_INSEGURA,
    N_RANURAS_DISCO,
    N_RANURAS_MEMORIA,
    N_TOKENS,
    PANTALLA_MAX,
    PRIMITIVAS_DATO_BUS,
    Primitiva,
    REGISTRO_CAPACIDAD,
    REGISTRO_PANEL_LINEAS,
    TABLA_PRIMITIVAS,
    formatear_registro,
)


@dataclass
class InstanteMaquina:
    """Fotografía del estado de la máquina visible para la IA.

    Contiene solo valores "de panel": no expone estructuras internas
    modificables ni nada que permita a la red escapar del contrato.
    """

    # teclado
    teclado_pendientes: int = 0          # tokens sin leer
    teclado_hay_datos: bool = False
    # bus de datos (resultado de la última lectura)
    bus_valor: int = 0
    bus_valido: bool = False
    # cpu
    acumulador: int = 0
    # pantalla
    pantalla: list = field(default_factory=list)   # tokens mostrados
    # disco
    disco_cabezal: int = 0
    disco_contenido: tuple = field(default_factory=tuple)  # solo para el monitor del sistema
    # ram
    memoria_puntero: int = 0
    memoria_contenido: tuple = field(default_factory=tuple)
    # audio
    pitidos: tuple = field(default_factory=tuple)   # (paso, frecuencia)
    # red
    red_paquetes: int = 0
    # registro del sistema (macro-primitiva REGISTRAR_LOG)
    # entradas (paso, id_mensaje) del anillo del kernel
    registro: tuple = field(default_factory=tuple)
    # dispositivo externo (Fase 1: pendrive virtual del conector USB)
    # conectado = presencia física (hot-plug externo); montado =
    # aceptado por decisión de la política con la macro-primitiva
    dispositivo_conectado: bool = False
    dispositivo_montado: bool = False
    # señalización del último ciclo
    ultimo_error: str = ""
    ultimo_evento: str = ""           # descripción legible del último ciclo
    # contadores por solicitud (los reinicia el núcleo)
    escrituras_disco: int = 0
    escrituras_memoria: int = 0
    usos_gpu: int = 0
    # tokens leídos del teclado durante la solicitud actual,
    # con el paso en que se leyeron (para evaluar AVISO)
    teclado_leidos: tuple = field(default_factory=tuple)  # (paso, token)

    @property
    def pantalla_libre(self) -> int:
        return max(0, PANTALLA_MAX - len(self.pantalla))


class InterfazPrimitivas(ABC):
    """Contrato que toda máquina debe cumplir para alojar a Brooder.

    Cada método corresponde a una primitiva del mismo nombre (ver
    `brooder.constantes.Primitiva`). Todos devuelven `bool`:
    True si la operación se ejecutó, False si fue rechazada
    (por validación, búfer vacío, dirección inválida...).
    """

    # --------------------------------------------------
    # ciclo de vida
    # --------------------------------------------------
    @abstractmethod
    def reiniciar(self) -> None:
        """Reinicio completo (como un arranque en frío)."""

    @abstractmethod
    def reiniciar_registros(self) -> None:
        """Limpia registros volátiles entre solicitudes.

        El núcleo llama a este método al comenzar cada solicitud:
        pantalla, bus, acumulador y contadores vuelven a cero.
        El disco y la RAM NO se limpian (son persistencia).
        """

    # --------------------------------------------------
    # entrada
    # --------------------------------------------------
    @abstractmethod
    def escribir_teclado(self, tokens) -> None:
        """El *entorno* (no la IA) deposita tokens en el teclado."""

    @abstractmethod
    def leer_teclado(self) -> bool:
        """bus <- siguiente token del teclado."""

    # --------------------------------------------------
    # salida
    # --------------------------------------------------
    @abstractmethod
    def mostrar_en_pantalla(self, valor: int) -> bool:
        """Añade un token a la pantalla (literal o ARG_BUS)."""

    # --------------------------------------------------
    # cpu
    # --------------------------------------------------
    @abstractmethod
    def cpu_poner(self, valor: int) -> bool: ...

    @abstractmethod
    def cpu_sumar(self, valor: int) -> bool: ...

    @abstractmethod
    def cpu_cociente(self) -> bool: ...

    @abstractmethod
    def cpu_resto(self) -> bool: ...

    @abstractmethod
    def leer_cpu(self) -> bool: ...

    # --------------------------------------------------
    # disco
    # --------------------------------------------------
    @abstractmethod
    def mover_cabezal_disco(self, direccion: int) -> bool: ...

    @abstractmethod
    def leer_disco(self) -> bool: ...

    @abstractmethod
    def escribir_disco(self, valor: int) -> bool: ...

    # --------------------------------------------------
    # memoria
    # --------------------------------------------------
    @abstractmethod
    def mover_puntero_memoria(self, direccion: int) -> bool: ...

    @abstractmethod
    def leer_memoria(self) -> bool: ...

    @abstractmethod
    def escribir_memoria(self, valor: int) -> bool: ...

    # --------------------------------------------------
    # audio / gpu / red
    # --------------------------------------------------
    @abstractmethod
    def reproducir_audio(self, frecuencia: int) -> bool: ...

    @abstractmethod
    def usar_gpu(self) -> bool: ...

    @abstractmethod
    def leer_red(self) -> bool: ...

    # --------------------------------------------------
    # registro del sistema (macro-primitivas)
    # --------------------------------------------------
    @abstractmethod
    def registrar_log(self, mensaje: int) -> bool: ...

    @abstractmethod
    def panel_registro(self) -> list: ...

    # --------------------------------------------------
    # dispositivo externo (Fase 1: pendrive virtual)
    # --------------------------------------------------
    @abstractmethod
    def montar_dispositivo(self) -> bool: ...

    @abstractmethod
    def desmontar_dispositivo(self) -> bool: ...

    # El hot-plug (conectar/desconectar) NO es una primitiva: es un
    # evento del mundo exterior. Lo aplica el kernel directamente
    # (demo, entorno de entrenamiento, :pendrive interactivo) — la IA
    # no puede enchufar ni desenchufar hardware, solo administrarlo.

    # --------------------------------------------------
    # observación
    # --------------------------------------------------
    @abstractmethod
    def instante(self) -> InstanteMaquina:
        """Fotografía segura del estado para la percepción de la IA."""

    # --------------------------------------------------
    # despacho (NO lo llama nunca la IA directamente:
    # lo usa el núcleo, que ya ha validado la acción)
    # --------------------------------------------------
    def ejecutar(self, primitiva: Primitiva, argumento: int) -> bool:
        """Ejecuta la primitiva solicitada con el argumento dado.

        `argumento` es un token literal (0..37) o ARG_BUS (38), que
        la implementación resuelve contra el bus de datos.
        """
        v = argumento
        if primitiva == Primitiva.NADA:
            return True
        if primitiva == Primitiva.LEER_TECLADO:
            return self.leer_teclado()
        if primitiva == Primitiva.MOSTRAR_EN_PANTALLA:
            return self.mostrar_en_pantalla(v)
        if primitiva == Primitiva.CPU_PONER:
            return self.cpu_poner(v)
        if primitiva == Primitiva.CPU_SUMAR:
            return self.cpu_sumar(v)
        if primitiva == Primitiva.CPU_COCIENTE:
            return self.cpu_cociente()
        if primitiva == Primitiva.CPU_RESTO:
            return self.cpu_resto()
        if primitiva == Primitiva.LEER_CPU:
            return self.leer_cpu()
        if primitiva == Primitiva.MOVER_CABEZAL_DISCO:
            return self.mover_cabezal_disco(v)
        if primitiva == Primitiva.LEER_DISCO:
            return self.leer_disco()
        if primitiva == Primitiva.ESCRIBIR_DISCO:
            return self.escribir_disco(v)
        if primitiva == Primitiva.MOVER_PUNTERO_MEMORIA:
            return self.mover_puntero_memoria(v)
        if primitiva == Primitiva.LEER_MEMORIA:
            return self.leer_memoria()
        if primitiva == Primitiva.ESCRIBIR_MEMORIA:
            return self.escribir_memoria(v)
        if primitiva == Primitiva.REPRODUCIR_AUDIO:
            return self.reproducir_audio(v)
        if primitiva == Primitiva.USAR_GPU:
            return self.usar_gpu()
        if primitiva == Primitiva.LEER_RED:
            return self.leer_red()
        if primitiva == Primitiva.REGISTRAR_LOG:
            return self.registrar_log(v)
        if primitiva == Primitiva.MONTAR_DISPOSITIVO:
            return self.montar_dispositivo()
        if primitiva == Primitiva.DESMONTAR_DISPOSITIVO:
            return self.desmontar_dispositivo()
        raise ValueError(f"Primitiva desconocida: {primitiva!r}")


class MaquinaBase(InterfazPrimitivas):
    """Implementación compartida de la lógica de registros.

    PCVirtual y PCReal heredan de aquí: la semántica de las
    primitivas es idéntica; solo cambia el respaldo del disco
    (memoria simulada frente a archivos reales en un sandbox).
    """

    # --------------------------------------------------
    # VALIDACIÓN DE ARGUMENTOS (contrato del núcleo)
    # --------------------------------------------------
    def ejecutar(self, primitiva: Primitiva, argumento: int) -> bool:
        """Valida el TIPO del argumento y despacha.

        Como un syscall de un kernel real: las primitivas de datos
        (mostrar, escribir, cpu_poner/sumar) exigen el BUS como
        fuente del valor —no se puede escribir lo que no se ha
        leído—; las de direccionamiento aceptan literal (0..9) o
        BUS; audio acepta cualquier valor.

        Un argumento de tipo incorrecto se rechaza con error
        controlado (la IA lo percibe en su observación).
        """
        if primitiva in PRIMITIVAS_DATO_BUS and argumento != ARG_BUS:
            self._error(
                f"{TABLA_PRIMITIVAS[primitiva].nombre}: exige BUS "
                f"(las primitivas de datos toman el valor del bus)"
            )
            return False
        return super().ejecutar(primitiva, argumento)

    # --------------------------------------------------
    # utilidades comunes
    # --------------------------------------------------
    @staticmethod
    def _resolver(valor: int, bus_valor: int) -> int:
        """Resuelve ARG_BUS contra el bus de datos."""
        if valor == ARG_BUS:
            return bus_valor
        return valor

    @staticmethod
    def _es_direccion_valida(direccion: int) -> bool:
        return 0 <= direccion < N_RANURAS_DISCO

    @staticmethod
    def _es_token_valido(token: int) -> bool:
        return 0 <= token < N_TOKENS

    def __init__(self) -> None:
        self._teclado: list = []
        self._bus_valor = 0
        self._bus_valido = False
        self._acumulador = 0
        self._pantalla: list = []
        self._disco: list = [0] * N_RANURAS_DISCO
        self._disco_cabezal = 0
        self._memoria: list = [0] * N_RANURAS_MEMORIA
        self._memoria_puntero = 0
        self._pitidos: list = []
        self._red_paquetes: list = []
        # registro del sistema: anillo del kernel (se vacía solo en
        # arranque en frío; persiste entre solicitudes, como dmesg)
        self._registro: deque = deque(maxlen=REGISTRO_CAPACIDAD)
        # dispositivo externo (Fase 1): el conector USB arranca vacío.
        # El estado persiste entre solicitudes (es hardware enchufado,
        # no estado del proceso) y se pierde solo en arranque en frío.
        self._disp_conectado: bool = False
        self._disp_montado: bool = False
        self._paso = 0
        self._ultimo_error = ""
        self._ultimo_evento = ""
        self._escrituras_disco = 0
        self._escrituras_memoria = 0
        self._usos_gpu = 0
        self._teclado_leidos: list = []
        # disco/RAM no se reinician entre solicitudes: se conservan
        # deliberadamente para que GUARDAR/RECORDAR tengan sentido.

    # --------------------------------------------------
    # ciclo de vida
    # --------------------------------------------------
    def reiniciar(self) -> None:
        self.__init__()  # type: ignore[misc]

    def reiniciar_registros(self) -> None:
        self._bus_valor = 0
        self._bus_valido = False
        self._acumulador = 0
        self._pantalla = []
        self._disco_cabezal = 0
        self._memoria_puntero = 0
        self._pitidos = []
        self._paso = 0
        self._ultimo_error = ""
        self._ultimo_evento = "registros limpios"
        self._escrituras_disco = 0
        self._escrituras_memoria = 0
        self._usos_gpu = 0
        self._teclado_leidos = []
        # el kernel descarta la entrada pendiente de la solicitud
        # anterior: cada solicitud empieza con el teclado limpio
        # (igual que en el entorno de entrenamiento)
        self._teclado = []

    # --------------------------------------------------
    # entrada
    # --------------------------------------------------
    def escribir_teclado(self, tokens) -> None:
        self._teclado.extend(int(t) for t in tokens)

    def leer_teclado(self) -> bool:
        if not self._teclado:
            self._error("leer_teclado: búfer vacío")
            return False
        token = self._teclado.pop(0)
        self._bus_valor = token
        self._bus_valido = True
        self._teclado_leidos.append((self._paso, token))
        self._evento(f"leer_teclado -> {token}")
        return True

    # --------------------------------------------------
    # salida
    # --------------------------------------------------
    def mostrar_en_pantalla(self, valor: int) -> bool:
        from brooder.constantes import ARG_BUS, PANTALLA_MAX

        if valor == ARG_BUS and not self._bus_valido:
            self._error("mostrar_en_pantalla: bus vacío")
            return False
        token = self._resolver(valor, self._bus_valor)
        if not self._es_token_valido(token):
            self._error("mostrar_en_pantalla: token inválido")
            return False
        if len(self._pantalla) >= PANTALLA_MAX:
            self._error("mostrar_en_pantalla: pantalla llena")
            return False
        self._pantalla.append(token)
        self._evento(f"mostrar_en_pantalla({valor}) -> {token}")
        return True

    # --------------------------------------------------
    # cpu
    # --------------------------------------------------
    def cpu_poner(self, valor: int) -> bool:
        from brooder.constantes import ARG_BUS

        if valor == ARG_BUS and not self._bus_valido:
            self._error("cpu_poner: bus vacío")
            return False
        v = self._resolver(valor, self._bus_valor)
        if not (0 <= v <= 999):
            self._error("cpu_poner: valor fuera de rango")
            return False
        self._acumulador = v
        self._evento(f"cpu_poner({valor}) -> acumulador={v}")
        return True

    def cpu_sumar(self, valor: int) -> bool:
        from brooder.constantes import ARG_BUS

        if valor == ARG_BUS and not self._bus_valido:
            self._error("cpu_sumar: bus vacío")
            return False
        v = self._resolver(valor, self._bus_valor)
        if not (0 <= v <= 999):
            self._error("cpu_sumar: valor fuera de rango")
            return False
        self._acumulador = max(0, min(999, self._acumulador + v))
        self._evento(f"cpu_sumar({valor}) -> acumulador={self._acumulador}")
        return True

    def cpu_cociente(self) -> bool:
        self._bus_valor = self._acumulador // 10
        self._bus_valido = True
        self._evento(f"cpu_cociente -> {self._bus_valor}")
        return True

    def cpu_resto(self) -> bool:
        self._bus_valor = self._acumulador % 10
        self._bus_valido = True
        self._evento(f"cpu_resto -> {self._bus_valor}")
        return True

    def leer_cpu(self) -> bool:
        self._bus_valor = self._acumulador
        self._bus_valido = True
        self._evento(f"leer_cpu -> {self._bus_valor}")
        return True

    # --------------------------------------------------
    # disco (el respaldo lo define cada subclase)
    # --------------------------------------------------
    def mover_cabezal_disco(self, direccion: int) -> bool:
        from brooder.constantes import ARG_BUS

        if direccion == ARG_BUS and not self._bus_valido:
            self._error("mover_cabezal_disco: bus vacío")
            return False
        d = self._resolver(direccion, self._bus_valor)
        if not self._es_direccion_valida(d):
            self._error("mover_cabezal_disco: dirección inválida")
            return False
        self._disco_cabezal = d
        self._evento(f"mover_cabezal_disco -> {d}")
        return True

    def leer_disco(self) -> bool:
        token = self._leer_disco_interno(self._disco_cabezal)
        self._bus_valor = token
        self._bus_valido = True
        self._evento(f"leer_disco[{self._disco_cabezal}] -> {token}")
        return True

    def escribir_disco(self, valor: int) -> bool:
        from brooder.constantes import ARG_BUS

        if valor == ARG_BUS and not self._bus_valido:
            self._error("escribir_disco: bus vacío")
            return False
        v = self._resolver(valor, self._bus_valor)
        if not self._es_token_valido(v):
            self._error("escribir_disco: token inválido")
            return False
        if not self._escribir_disco_interno(self._disco_cabezal, v):
            self._error("escribir_disco: rechazado por el dispositivo")
            return False
        self._escrituras_disco += 1
        self._evento(f"escribir_disco[{self._disco_cabezal}] <- {v}")
        return True

    # --------------------------------------------------
    # memoria
    # --------------------------------------------------
    def mover_puntero_memoria(self, direccion: int) -> bool:
        from brooder.constantes import ARG_BUS

        if direccion == ARG_BUS and not self._bus_valido:
            self._error("mover_puntero_memoria: bus vacío")
            return False
        d = self._resolver(direccion, self._bus_valor)
        if not (0 <= d < N_RANURAS_MEMORIA):
            self._error("mover_puntero_memoria: dirección inválida")
            return False
        self._memoria_puntero = d
        self._evento(f"mover_puntero_memoria -> {d}")
        return True

    def leer_memoria(self) -> bool:
        self._bus_valor = self._memoria[self._memoria_puntero]
        self._bus_valido = True
        self._evento(f"leer_memoria[{self._memoria_puntero}] -> {self._bus_valor}")
        return True

    def escribir_memoria(self, valor: int) -> bool:
        from brooder.constantes import ARG_BUS

        if valor == ARG_BUS and not self._bus_valido:
            self._error("escribir_memoria: bus vacío")
            return False
        v = self._resolver(valor, self._bus_valor)
        if not self._es_token_valido(v):
            self._error("escribir_memoria: token inválido")
            return False
        self._memoria[self._memoria_puntero] = v
        self._escrituras_memoria += 1
        self._evento(f"escribir_memoria[{self._memoria_puntero}] <- {v}")
        return True

    # --------------------------------------------------
    # audio / gpu / red
    # --------------------------------------------------
    def reproducir_audio(self, frecuencia: int) -> bool:
        from brooder.constantes import ARG_BUS

        if frecuencia == ARG_BUS and not self._bus_valido:
            self._error("reproducir_audio: bus vacío")
            return False
        f = self._resolver(frecuencia, self._bus_valor)
        self._pitidos.append((self._paso, f))
        self._evento(f"reproducir_audio({f})")
        return True

    def usar_gpu(self) -> bool:
        # el compositor GPU prepara un frame nuevo: la pantalla
        # se vacía y puede volver a dibujarse (errores recuperables)
        self._pantalla = []
        self._usos_gpu += 1
        self._evento("usar_gpu: frame compuesto, pantalla vaciada")
        return True

    def leer_red(self) -> bool:
        # v1: sin dispositivo de red activo. La primitiva existe,
        # está documentada y devuelve error controlado.
        self._error("leer_red: sin red (desactivado)")
        return False

    # --------------------------------------------------
    # registro del sistema (primera macro-primitiva)
    # --------------------------------------------------
    def registrar_log(self, mensaje: int) -> bool:
        """Añade una entrada al anillo del registro del sistema.

        El mensaje es un id de MENSAJES_LOG (vocabulario cerrado):
        la IA nunca dicta texto libre — solo elige QUÉ evento
        declarar, y la línea formateada la produce la máquina.
        """
        if not (0 <= mensaje < len(MENSAJES_LOG)):
            self._error("registrar_log: mensaje desconocido")
            return False
        self._registro.append((self._paso, mensaje))
        nivel, texto = MENSAJES_LOG[mensaje]
        self._evento(f"registrar_log[{mensaje}] {nivel}: {texto}")
        return True

    def panel_registro(self) -> list:
        """Últimas entradas formateadas para el panel de la TUI."""
        ultimas = list(self._registro)[-REGISTRO_PANEL_LINEAS:]
        lineas = [formatear_registro(paso, mensaje) for paso, mensaje in ultimas]
        lineas += [""] * (REGISTRO_PANEL_LINEAS - len(lineas))
        return lineas

    # --------------------------------------------------
    # dispositivo externo (Fase 1: pendrive virtual)
    # --------------------------------------------------
    # Ciclo de vida: conectar (externo) -> montar (la IA decide) ->
    # desmontar (la IA decide) -> desconectar (externo). Si el mundo
    # exterior desconecta con el pendrive montado, el kernel anota
    # "extraccion insegura" en el registro y libera el estado: es la
    # versión virtual de perder el búfer sin sincronizar.
    def montar_dispositivo(self) -> bool:
        """Acepta el pendrive presente en el conector USB."""
        if not self._disp_conectado:
            self._error("montar_dispositivo: no hay dispositivo en el conector")
            return False
        if self._disp_montado:
            self._error("montar_dispositivo: ya montado")
            return False
        self._disp_montado = True
        # dmesg del kernel: la operación queda anotada en el registro
        self._registro.append((self._paso, MENSAJE_LOG_DISP_MONTADO))
        self._evento("montar_dispositivo: dispositivo montado")
        return True

    def desmontar_dispositivo(self) -> bool:
        """Libera el pendrive de forma limpia (extracción segura)."""
        if not self._disp_montado:
            self._error("desmontar_dispositivo: no hay dispositivo montado")
            return False
        self._disp_montado = False
        self._registro.append((self._paso, MENSAJE_LOG_DISP_DESMONTADO))
        self._evento("desmontar_dispositivo: dispositivo desmontado")
        return True

    def conectar_dispositivo(self) -> bool:
        """Hot-plug: el mundo exterior enchufa el pendrive (kernel)."""
        if self._disp_conectado:
            self._error("conectar_dispositivo: ya hay dispositivo")
            return False
        self._disp_conectado = True
        self._evento("conector USB: pendrive conectado")
        return True

    def desconectar_dispositivo(self) -> bool:
        """Hot-plug: el mundo exterior retira el pendrive (kernel).

        Si el pendrive estaba montado, la extracción es INSEGURA: el
        kernel lo anota como ERROR en el registro (la traza que un SO
        real deja al retirar un USB sin desmontar) y libera el estado
        del conector.
        """
        if not self._disp_conectado:
            self._error("desconectar_dispositivo: el conector está vacío")
            return False
        insegura = self._disp_montado
        self._disp_conectado = False
        self._disp_montado = False
        if insegura:
            self._registro.append((self._paso, MENSAJE_LOG_EXTRACCION_INSEGURA))
            self._evento("conector USB: EXTRACCION INSEGURA (pendrive montado)")
        else:
            self._evento("conector USB: pendrive desconectado")
        return not insegura

    @property
    def dispositivo_conectado(self) -> bool:
        return self._disp_conectado

    @property
    def dispositivo_montado(self) -> bool:
        return self._disp_montado

    # --------------------------------------------------
    # observación
    # --------------------------------------------------
    def instante(self) -> InstanteMaquina:
        return InstanteMaquina(
            teclado_pendientes=len(self._teclado),
            teclado_hay_datos=bool(self._teclado),
            bus_valor=self._bus_valor,
            bus_valido=self._bus_valido,
            acumulador=self._acumulador,
            pantalla=list(self._pantalla),
            disco_cabezal=self._disco_cabezal,
            disco_contenido=tuple(self._disco),
            memoria_puntero=self._memoria_puntero,
            memoria_contenido=tuple(self._memoria),
            pitidos=tuple(self._pitidos),
            red_paquetes=len(self._red_paquetes),
            registro=tuple(self._registro),
            dispositivo_conectado=self._disp_conectado,
            dispositivo_montado=self._disp_montado,
            ultimo_error=self._ultimo_error,
            ultimo_evento=self._ultimo_evento,
            escrituras_disco=self._escrituras_disco,
            escrituras_memoria=self._escrituras_memoria,
            usos_gpu=self._usos_gpu,
            teclado_leidos=tuple(self._teclado_leidos),
        )

    # ganchos de persistencia del disco
    def _leer_disco_interno(self, direccion: int) -> int:
        return self._disco[direccion]

    def _escribir_disco_interno(self, direccion: int, token: int) -> bool:
        self._disco[direccion] = token
        return True

    # señalización interna
    def _error(self, mensaje: str) -> None:
        self._ultimo_error = mensaje
        self._ultimo_evento = mensaje

    def _evento(self, mensaje: str) -> None:
        self._ultimo_error = ""
        self._ultimo_evento = mensaje

    def avanzar_paso(self) -> None:
        """Lo llama el núcleo al cerrar cada ciclo."""
        self._paso += 1

    @property
    def teclado_pendientes(self) -> int:
        return len(self._teclado)
