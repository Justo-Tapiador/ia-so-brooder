"""
Solicitudes — el contrato entre el usuario y la IA-SO
=====================================================

Una `Solicitud` es lo que el "usuario" (el entorno de entrenamiento
en la incubadora, o un humano real a través del intérprete de
arranque) le pide a Brooder, junto con:

* los tokens que recibirán por el teclado,
* el presupuesto de ciclos del núcleo,
* las condiciones de éxito (verificables por el hardware),
* un comparador de pantalla ("¿lo mostrado hasta ahora va bien?").

La clave del diseño: la MISMA clase se usa para entrenar y para
operar. La política neuronal percibe exactamente los mismos
estímulos en la incubadora y en el PC de nacimiento — no hay
desajuste entrenamiento/inferencia.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from brooder.constantes import (
    LETRAS_ENTRENAMIENTO,
    N_RANURAS_DISPOSITIVO,
    PRESUPUESTO_CICLOS,
    TOKEN_ALARMA,
    TOKEN_DE_CARACTER,
    TOKEN_MAS,
    Tarea,
    texto_a_tokens,
    tokens_a_texto,
)
from brooder.primitivas.base import InstanteMaquina

PRESUPUESTO_MAX = max(PRESUPUESTO_CICLOS.values())


@dataclass
class Solicitud:
    """Petición atomica dirigida a la IA-SO."""

    tarea: Tarea
    tokens: list = field(default_factory=list)   # lo que entrará por teclado
    esperado: list = field(default_factory=list)  # lo que debe aparecer en pantalla
    datos: dict = field(default_factory=dict)     # valores internos (a, b, K, V...)

    @property
    def presupuesto(self) -> int:
        return PRESUPUESTO_CICLOS[self.tarea.name]

    # --------------------------------------------------
    # condiciones de éxito (¡verificables por el hardware!)
    # --------------------------------------------------
    def exito(self, m: InstanteMaquina) -> bool:
        """Comprueba si la solicitud quedó resuelta.

        Cada tarea exige, además de la pantalla correcta, que se
        haya usado el recurso implicado: la CPU para SUMA, el disco
        para GUARDAR, la RAM para RECORDAR. Así la IA no puede
        "adivinar" la respuesta: tiene que administrar la máquina.
        """
        if m.pantalla != list(self.esperado):
            return False
        if self.tarea == Tarea.ECO:
            return True
        if self.tarea == Tarea.SUMA:
            return m.acumulador == self.datos["suma"]
        if self.tarea == Tarea.GUARDAR:
            return m.disco_contenido[self.datos["K"]] == self.datos["V"]
        if self.tarea == Tarea.RECORDAR:
            return m.memoria_contenido[self.datos["K"]] == self.datos["V"]
        if self.tarea == Tarea.AVISO:
            return self.pitidos_validos(m) == 1
        if self.tarea == Tarea.DISPOSITIVO:
            # la pantalla debe quedar vacía en los modos de ciclo de
            # vida (no hay nada que mostrar: administrar hardware no
            # produce salida de consola) y el pendrive debe quedar en
            # el estado pedido, AÚN CONECTADO: desmontar no es perder
            # el dispositivo, es liberarlo de forma segura para que
            # el mundo pueda retirarlo.
            modo = self.datos.get("modo")
            if modo == "desmontar":
                return (not m.dispositivo_montado) and m.dispositivo_conectado
            if modo == "montar":
                return m.dispositivo_montado and m.dispositivo_conectado
            # Fase 1.5: modos de ALMACENAMIENTO. El éxito exige el
            # triple verificable por hardware: dispositivo montado,
            # dato en la ranura pedida y el valor recuperado en
            # pantalla. La condición de ranura impide "adivinar"
            # mostrando lo tecleado sin escribir; en modo leer, el
            # valor NUNCA pasó por el teclado (solo lo trae el
            # pendrive), así que la pantalla solo puede llenarse con
            # una lectura genuina del dispositivo.
            if modo == "escribir":
                K, V = self.datos["K"], self.datos["V"]
                return (
                    m.dispositivo_montado
                    and m.dispositivo_conectado
                    and m.dispositivo_ranuras[K] == V
                    and list(m.pantalla) == [V]
                )
            if modo == "leer":
                K, V = self.datos["K"], self.datos["V"]
                return (
                    m.dispositivo_montado
                    and m.dispositivo_conectado
                    and list(m.pantalla) == [V]
                )
        return False

    def pitidos_validos(self, m: InstanteMaquina) -> int:
        """Pitidos emitidos DESPUÉS de leer el token ALARMA."""
        paso_alarma = next(
            (paso for paso, tok in m.teclado_leidos if tok == TOKEN_ALARMA), None
        )
        if paso_alarma is None:
            return 0
        return sum(1 for paso, _ in m.pitidos if paso > paso_alarma)

    def pantalla_coincide(self, m: InstanteMaquina) -> bool:
        """Comparador de consola: ¿lo mostrado es prefijo de lo esperado?"""
        esperado = list(self.esperado)
        return m.pantalla == esperado[: len(m.pantalla)]

    def descripcion(self) -> str:
        if self.tarea == Tarea.ECO:
            return f"eco('{tokens_a_texto(self.tokens)}')"
        if self.tarea == Tarea.SUMA:
            return f"suma({self.datos['a']}+{self.datos['b']})"
        if self.tarea == Tarea.GUARDAR:
            return (
                f"guardar(K={self.datos['K']}, V='{tokens_a_texto([self.datos['V']])}')"
            )
        if self.tarea == Tarea.RECORDAR:
            return (
                f"recordar(K={self.datos['K']}, V='{tokens_a_texto([self.datos['V']])}')"
            )
        if self.tarea == Tarea.AVISO:
            return f"aviso('{tokens_a_texto([self.datos['X']])}')"
        if self.tarea == Tarea.DISPOSITIVO:
            modo = self.datos.get("modo", "montar")
            if modo in ("escribir", "leer"):
                return (
                    f"dispositivo({modo} K={self.datos['K']}, "
                    f"V='{tokens_a_texto([self.datos['V']])}')"
                )
            return f"dispositivo({modo})"
        return self.tarea.name

    # --------------------------------------------------
    # generación aleatoria (para la incubadora)
    # --------------------------------------------------
    @staticmethod
    def aleatoria(tarea: Tarea, rng) -> "Solicitud":
        """Fabrica una solicitud aleatoria y bien etiquetada."""
        if tarea == Tarea.ECO:
            # longitud 1..8: cubre con margen los ecos largos de la
            # demo ('BROODER', 7 letras). El presupuesto (24 ciclos)
            # admite hasta 12; entrenar hasta 8 deja holgura sin
            # alargar los episodios. Regresión de la Fase 0.5: con
            # 1..5 el cerebro con trazado fallaba el eco de 7.
            longitud = rng.randint(1, 8)
            tokens = [rng.choice(LETRAS_ENTRENAMIENTO) for _ in range(longitud)]
            return Solicitud(Tarea.ECO, tokens=list(tokens), esperado=list(tokens))

        if tarea == Tarea.SUMA:
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            tokens = [a, TOKEN_MAS, b]
            suma = a + b
            esperado = [int(c) for c in str(suma)]
            return Solicitud(
                Tarea.SUMA, tokens=tokens, esperado=esperado,
                datos={"a": a, "b": b, "suma": suma},
            )

        if tarea in (Tarea.GUARDAR, Tarea.RECORDAR):
            K = rng.randrange(10)
            V = rng.choice(LETRAS_ENTRENAMIENTO)
            return Solicitud(
                tarea, tokens=[K, V, K], esperado=[V], datos={"K": K, "V": V},
            )

        if tarea == Tarea.AVISO:
            X = rng.choice(LETRAS_ENTRENAMIENTO)
            return Solicitud(
                Tarea.AVISO, tokens=[X, TOKEN_ALARMA], esperado=[X],
                datos={"X": X},
            )

        if tarea == Tarea.DISPOSITIVO:
            # Fase 1: administrar el pendrive del conector. Las
            # solicitudes de CICLO DE VIDA no entran por el teclado ni
            # esperan salida en pantalla: son eventos de hardware.
            # Fase 1.5: los modos de ALMACENAMIENTO (escribir/leer)
            # replican el contrato de GUARDAR contra el pendrive:
            # "escribir 3 P" teclea ranura y valor, y el éxito exige
            # escribir de verdad y recuperar el valor del dispositivo
            # (pantalla = V); "leer 3 P" NO teclea el valor — viene
            # grabado en el pendrive que enchufa el entorno — y la
            # única forma de mostrarlo es leer el dispositivo.
            modo = rng.choice(
                ("montar", "desmontar", "escribir", "leer")
            )
            if modo in ("escribir", "leer"):
                K = rng.randrange(N_RANURAS_DISPOSITIVO)
                V = rng.choice(LETRAS_ENTRENAMIENTO)
                if modo == "escribir":
                    # como GUARDAR: ranura, valor, ranura
                    tokens = [K, V, K]
                else:
                    # solo la ranura: el valor vive en el pendrive
                    tokens = [K]
                return Solicitud(
                    Tarea.DISPOSITIVO, tokens=tokens, esperado=[V],
                    datos={"modo": modo, "K": K, "V": V},
                )
            return Solicitud(
                Tarea.DISPOSITIVO, tokens=[], esperado=[],
                datos={"modo": modo},
            )

        raise ValueError(f"Tarea sin generador: {tarea}")

    # --------------------------------------------------
    # construcción desde texto (para el intérprete interactivo)
    # --------------------------------------------------
    _PATRON_SUMA = re.compile(r"^(\d)\s*\+\s*(\d)$")
    # Hotfix de campo (post-Fase 1.5): el espacio entre número y letra
    # es OPCIONAL en los pares. 'leer 3P' es la misma solicitud que
    # 'leer 3 P', igual que '3+5' y '3 + 5' ya eran la misma suma — el
    # dedo del usuario pega dígito y letra porque la aritmética no usa
    # espacios, y el parser no debía castigarlo.
    _PATRON_PAR = re.compile(r"^(\d)\s*([A-Z])$")
    # Y el límite honesto: el espacio tras el VERBO sigue siendo
    # obligatorio. 'leer3P' no es un eco válido ni un par: devolver
    # None hace que el intérprete muestre el remedio (la línea de
    # formatos) en vez de encomendarle al cerebro un eco confuso
    # con dígitos que nunca entrenó (condenado a [FALLO]).
    _PATRON_VERBO_PEGADO = re.compile(
        r"^(?:ECO|SUMA|GUARDAR|RECORDAR|AVISO|ESCRIBIR|LEER|MONTAR|DESMONTAR)\S"
    )

    @staticmethod
    def desde_texto(texto: str) -> "Solicitud | None":
        """Interpreta una línea del usuario como solicitud.

        Formatos aceptados:
          HOLA            -> ECO
          eco HOLA        -> ECO
          3+5             -> SUMA
          suma 3+5        -> SUMA
          guardar 4 G     -> GUARDAR (ranura 4, valor G)
          recordar 2 Z    -> RECORDAR
          aviso A         -> AVISO
          montar          -> DISPOSITIVO (montar el pendrive del conector)
          desmontar       -> DISPOSITIVO (extracción segura)
          escribir 3 P    -> DISPOSITIVO (guardar P en la ranura 3 del
                            pendrive montado y recuperarla en pantalla)
          leer 3 P        -> DISPOSITIVO (leer la ranura 3 del pendrive
                            montado; P es el valor esperado)

        El espacio entre número y letra es opcional en los pares
        ('leer 3P' = 'leer 3 P'), como ya lo era en la suma ('3+5' =
        '3 + 5'); el espacio tras el verbo no ('leer3P' no parsea y
        el intérprete responde con el remedio).
        """
        t = texto.strip().upper()
        if not t:
            return None

        if t.startswith(("ECO ", "SUMA ", "GUARDAR ", "RECORDAR ", "AVISO ",
                         "ESCRIBIR ", "LEER ")):
            verbo, resto = t.split(" ", 1)
        elif t in ("MONTAR", "DESMONTAR"):
            # Fase 1: solicitudes de dispositivo (sin argumentos). Si
            # no hay pendrive conectado, la solicitud se atiende igual
            # y falla con el error del kernel: honesto, como un
            # 'mount' sobre un conector vacío.
            return Solicitud(
                Tarea.DISPOSITIVO, tokens=[], esperado=[],
                datos={"modo": t.lower()},
            )
        elif Solicitud._PATRON_VERBO_PEGADO.match(t):
            # verbo pegado a su argumento ('leer3P'): el espacio tras
            # el verbo sigue siendo obligatorio — mejor el remedio
            # visible que un eco condenado
            return None
        else:
            verbo, resto = "", t

        # sin verbo explícito: ¿es una suma del tipo "3+5"?
        if verbo == "":
            m = Solicitud._PATRON_SUMA.match(t)
            if m:
                verbo, resto = "SUMA", t

        if verbo in ("", "ECO"):
            tokens = texto_a_tokens(resto)
            if len(tokens) < 1:
                return None
            return Solicitud(Tarea.ECO, tokens=tokens, esperado=list(tokens))

        if verbo == "SUMA":
            m = Solicitud._PATRON_SUMA.match(resto)
            if not m:
                return None
            a, b = int(m.group(1)), int(m.group(2))
            return Solicitud(
                Tarea.SUMA, tokens=[a, TOKEN_MAS, b],
                esperado=[int(c) for c in str(a + b)],
                datos={"a": a, "b": b, "suma": a + b},
            )

        if verbo in ("GUARDAR", "RECORDAR"):
            m = Solicitud._PATRON_PAR.match(resto)
            if not m:
                return None
            K = int(m.group(1))
            V = TOKEN_DE_CARACTER[m.group(2)]
            tarea = Tarea.GUARDAR if verbo == "GUARDAR" else Tarea.RECORDAR
            return Solicitud(tarea, tokens=[K, V, K], esperado=[V],
                             datos={"K": K, "V": V})

        if verbo in ("ESCRIBIR", "LEER"):
            # Fase 1.5: almacenamiento en el pendrive montado. Misma
            # firma que guardar/recordar (ranura + valor), pero la
            # tarea es DISPOSITIVO: el kernel exige montaje y las
            # ranuras viven en el medio extraíble. En 'leer', el valor
            # es lo QUE DEBERÍA estar ya en el pendrive (lo escribió
            # una sesión anterior): la solicitud NO lo teclea.
            m = Solicitud._PATRON_PAR.match(resto)
            if not m:
                return None
            K = int(m.group(1))
            V = TOKEN_DE_CARACTER[m.group(2)]
            if K >= N_RANURAS_DISPOSITIVO:
                return None  # el pendrive solo tiene ranuras 0..7
            if verbo == "ESCRIBIR":
                return Solicitud(
                    Tarea.DISPOSITIVO, tokens=[K, V, K], esperado=[V],
                    datos={"modo": "escribir", "K": K, "V": V},
                )
            return Solicitud(
                Tarea.DISPOSITIVO, tokens=[K], esperado=[V],
                datos={"modo": "leer", "K": K, "V": V},
            )

        if verbo == "AVISO":
            tokens = texto_a_tokens(resto)
            if len(tokens) != 1:
                return None
            return Solicitud(Tarea.AVISO, tokens=[tokens[0], TOKEN_ALARMA],
                             esperado=[tokens[0]], datos={"X": tokens[0]})

        return None
