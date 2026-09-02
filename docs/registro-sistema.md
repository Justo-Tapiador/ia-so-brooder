# El registro del sistema — Fase 0 del escalado

**Macro-primitiva `REGISTRAR_LOG` (id 17) · primera pieza del contrato de
nivel de sistema de IA-SO Brooder.**

## Motivación

Las primitivas 0-16 son operaciones *micro*: mueven un token entre dos
dispositivos y consumen un ciclo. Sirven para el currículo actual (eco,
suma, guardado), pero el objetivo declarado del proyecto —aprender a
operar dispositivos ricos como almacenamiento extraíble o red— exige
acciones con semántica de *sistema*, no de circuito.

La respuesta no es enseñarle a la red la mecánica de cada dispositivo
(eso es trabajo de *drivers*, código determinista), sino subir el nivel
del contrato: la IA decide **qué y cuándo**; el núcleo hace el **cómo**.
Eso es exactamente lo que hace un kernel clásico con sus `syscalls`, y
`REGISTRAR_LOG` es la primera de esa familia: una "syscall" mínima, con
superficie de riesgo nula, que valida el patrón completo de extremo a
extremo antes de que lleguen las macro-primitivas de verdad (abrir
archivos, pintar búferes, descargar de red).

El registro además responde a un requisito recurrente de los escenarios
objetivo: «escribir en pantalla logs de los estados principales de los
procesos». Con `REGISTRAR_LOG`, un cerebro entrenado podrá declarar
eventos de proceso mientras ejecuta otras tareas, y el panel los
mostrará junto a la pantalla del usuario.

## Diseño

### Vocabulario cerrado, no texto libre

La IA emite `registrar_log(m)` con `m` ∈ 0..7, un id de la tabla
`MENSAJES_LOG` (`brooder/constantes.py`): ocho mensajes con nivel INFO,
AVISO o ERROR. La línea formateada («[0012] ERROR| dispositivo no
listo») la produce la máquina, no la red. Esta es la misma política de
seguridad del resto del contrato: nada de lo que emite el agente se
interpreta como texto arbitrario, igual que `mostrar_en_pantalla` solo
acepta tokens del alfabeto.

El tipo de argumento nuevo `"mensaje"` entra en la máquina de estados
de `mascara_argumentos()`: durante el entrenamiento, la cabeza de
argumento solo propone ids válidos —las combinaciones que la máquina
siempre rechazaría ni siquiera se exploran, como exige el contrato de
tipos del kernel.

### Anillo a lo dmesg

El registro vive en `MaquinaBase._registro`: un `deque` con capacidad
fija de 8 entradas `(paso, mensaje)`. Cuando llega la novena, la
primera se pierde — como el *kernel ring buffer* de Linux. Decisiones
de ciclo de vida:

- `reiniciar_registros()` **no** lo limpia: es la consola del *sistema*,
  no un registro volátil del *proceso* (igual que el disco y la RAM,
  que persisten entre solicitudes).
- `reiniciar()` (arranque en frío) **sí** lo vacía: un reboot nuevo
  arranca con la consola limpia.

El panel de la TUI (`pantalla.render_panel_registro`) muestra las 4
últimas entradas, formateadas con `formatear_registro()`.

### Aislamiento total del estado de usuario

`registrar_log` no toca el bus, el acumulador, la pantalla, el disco ni
la RAM: la consola del kernel no es un dispositivo de datos. Los tests
fijan expresamente este aislamiento. La observación de la red tampoco
cambia en esta fase: `OBS_DIM` sigue siendo 21 y `InstanteMaquina`
expone el registro como tupla inmutable por si el currículo futuro
quiere premiar su uso, pero hoy ningún canal de percepción lo lee.

### Compatibilidad de los cerebros existentes

El id 17 se añade **al final** del enum: los ids 0-16 conservan su
significado. `montar_ssd` y `CerebroBrooder.cargar` construyen la red
desde la **config guardada en el checkpoint** — que incluye su propio
`n_primitivas` — así que los SSD incubados con el contrato viejo
montan exactamente igual que antes; su cabeza de 17 salidas
simplemente no puede alcanzar el id 17. La macro-primitiva queda
disponible para los cerebros que se reentrenen con este código. No hay
migración de pesos: la decisión deliberada es que el contrato nuevo se
incuba, no se parchea.

## Qué cambió, archivo a archivo

| Archivo | Cambio |
|---|---|
| `brooder/constantes.py` | `REGISTRAR_LOG = 17`, `MENSAJES_LOG`, `REGISTRO_CAPACIDAD`, `REGISTRO_PANEL_LINEAS`, `formatear_registro()`, tipo de argumento `"mensaje"` en la máscara |
| `brooder/primitivas/base.py` | `registrar_log()` y `panel_registro()` en el contrato y en `MaquinaBase`; anillo `_registro`; `InstanteMaquina.registro` |
| `brooder/pantalla.py` | `render_panel_registro()` y `mostrar_registro()` |
| `brooder/cli.py` | sección de demostración de la macro-primitiva en `brooder demo` |
| `tests/test_registro.py` | 10 tests: contrato, máscara, anillo, aislamiento, persistencia, panel y compatibilidad |
| `README.md` / `README-ES.md` | fila 17 en la tabla de primitivas y nota explicativa |

## Cómo se usa hoy (y mañana)

Hoy (Fase 0.5): el cerebro incubado **decide** cuándo registrar. La
demo muestra la sección «REGISTRAR_LOG — decisión propia del cerebro»
con los eventos que la política neuronal emitió durante las
solicitudes reales; con un cerebro del contrato viejo (17 salidas) la
demo detecta el contrato y muestra la vía sintética de la Fase 0.

Mañana (Fase 1): las siguientes macro-primitivas de la hoja de ruta:
`ABRIR_ARCHIVO`, `LEER_BLOQUE`, `PINTAR_TEXTO` — el pendrive virtual
y el modo píxel.

## Fase 0.5 — el cerebro traza

La Fase 0 dejó la fontanería y un guion que emitía eventos sintéticos.
La Fase 0.5 cierra el círculo: **la política neuronal aprende a usar su
syscall**. El cambio vive en el entorno de entrenamiento
(`brooder/entorno.py`) y es deliberadamente mínimo:

| Pieza | Diseño |
|---|---|
| Premio | `+0.10` (`R_TRAZO_VALIDO`) por trazar la operación de almacenamiento correcta |
| Ruido | `-0.03` (`R_TRAZO_RUIDO`) por repetir un mensaje ya emitido en la solicitud (rompe los bucles de log) |
| Correspondencia | tras escribir disco/RAM → mensaje 2 («escritura completada»); tras leer → mensaje 1 («lectura completada») |
| Ventana | la traza vale si llega en ≤ 2 ciclos (`TRAZO_VENTANA`): declarar *al momento*, no de memoria |
| Acotación | máximo **1 premio por tipo y por solicitud**: imposible cultivar recompensa con log-spam (que además paga su ciclo) |
| Alcance | solo GUARDAR/RECORDAR ofrecen trazado: la política natural es «un SO traza su I/O de almacenamiento» |
| Exploración | dos fases: entropía alta mientras no exista semilla de trazado (eval determinista < 20 %), consolidación después |
| Convergencia | desde la etapa con almacenamiento, la incubación solo se da por completa si el trazado ≥ 70 % (`umbral_trazado`) |

Por qué solo mensajes INFO de almacenamiento: es el subconjunto del
vocabulario con una correspondencia biunívoca operación↔mensaje, y por
tanto el único **verificable** como «correcto» sin ambigüedad. Los
mensajes de AVISO/ERROR siguen disponibles para la política (el
contrato no los restringe); premiarlos exigiría definir qué error
«toca» declarar en cada fallo, y eso es diseño de una fase posterior.

La misma tabla de recompensas enseña la *correspondencia*, no el
volumen: un mensaje equivocado o tardío no castiga (paga su ciclo como
cualquier acción no productiva), y el oráculo —la política ideal
escrita a mano— traza el 100 % y sirve de referencia de la métrica
`tasa_trazado`: trazas correctas / operaciones trazables.

Qué cambió, archivo a archivo:

| Archivo | Cambio |
|---|---|
| `brooder/constantes.py` | `MENSAJE_LOG_LECTURA`, `MENSAJE_LOG_ESCRITURA` (ids canónicos) |
| `brooder/entorno.py` | `R_TRAZO_VALIDO`, `TRAZO_VENTANA`, moldeado del trazado, `tasa_trazado()`, oráculo que traza |
| `brooder/nucleo.py` | `ResultadoSolicitud.trazos` (evidencia por solicitud) |
| `brooder/incubadora.py` | `evaluar(con_trazado=True)`, métrica `trazado_eval` en el entrenamiento |
| `brooder/cli.py` | demo con decisión propia del cerebro (fallback sintético), trazado en `diagnostico` e `incubar` |
| `tests/test_trazado.py` | 10 tests: premio exacto, ventana, acotación, oráculo 100 %, evidencia en el núcleo |

La observación no cambia (21 canales): el cerebro ya percibe si hubo
escritura/lectura en la solicitud; solo crece la cabeza de decisión
(17 → 18 salidas), exactamente como previó la Fase 0.

## Por qué esta fase existe

Cada pieza del patrón "syscall" se estrena aquí con riesgo mínimo:
extensión del enum sin renumerar, tipo de argumento nuevo en la máscara,
estado nuevo en la máquina, panel nuevo en la TUI, sección nueva en la
demo y tests de compatibilidad. Cuando lleguen las macro-primitivas de
almacenamiento y red, el mecanismo ya estará probado y solo habrá que
decidir su semántica — no su fontanería.
