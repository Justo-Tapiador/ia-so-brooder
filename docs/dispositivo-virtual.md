# El pendrive virtual — Fase 1 del escalado

**Macro-primitivas `MONTAR_DISPOSITIVO` (18) y `DESMONTAR_DISPOSITIVO`
(19) · primer dispositivo externo con conexión en caliente.**

## Motivación

La Fase 0 estrenó el patrón «syscall» con `REGISTRAR_LOG`: la IA decide,
el núcleo ejecuta. La Fase 0.5 cerró el círculo recompensando el
trazado: el registro del sistema salió de las decisiones de la red.

La Fase 1 ataca el siguiente peldaño del objetivo declarado del
proyecto: **dispositivos que aparecen y desaparecen**. Un SO real no
nace con todo su hardware soldado: el usuario enchufa memorias USB,
retira discos, conecta redes. La máquina de Brooder era estática —su
conjunto de dispositivos era fijo desde el arranque—, y una política
entrenada contra una máquina estática no puede aprender a *reaccionar*
a eventos de hardware, solo a operar lo que ya está.

El pendrive virtual introduce ese eje con la mínima superficie:

* **El mundo exterior** conecta y desconecta el dispositivo (hot-plug).
* **El kernel** valida las transiciones y anota el ciclo en su registro.
* **La IA** percibe el estado del conector y decide si lo monta o lo
  libera. Nada más — y nada menos: la decisión es suya.

## Diseño

### Conector USB virtual y sus dos estados

La máquina (`MaquinaBase`, heredada por `PCVirtual` y `PCReal`) gana
dos banderas: `dispositivo_conectado` (presencia física) y
`dispositivo_montado` (aceptado por la política). El ciclo de vida:

```
conectar (mundo exterior)  ->  montar (decisión de la IA)
                          ->  desmontar (decisión de la IA)
                          ->  desconectar (mundo exterior)
```

Tres decisiones de diseño que conviene anotar:

1. **El hot-plug NO es una primitiva.** No existe `conectar_dispositivo`
   en el contrato de la IA: enchufar y desenchufar son acciones del
   mundo físico, aplicadas directamente por el kernel (la demo, el
   entorno de entrenamiento, o el comando interactivo `:pendrive`). La
   IA no puede fabricar hardware para administrarlo — igual que un
   proceso de usuario no puede crear un USB por syscall.

2. **La extracción insegura es un ERROR del kernel.** Si el mundo
   retira el pendrive mientras está montado, el kernel registra
   `ERROR| extraccion insegura` por sí mismo y libera el estado del
   conector. Es la traza que un SO real deja al arrancar un USB sin
   desmontar: la protección existe aunque la IA no reaccione.

3. **El kernel anota el ciclo (dmesg).** `montar` y `desmontar` dejan
   `INFO| dispositivo montado/desmontado` en el anillo del registro.
   A diferencia del trazado de la Fase 0.5 —donde la IA *declara* su
   propia I/O con `REGISTRAR_LOG`— estas líneas las escribe el kernel
   al ejecutar la operación, como hace dmesg con `usb 1-1: new device`.
   El vocabulario cerrado `MENSAJES_LOG` crece de 8 a 11 ids (los tres
   nuevos se añaden al final; los existentes no se renumeran).

El estado del conector persiste entre solicitudes (es hardware
enchufado, no estado del proceso) y se pierde solo en arranque en frío,
exactamente igual que el anillo del registro.

### La tarea DISPOSITIVO

`Tarea.DISPOSITIVO` (id 5) encapsula los dos escenarios:

* **modo `montar`**: el mundo enchufa el pendrive; éxito cuando está
  montado y conectado.
* **modo `desmontar`**: el entorno lo enchufa *y lo deja ya montado*
  (lo montó «la sesión anterior»); éxito cuando queda desmontado y
  **aún conectado** — la extracción segura libera sin perder.

La solicitud no usa teclado ni pantalla: es un evento de hardware. La
política lee los canales y decide. El presupuesto es corto (12 ciclos):
la acción correcta es de un ciclo; el margen es para explorar sin
regalar tiempo infinito.

La recompensa de administración (`R_DISP_OK = 0.25`) solo existe en la
tarea DISPOSITIVO — la misma lección anti-señuelo de la Fase 0.5: en
las tareas clásicas el conector está vacío y `MONTAR`/`DESMONTAR`
fracasan en la máquina pagando su ciclo, sin cultivar premio.

### Compatibilidad de contratos: la observación se extiende por el final

La Fase 1 cambia dos dimensiones a la vez: la percepción (3 canales
nuevos) y las salidas (2 primitivas nuevas). Las primitivas ya se
añadían al final del enum (lección de la Fase 0); la percepción exige
más cuidado porque el vector se consume *posicionalmente*:

```
[0..4]   one-hot de las tareas CLÁSICAS      (las 5 de siempre)
[5..20]  las 16 señales de siempre
[21]     disp_tarea: 1.0 si la solicitud es de dispositivo
[22]     disp_conectado
[23]     disp_montado
```

La tarea DISPOSITIVO **no entra en el one-hot**: usa su canal escalar
propio al final. Así las 21 primeras posiciones son bit a bit el
contrato viejo, y el núcleo (y `evaluar`) recortan la observación a
`dim_entrada` del cerebro montado. Un cerebro de la Fase 0.5 arranca
con el kernel nuevo, resuelve sus 5 tareas al 100 % y simplemente no ve
el conector: la demo detecta el contrato y muestra la vía sintética
del núcleo, como ya hacía con `REGISTRAR_LOG`.

### El trasplante de contrato

Reentrenar desde cero desperdiciaría el millón y medio de pasos de la
Fase 0.5. En su lugar, `brooder incubar --reanudar` detecta un
checkpoint del contrato viejo y lo **trasplanta**
(`incubadora.expandir_estado_contrato`):

* `codificador.0.weight` [96×21] → [96×24]: las columnas nuevas (los
  canales del dispositivo) se añaden a **cero**. Con los canales en
  0.0 — exactamente como están en las tareas clásicas — la salida del
  codificador es idéntica: cero regresión de partida (verificado por
  test: el cerebro migrado toma las mismas decisiones que el viejo,
  ciclo a ciclo, en tareas clásicas).
* `cabeza_primitiva` [18→20] y `embebido_primitiva` [18→20]: las filas
  nuevas se añaden a cero, con sesgo inicial −4.0 para que las
  macro-primitivas nuevas arranquen fuertemente desfavorecidas. Solo
  ganan masa cuando PPO encuentra recompensa en ellas; el bonus de
  novedad del entorno se encarga de que se muestreen.
* El estado de Adam se descarta (sus formas cambian): se recalienta en
  unos cientos de pasos, que es el régimen natural de un fine-tuning.

La incubación real usó además dos lecciones de esta fase (ambas
documentadas en el worklog): el refinamiento con zanahoria de trazado
reforzada para que la entropía de consolidación no erosionara el
trazado ganado, y una puerta de convergencia dual (todas las tareas
≥ 99 % **y** trazado ≥ 90 %) que congela el cerebro solo cuando todo
está verde a la vez.

## Archivos cambiados

| Archivo | Cambio |
|---------|--------|
| `brooder/constantes.py` | `Tarea.DISPOSITIVO`, presupuesto 12, etapa 5 del currículo, `MONTAR/DESMONTAR` (18/19), `MENSAJES_LOG` 8→11, `OBS_DIM` 21→24 con compatibilidad de prefijo |
| `brooder/primitivas/base.py` | banderas del conector, `montar/desmontar/conectar/desconectar_dispositivo`, dmesg del kernel, extracción insegura |
| `brooder/percepcion.py` | canales [21..23] con one-hot clásico intacto |
| `brooder/solicitudes.py` | tarea DISPOSITIVO (exito, descripción, generador, `montar`/`desmontar` en texto) |
| `brooder/entorno.py` | hot-plug del entorno, `R_DISP_OK` anti-señuelo, oráculo de dispositivo |
| `brooder/incubadora.py` | `expandir_estado_contrato` (trasplante) + reanudación con migración + `evaluar` con recorte |
| `brooder/nucleo.py` | POST del conector, recorte de observación por contrato, detalle del pendrive |
| `brooder/cli.py` | sección `DISPOSITIVO EXTERNO` de la demo (con vía sintética) y `:pendrive` interactivo |
| `brooder/pantalla.py` | pendrive en el monitor y en la ayuda interactiva |
| `tests/test_dispositivo.py` | 30 tests: ids, ciclo de vida, percepción, tarea, anti-señuelo, oráculo, trasplante, demo |

## Verificación

* Suite: **106/106** (76 de Fases 0/0.5 + 30 nuevas).
* Cerebro trasplantado y refinado: **100 % en las 6 tareas** (240
  solicitudes/tarea, determinista) con trazado 97 %/100 %.
* Demo: 7/7 + trazado propio + ciclo completo del pendrive por
  decisión del cerebro (1 ciclo por decisión) + ERROR de extracción
  forzada registrado por el kernel. Exit 0 también en consola cp1252.
* Compatibilidad real: la imagen de la Fase 0.5 arranca con el kernel
  nuevo (5 tareas al 100 %, sección sintética del dispositivo).

## Hotfix de contrato (post-entrega)

Nació de un fallo real en campo: parche de la Fase 1 aplicado **sin
copiar su imagen SSD**. El cerebro de la Fase 0.5 (21×18) arranca sin
error con el kernel nuevo —la compatibilidad de prefijo lo permite—
pero no puede decidir sobre el hardware: sus cabezas nunca emiten ids
≥ n_primitivas y los canales del dispositivo quedan fuera de su
ventana de percepción. `montar` agotaba su presupuesto en [FALLO]s
mudos (0 % resueltas).

El remedio convierte el desfase en información de primera clase:

* `nucleo.aviso_contrato(cerebro)` describe las faltas exactas
  (primitivas que faltan + canales no percibidos).
* El POST añade la línea `Cerebro`: `[ OK ] contrato 24x20` o
  `[ AVISO ] contrato 21x18 (imagen antigua)`.
* `arrancar`, `demo` y `diagnostico` imprimen el aviso en amarillo
  con el remedio (copiar la imagen SSD reentrenada sobre
  `ssd/brooder.img`); en la demo, la vía sintética del contrato viejo
  ahora también es amarilla.
* `splash_bios` acepta comprobaciones con nivel: la etiqueta ocupa
  ancho fijo (`[ OK    ]` / `[ AVISO ]`) para mantener la alineación.
* Se reparó `rojo_local` (ausente desde el primer commit):
  `brooder diagnostico` ya no crashea cuando una tarea cae < 85 % —
  la rama ✘ se ejecutó por primera vez con la imagen antigua.

Tests: 6 nuevos en `tests/test_contrato.py` (líneas del POST, faltas
del aviso, alineación del splash, regresión de `rojo_local`); suite
**112/112** (106 + 6).

## Hoja de ruta

* **Fase 1.5**: almacenamiento real en el pendrive montado — leer y
  escribir datos del dispositivo una vez aceptado, con trazado de su
  I/O por decisión propia.
* Más dispositivos por el mismo conector (identificar cuál llegó).
* Red activada como segunda fuente de eventos externos.
