<div align="center">
<p><img src="docs/ia-so-brooder-2.png" alt="IA-SO BROODER"></p>

# IA-SO BROODER

### *Un sistema operativo que se incuba, no se instala.*

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Licencia](https://img.shields.io/badge/licencia-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

**Una red neuronal aprende, por refuerzo puro, a administrar un ordenador.
Se entrena en una máquina «incubadora», se exporta como imagen SSD
y «nace» en otro PC, donde atiende solicitudes gestionando CPU, RAM,
disco, pantalla y audio a través de un contrato de primitivas.**

**Autor:** [Justo Tapiador García](mailto:justo.tapiador@gmail.com)
**Idea original junto a:** Albert Zotkin

English version: **[README.md](README.md)**

</div>

---

## Índice

1. [¿Qué es esto?](#qué-es-esto)
2. [Resultados](#resultados)
3. [Arranque rápido](#arranque-rápido)
4. [El emulador web — la IA-SO en el navegador](#el-emulador-web--la-ia-so-en-el-navegador)
5. [El SSD de nacimiento: hardware recomendado](#el-ssd-de-nacimiento-hardware-recomendado)
6. [Seguridad: primitivas, no hardware](#seguridad-primitivas-no-hardware)
7. [Arquitectura](#arquitectura)
8. [Cómo aprende Brooder](#cómo-aprende-brooder)
9. [Hoja de ruta](#hoja-de-ruta)
10. [Autores y reconocimiento](#autores-y-reconocimiento)
11. [Licencia](#licencia)


## ¿Qué es esto?

IA-SO Brooder es un experimento de **sistema operativo gobernado por una
red neuronal**. Nació de una conversación entre dos amigos:

> «¿Y si una IA, una vez entrenada, se almacenara en un SSD y al arrancar
> el PC se comportara como un verdadero sistema operativo, esperando
> entradas de sus periféricos para producir salidas?»

La pieza maestra del diseño: **la IA jamás accede directamente al
hardware**. En su lugar recibe un conjunto de **interfaces primitivas**
— `leer_teclado()`, `escribir_disco()`, `usar_cpu()`,
`mostrar_en_pantalla()`... — y **aprende por sí misma** a utilizarlas.
Así se evita el problema monstruoso de «cómo hacer que una red neuronal
entienda todas las instrucciones de una CPU moderna».

```
   INCUBADORA                SSD                  PC DE NACIMIENTO
┌─────────────────┐     ┌───────────┐      ┌────────────────────────┐
│  máquina potente│     │  brooder  │      │  CPU + GPU             │
│                 │     │  .img     │      │                        │
│  PPO + currículo│────►│ (262 KiB) │─────►│  BIOS → núcleo → IA-SO │
│  ~14 min en CPU │     └───────────┘      │                        │
└─────────────────┘    exportar / montar   │  ECO · SUMA · GUARDAR  │
     entrenar                             │  RECORDAR · AVISO      │
                                          └────────────────────────┘
```

---

## Resultados

El cerebro incluido en `ssd/brooder.img` es el mismo de la Fase 0.5,
**trasplantado** al contrato nuevo (percepción 21→24 entradas,
primitivas 18→20 salidas: ver `docs/dispositivo-virtual.md`), refinado
hasta 1.975.637 pasos totales de PPO y **reencubado en v0.4.0 con
variabilidad del conector** (+20.585 pasos: ver
`docs/variabilidad-conector.md`). Evaluación determinista sobre
solicitudes nuevas, verificadas por el propio hardware — ahora en
cada estado del conector USB:

| Tarea | ¿Qué debe hacer Brooder? | Éxito |
|-------|--------------------------|-------|
| `ECO` | leer el teclado y repetirlo en pantalla | **100 %** |
| `SUMA` | leer dos dígitos, sumarlos **con la CPU** y mostrar el resultado | **100 %** |
| `GUARDAR` | almacenar un valor en el **disco** y recuperarlo | **100 %** (trazado 91 %) |
| `RECORDAR` | igual, pero en la **RAM** | **100 %** (trazado 91 %) |
| `AVISO` | mostrar un carácter y **pitar** al leer la alarma | **100 %** |
| `DISPOSITIVO` | montar/desmontar el **pendrive virtual** según su estado | **100 %** |

**Invarianza del conector (v0.4.0):** las clásicas se resuelven al
100 % con el pendrive **vacío, enchufado o montado con datos
residuales** — el mínimo entre estados, no la media (la v0.3.0 caía al
48 % con el pendrive montado; ver
[`docs/variabilidad-conector.md`](docs/variabilidad-conector.md)).

Nada de esto está programado a mano: **la red neuronal decide la
secuencia de primitivas en cada ciclo**. Ni siquiera imita el «programa
ideal»: en GUARDAR, por ejemplo, Brooder inventó su propia variante
legal.

![Curva de incubación](img/curva_recompensa.png)

---

## Arranque rápido

```bash
# clonar e instalar (Python 3.9+)
git clone https://github.com/Justo-Tapiador/ia-so-brooder.git
cd ia-so-brooder
pip install -e ".[dev]"

# probar el sistema con el cerebro ya incubado (7/7 solicitudes)
brooder demo

# encender la IA-SO y hablar con ella (sesión interactiva)
brooder arrancar

#   brooder> HOLA            <- eco
#   brooder> 3+5             <- suma con la CPU (resultado: 8)
#   brooder> guardar 4 G     <- escribe 'G' en la ranura 4 del disco y la recupera
#   brooder> recordar 2 Z    <- igual, pero en RAM
#   brooder> aviso A         <- muestra 'A' y pita al leer la alarma
#   brooder> :pendrive       <- enchufa el pendrive (hot-plug del mundo)
#   brooder> montar          <- la IA monta el pendrive del conector
#   brooder> escribir 3 P    <- guarda 'P' en la ranura 3 del pendrive y la recupera
#   brooder> desmontar       <- extracción segura (el dato viaja con el medio)
#   brooder> :recovery       <- menú de emergencia, independiente de la IA

# (opcional) incubar tu propio cerebro y arrancarlo
brooder incubar              # ~14 min en CPU; las 6 tareas con variabilidad del conector
brooder exportar             # empaqueta entrenamiento/mejor.pt en ssd/brooder.img
brooder demo                 # ahora la IA es tuya

# (opcional) sobre TU sistema de archivos, con sandbox
brooder demo --maquina-real  # el disco de Brooder son archivos de verdad
ls brooder_sandbox/disco/    # 0.tok ... 9.tok: la ranura 4 contiene 'G'
```

> El repositorio incluye la imagen SSD ya incubada (`ssd/brooder.img`,
> 262 KiB): puedes arrancar la IA-SO sin entrenar nada.

### Windows: caracteres `?` en la demo o UnicodeEncodeError

PowerShell 5.1 conecta la salida de los procesos externos mediante una
tubería, por lo que Python usa la página de código ANSI (cp1252) en lugar
de UTF-8. Ahí los caracteres de caja (`─`, `│`, `✔`) no existen. Desde el
hotfix de consola el CLI los degrada a `?` en vez de romperse, así que la
demo termina siempre. Para verla con su formato completo, ejecuta antes:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
```

(También puedes lanzar `brooder demo` desde `cmd.exe`, que usa la API
Unicode de la consola y no necesita nada.)

---

## El emulador web — la IA-SO en el navegador

Sin máquina donde conectar la imagen SSD, o con ganas de enseñar el
proyecto desde cualquier dispositivo: el repo incluye un **emulador
web** que sirve la MISMA consola de máquina real en el navegador.

```bash
brooder servidor                      # desde la raíz del repo
# Consola lista: http://127.0.0.1:7800/
```

```bash
brooder> (en el navegador, con el POST animado, el prompt cian y los
          mismos veredictos [ OK ]/[FALLO] que en el terminal)
```

* **Idéntica por construcción**: la sesión vive en `brooder/sesion.py`
  y la comparten el CLI y el servidor — no hay dos consolas que
  desincronizarse. El test de equivalencia lo certifica byte a byte.
* **100 % stdlib** (`http.server` + `json` + `threading`): cero
  dependencias nuevas.
* **`config.json`** en la raíz: perfil de la máquina emulada (A·PC
  reciclado 80×25 con POST perezoso, B·equilibrado, C·incubadora con
  panel de métricas, D·SSD viajero), tamaño del monitor, tema del CRT,
  máquina virtual o real (con `real`, **el pendrive recuerda entre
  apagados también desde el navegador**) e imagen SSD a montar —
  apuntando a `ssd/brooder-fase0.img` se ve al guardián de contrato
  arrancar el cerebro de la Fase 0 sobre la placa actual.
* **Seguridad de siempre**: escucha solo en `127.0.0.1` por defecto
  (como el «Red desactivada por seguridad» del POST); exponerlo a la
  LAN es decisión explícita y se anuncia con alerta.

Ver [docs/servidor-web.md](docs/servidor-web.md) para la referencia
completa del `config.json`, la API JSON y los límites honestos
(sesión única, recovery por líneas).

## El SSD de nacimiento: hardware recomendado

En la metáfora de Brooder, la incubadora exporta el cerebro a un SSD y
el PC de nacimiento lo monta al arrancar. Esta sección es la guía
práctica: qué SSD comprar, cómo instalarlo físicamente y qué
configuraciones de sobremesa recomendamos.

### Qué es exactamente `brooder.img`

`ssd/brooder.img` (262 KiB) empaqueta el cerebro entrenado, su estado
persistido y un manifiesto con las métricas de incubación. En la
versión actual **no es una imagen arrancable a nivel de BIOS**: el
«cargador de arranque» de la IA-SO es el núcleo de Brooder
(`brooder arrancar`), que se ejecuta sobre el sistema operativo del PC
de nacimiento. El SSD es el medio de transporte y la casa del cerebro;
el arranque directo (BIOS → cerebro, sin SO anfitrión) figura en la hoja
de ruta.

Consecuencia práctica: **sirve cualquier SSD y cualquier formato** que
el PC pueda leer. La velocidad es irrelevante — 262 KiB se leen en un
instante incluso por SATA —: elige por factor de forma y comodidad, no
por benchmark.

### Qué tipo de SSD elegir

| Tipo | Conexión al PC | Velocidad | Cuándo elegirlo |
|------|----------------|-----------|-----------------|
| **M.2 NVMe** (PCIe 4.0/5.0) | ranura M.2 de la placa | 3.500–14.000 MB/s | PC moderno con ranura M.2 libre: la opción por defecto |
| **M.2 NVMe** (PCIe 3.0) | ranura M.2 | ~2.400 MB/s | placas de ~2016–2020 |
| **M.2 SATA** | ranura M.2 | ~550 MB/s | placas con ranura M.2 pero sin NVMe |
| **2,5" SATA III** | cable de datos + alimentación SATA | ~550 MB/s | la opción universal: vale para cualquier torre |
| **Externo USB 3.2** o caja NVMe | puerto USB-A/USB-C | 400–1.000+ MB/s | mover el cerebro entre varios PCs («SSD viajero») |

**¿Capacidad?** El cerebro ocupa menos de 1 MiB, así que 120 GB bastan
si solo transportará imágenes. Recomendamos **250–500 GB**: el mismo
disco alojará con holgura los checkpoints de futuras re-incubaciones y,
si quieres, un sistema operativo completo dedicado al PC de nacimiento.

### Cómo instalarlo físicamente (PC de sobremesa)

- **Si es M.2 (NVMe o SATA):** apaga el PC, desenchufa la corriente y
  descarga la electricidad estática tocando la caja metálica. Localiza
  la ranura M.2 (entre el zócalo de la CPU y las ranuras PCIe; consulta
  el manual porque en algunas placas comparte líneas con puertos SATA).
  Inserta el módulo (formato 2280) en ángulo de ~30°, sin forzar, y
  fíjalo con su tornillo. No lleva cables. Si la ranura trae disipador,
  recolócalo.
- **Si es 2,5" SATA:** mismas precauciones de apagado y antiestática.
  Monta el disco en una bahía de 3,5" (con su adaptador) y conecta el
  **cable de datos SATA** a la placa y el **cable de alimentación SATA**
  desde la fuente — ambos solo entran en una posición, nunca los
  fuerces. Enciende y comprueba en la BIOS/UEFI que el disco aparece
  (modo AHCI, no RAID).
- **Si es externo USB:** conéctalo a un puerto **trasero** USB 3.0 o
  superior (van directos a la placa; evita los hubs). Es la opción más
  rápida y la única que no requiere destornillador.

### Formato del disco y copia de la imagen

El SSD solo contiene archivos, así que el formato es libre. **exFAT**
es el recomendado si el disco viajará entre Windows, Linux y macOS (es
el «idioma común» de los tres); **ext4** o **NTFS** si vivirá siempre en
el mismo sistema.

```bash
# Linux (ajusta la ruta de montaje a tu equipo)
cp ssd/brooder.img /run/media/$USER/BROODER_SSD/
brooder arrancar --ssd /run/media/$USER/BROODER_SSD/brooder.img

# Windows (PowerShell)
copy ssd\brooder.img E:\
brooder arrancar --ssd E:\brooder.img
```

Etiqueta el disco como `BROODER_SSD` para reconocerlo de un vistazo en
cualquier sistema y verifica el montaje con:

```bash
brooder diagnostico --ssd /ruta/al/SSD/brooder.img
```

### Requisitos del PC de nacimiento

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| CPU | x86-64 con AVX2: Intel Haswell (2013) o posterior, o cualquier AMD Ryzen | Ryzen 5 / Core i5 de 2020 en adelante |
| RAM | 4 GB | 16 GB |
| GPU | **no es necesaria**: el cerebro es diminuto y su inferencia es más rápida en CPU | solo para re-incubar cerebros mayores (CUDA/MPS) |
| Disco | ~5 GB libres (Python + PyTorch) | 20 GB + el SSD de nacimiento |
| SO | Linux de 64 bits, Windows 10/11 o macOS | Ubuntu 22.04+ o Debian 12+ |
| Python | 3.9 | 3.11 / 3.12 |

Dos advertencias honestas: los binarios oficiales de PyTorch 2.x exigen
**AVX2** en x86 (un Intel anterior a 2013 se queda fuera), y una GPU
dedicada **no acelera la versión actual** — el modelo es tan pequeño
que copiarlo a la GPU cuesta más que calcularlo en CPU. La GPU solo
compensa si re-incubas.

### Cuatro configuraciones recomendadas

| Perfil | CPU | RAM | GPU | SSD de nacimiento | Precio orientativo |
|--------|-----|-----|-----|-------------------|--------------------|
| **A · PC reciclado** | Intel i5-8400 / Ryzen 5 2600 | 16 GB DDR4 | la integrada (UHD 630 / Vega) | 2,5" SATA 500 GB (Crucial MX500, Samsung 870 EVO) | 0–250 € |
| **B · Sobremesa equilibrado** | Intel i5-13400F / Ryzen 5 7600 | 32 GB DDR5 | opcional: RTX 4060 | M.2 NVMe 500 GB–1 TB (WD SN770, Samsung 990 EVO, Crucial P5 Plus) | 600–900 € |
| **C · Incubadora + nacimiento** | Intel i7-14700K / Ryzen 7 7700X | 64 GB DDR5 | RTX 4070 Super / 4080 | 2.º NVMe dedicado (Samsung 990 Pro, WD SN850X) + NVMe 1 TB de sistema | 1.500–2.500 € |
| **D · SSD viajero** | cualquier PC con Python 3.9+ | 4 GB+ | sin necesidad | SSD externo USB 3.2 Gen 2 (Samsung T7, Crucial X9) | 30–100 € |

- **A — el PC reciclado** es la configuración más fiel a la metáfora:
  revivir un equipo olvidado. La demo completa corre en segundos y cada
  interacción tarda menos de un segundo.
- **B — el sobremesa equilibrado** deja margen de sobra para tener la
  IA-SO como proyecto fijo.
- **C — incubadora y nacimiento en la misma máquina**: una sola torre
  que primero entrena y después recibe al cerebro. v1 incubó en ~14 min
  usando solo la CPU; la GPU cobrará sentido con los cerebros mayores de
  la hoja de ruta.
- **D — el SSD viajero** es el cerebro en el bolsillo: cualquier equipo
  con Python y PyTorch se convierte en PC de nacimiento durante unos
  minutos. Solo requiere `brooder arrancar --ssd /ruta/al/disco/brooder.img`.

---

## Seguridad: primitivas, no hardware

Brooder percibe la máquina a través de una fotografía inocua del estado
(`InstanteMaquina`) y actúa emitiendo **solicitudes de primitiva**. El
**núcleo** — código de confianza, no la red — valida cada solicitud y la
ejecuta. Como los `syscalls` de un kernel clásico, el contrato incluye
**tipos de argumento**:

| ID | Primitiva | Argumento | Efecto |
|----|-----------|-----------|--------|
| 0  | `nada()` | — | ciclo en vacío |
| 1  | `leer_teclado()` | — | deposita el siguiente token en el bus de datos |
| 2  | `mostrar_en_pantalla(bus)` | **bus** | escribe en pantalla el carácter del bus |
| 3  | `cpu_poner(bus)` | **bus** | acumulador ← valor del bus |
| 4  | `cpu_sumar(bus)` | **bus** | acumulador += valor del bus |
| 5  | `cpu_cociente()` | — | bus ← acumulador // 10 |
| 6  | `cpu_resto()` | — | bus ← acumulador % 10 |
| 7  | `leer_cpu()` | — | bus ← acumulador |
| 8  | `mover_cabezal_disco(d)` | dirección 0-9 \| bus | posiciona el cabezal del disco |
| 9  | `leer_disco()` | — | bus ← disco[cabezal] |
| 10 | `escribir_disco(bus)` | **bus** | disco[cabezal] ← bus |
| 11 | `mover_puntero_memoria(d)` | dirección 0-9 \| bus | posiciona el puntero de RAM |
| 12 | `leer_memoria()` | — | bus ← RAM[puntero] |
| 13 | `escribir_memoria(bus)` | **bus** | RAM[puntero] ← bus |
| 14 | `reproducir_audio(f)` | libre | emite un pitido |
| 15 | `usar_gpu()` | — | compone un frame nuevo (vacía la pantalla) |
| 16 | `leer_red()` | — | **desactivada** en v1 (error controlado) |
| 17 | `registrar_log(m)` | mensaje 0-10 | añade una entrada al registro del sistema (consola del kernel) |
| 18 | `montar_dispositivo()` | — | monta el pendrive del conector USB virtual |
| 19 | `desmontar_dispositivo()` | — | extracción segura: libera el pendrive montado |

*(Ejecuta `brooder primitivas` para ver esta tabla en tu terminal.)*

`registrar_log` es la primera **macro-primitiva** y el germen de las
«syscalls» de Brooder: acciones de nivel de sistema que la IA decide y
el núcleo ejecuta como código de confianza. El registro es un anillo a lo
*dmesg* —retiene las últimas 8 entradas y persiste entre solicitudes— y su
vocabulario de mensajes es cerrado (11 ids de la tabla `MENSAJES_LOG`): la
IA elige *qué* evento declarar, nunca dicta texto libre. Desde la
**Fase 0.5** el cerebro incubado *traza por sí mismo*: el entorno premia
declarar cada escritura/lectura de disco o RAM con el mensaje correcto
en el momento oportuno, y la sección
`REGISTRAR_LOG — decisión propia del cerebro` de la demo muestra los
eventos que la política neuronal emitió durante las solicitudes reales.
Los cerebros incubados con contratos anteriores (17 o 18 salidas)
siguen montando sin cambios (la demo detecta el contrato y muestra la
vía sintética del núcleo; ver `docs/registro-sistema.md`).

**Fase 1 — el pendrive virtual.** Las primitivas 18 y 19 administran un
dispositivo externo que aparece y desaparece *en caliente*: el mundo
exterior conecta/desconecta el pendrive (no hay primitiva para ello —
la IA no puede enchufar hardware, solo administrar lo que hay) y la
política percibe el conector en tres canales nuevos de su observación
(presencia, montado, solicitud de dispositivo). El ciclo de vida queda
anotado por el kernel en su registro como un *dmesg* real: montar y
desmontar dejan INFO; si el mundo retira el pendrive **montado**, el
kernel registra el ERROR «extraccion insegura» por sí mismo. La sección
`DISPOSITIVO EXTERNO` de la demo muestra el ciclo completo — decisión
propia del cerebro incluida. La sesión interactiva lo permite a mano:
`:pendrive` enchufa/retira, y las solicitudes `montar` / `desmontar` se
las planteas a la IA. Diseño y compatibilidad de contratos:
`docs/dispositivo-virtual.md`.

**Hotfix de contrato (Fase 1, post-entrega).** Nació de un tropiezo
real en campo: aplicar el parche de la Fase 1 sin copiar su imagen SSD
deja montado un cerebro del contrato viejo (21×18) en un kernel que
habla 24×20. El arranque es LEGAL —la compatibilidad de prefijo lo
garantiza— pero `montar` fallaba en silencio: las cabezas del cerebro
nunca emiten ids ≥ n_primitivas. Ahora el POST declara el contrato
montado (`[ OK ] Cerebro contrato 26x23` / `[ AVISO ] … (imagen
antigua)`), y `arrancar`, `demo` y `diagnostico` explican el desfase y
su remedio antes de que aparezca el primer [FALLO]. De regalo, se
reparó `rojo_local`: faltaba desde el primer commit y
`brooder diagnostico` crasheaba en cuanto una tarea caía por debajo
del 85 % (rama ✘ nunca ejecutada hasta entonces). Suite: 112/112
(106 + 6 tests nuevos en `tests/test_contrato.py`).

**Fase 1.5 — almacenamiento real: el pendrive recuerda.** El
pendrive montado deja de ser un objeto hueco: gana un plano de datos
propio (primitivas 20/21/22 — mover puntero, leer y escribir contra
sus 8 ranuras —, espejo del par disco/RAM y **solo utilizable con el
dispositivo montado**, como un USB real) y un **trazado I/O propio**:
cada lectura/escritura deja su entrada en el anillo del propio medio
(`[0010] E ranura[3] <- 'Q'`), separado del dmesg del kernel. Las
ranuras viven EN EL DISPOSITIVO: sobreviven a desmontar y a que el
mundo retire y reenchufe el pendrive (extracción segura) — **el
pendrive recuerda** lo que se escribió en él — y se pierden con la
extracción insegura, como el búfer sin sincronizar de un USB físico.
En `--maquina-real` el pendrive es el archivo
`brooder_sandbox/pendrive.json`: lo que se escribe ahí **sobrevive a
apagar la IA-SO y volver a encenderla**. La tarea DISPOSITIVO estrena
los modos `escribir 3 P` / `leer 3 P` con éxito verificable y a prueba
de adivinanzas (en `leer`, el valor nunca pasa por el teclado: la
única fuente es el propio medio). El espacio entre número y letra es
opcional — `leer 3P` es la misma solicitud que `leer 3 P`, como `3+5`
ya lo era para la suma (hotfix de campo tras la primera sesión real).
Contrato: 24×20 → **26×23** (la
compatibilidad de prefijo y el guardián de contrato del hotfix lo
cubren: un cerebro Fase 1 arranca y recibe su AVISO con las faltas
exactas). Suite: **137/137** (112 + 23 de la fase + 2 del hotfix del
parser, en `tests/test_almacenamiento.py`).

El **bus de datos** es el corazón del diseño: las primitivas de datos
solo aceptan el valor que ya ha sido leído en el bus. No se puede
«mostrar» lo que no se ha leído: el flujo de datos es real, como en una
máquina de verdad.

Consecuencias de seguridad:

- **Sin ejecución de código**: no existe primitiva para ejecutar nada.
- **Sandbox real**: en `--maquina-real`, el disco son los archivos
  `0.tok`…`9.tok` de un directorio fijo y el «cabezal» es un entero 0-9
  validado: no hay rutas arbitrarias ni escritura fuera del sandbox.
- **Red desactivada** por política en v1.
- **Recovery independiente**: si el cerebro falla, el núcleo sigue vivo
  y ofrece `:recovery` (reiniciar IA, estado, diagnóstico, apagar).

### Integridad del cerebro distribuido

`ssd/brooder.img` viaja por la red cada vez que alguien clona este
repositorio. Como un archivo `.pt` es un pickle de Python, montar una
imagen manipulada podría, en principio, ejecutar código en el arranque.
Brooder impone la política contraria: todos los `torch.load` del código
se ejecutan con `weights_only=True` (solo tensores y tipos básicos), y
`brooder exportar` empaqueta en la imagen **únicamente** `config` +
pesos — el estado del optimizador nunca viaja. Una imagen manipulada,
como mucho, no carga; no puede ejecutar. Hay tests de regresión en
`tests/test_seguridad.py` y la política completa está en
[SECURITY.md](SECURITY.md).

Verifica la imagen antes de arrancarla:

```bash
sha256sum ssd/brooder.img
# 5e9e66cbfa9679c7d62eaf3bfc8537fa22fbd3c138227110fadee5236d6b3590  brooder.img
```

*(Actualiza ese hash si regeneras la imagen con `brooder exportar`;
cada release de GitHub publica también el hash de la imagen adjunta.)*

---

## Arquitectura

```
                ┌─────────────────────────────────────────────┐
                │                SOLICITUD                    │
                │  (usuario o entorno de entrenamiento)       │
                └──────────────────────┬──────────────────────┘
                                       │ tokens por teclado
                                       ▼
 ┌────────────┐   observación    ┌──────────────────────┐
 │ PERCEPCIÓN │◄─────────────────│       NÚCLEO         │
 │ (21 canales:│                 │ (código de confianza:│
 │ teclado, bus,│                │ valida y ejecuta)    │
 │ CPU, disco…)│                 └─────────┬────────────┘
 └──────┬─────┘                           │ primitiva + argumento
        │                                 ▼
        │                       ┌──────────────────────┐
        │  ciclo percibir →     │  MÁQUINA (virtual o  │
        │  decidir → actuar     │  real con sandbox)   │
        ▼                       │  CPU RAM disco       │
 ┌─────────────────────────┐    │  pantalla audio GPU  │
 │    CEREBRO (Brooder)    │    └──────────────────────┘
 │ GRU + memoria asociativa│
 │ (8 ranuras × 16 dim,    │
 │  direccionable por      │
 │  contenido) + cabezal   │
 │  de argumentos          │
 └─────────────────────────┘
```

| Módulo | Responsabilidad |
|--------|-----------------|
| `brooder/constantes.py` | vocabulario de tokens, tareas, tabla de primitivas, máscaras de tipos |
| `brooder/primitivas/` | el contrato de hardware: máquina virtual y máquina real (sandbox) |
| `brooder/percepcion.py` | el vector de 21 canales — idéntico en entrenamiento y producción |
| `brooder/solicitudes.py` | las peticiones del usuario y sus condiciones de éxito verificables |
| `brooder/entorno.py` | el entorno de entrenamiento con recompensas + la **política oráculo** |
| `brooder/cerebro.py` | GRU + memoria asociativa + cabezas PPO |
| `brooder/incubadora.py` | PPO con currículo por etapas, entropía adaptativa y curiosidad |
| `brooder/nucleo.py` | BIOS/POST, ciclo percibir→decidir→actuar, montaje/exportación SSD |
| `brooder/estado.py` | registro persistente de lo que la IA-SO ha vivido |
| `brooder/pantalla.py` | TUI ANSI: pantalla de Brooder, monitor del sistema, recovery |
| `brooder/cli.py` | `incubar · exportar · arrancar · demo · diagnostico · graficar · primitivas` |

---

## Cómo aprende Brooder

**PPO** (Proximal Policy Optimization) sobre un entorno virtual que
genera miles de solicitudes aleatorias y devuelve recompensas o
penalizaciones. Nada está pre-programado: la red descubre las secuencias
de primitivas. Tres decisiones de diseño marcaron la diferencia entre
«no aprende nada» y «dominio completo»:

1. **Currículo por etapas.** Primero solo ECO (el «hola mundo»
   operativo: leer → mostrar). Al superarlo se añade SUMA, luego
   GUARDAR/RECORDAR, luego AVISO. Cada etapa conserva las tareas
   anteriores (retención).
2. **Recompensas que no castigan la exploración.** Castigar el «fallo
   exprés» creaba un atajo perverso; castigar escrituras de disco en
   tareas de eco creaba un sesgo anti-disco que luego impedía aprender
   GUARDAR. La política final **premia el progreso** (leer datos,
   producir salida, posicionar el cabezal, el carácter correcto) y deja
   que el coste de ciclo y el veredicto final ordenen el resto.
3. **Curiosidad con autoextinción** (bonus 1/√n). Sin ella, una política
   que ya resuelve ECO/SUMA deja de muestrear `escribir_disco` y jamás
   descubriría GUARDAR: la recompensa del disco existe, pero es
   inalcanzable sin exploración.

Además, la política opera bajo **máscaras de tipos**: el espacio de
acciones respeta la misma firma que valida el kernel (no se puede
proponer `mostrar(literal)`), igual que un programa no puede pasar un
puntero donde un syscall espera un entero.

El **test del oráculo** (`tests/test_entorno.py`) es la red de seguridad
del proyecto: la política ideal escrita a mano debe resolver el 100 % de
solicitudes aleatorias. Si el entorno estuviera mal construido, el
oráculo fallaría y el test lo delataría.

```bash
pytest -q   # 182 tests: primitivas, entorno, oráculo, cerebro, núcleo,
            # SSD, sandbox, dispositivo, trazado, sesión web y
            # variabilidad del conector (invarianza OOD)
```

---

## Hoja de ruta

- [x] ~~Fase 0: registro del sistema (`REGISTRAR_LOG`) y Fase 0.5: el
      cerebro traza por decisión propia.~~
- [x] ~~Fase 1: dispositivo externo con hot-plug (pendrive virtual,
      `montar`/`desmontar`).~~
- [x] ~~Fase 1.5: almacenamiento real en el pendrive montado —
      `escribir`/`leer` datos del dispositivo con trazado I/O propio,
      persistencia real y extracción insegura que pierde datos.~~
- [x] ~~Fase 2: emulador web — la consola de la IA-SO en el navegador
      (`py -m brooder servidor`).~~
- [x] ~~v0.4.0: fix OOD — reencubación con variabilidad del conector:
      las clásicas al 100 % con el pendrive en cualquier estado y
      gate de invarianza en incubadora y diagnóstico.~~
- [ ] Más tokens y pantallas de varias líneas; edición de pantalla.
- [ ] Un «sistema de archivos» mínimo del pendrive (múltiples tokens por
      ranura, checksum), el paso natural tras las ranuras planas.
- [ ] Tareas nuevas sobre las primitivas existentes: RESTA, cadenas de
      archivos, búsqueda en disco, alarmas periódicas.
- [ ] Red activada con paquetes reales (otra fuente de entrada).
- [ ] Memoria asociativa de mayor capacidad y persistencia entre sesiones.
- [ ] Interfaz gráfica compuesta por la GPU con ventanas generadas por
      la propia IA (el «espacio de trabajo alrededor de la tarea» de la
      conversación original).
- [ ] Múltiples monitores como extensión de la memoria de trabajo.
- [ ] Currículo auto-generado (la incubadora inventa tareas nuevas).
- [ ] Arranque directo desde el SSD sin SO anfitrión (BIOS → cerebro).

---

## Contribuir

Las contribuciones son bienvenidas, especialmente las que respeten los
tres principios del proyecto:

1. La IA no toca el hardware: solo primitivas validadas por el núcleo.
2. Lo que la IA hace bien debe ser **aprendido**, no programado.
3. Toda mejora del entorno debe pasar el test del oráculo.

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Autores y reconocimiento

**Autor:** [Justo Tapiador García](mailto:justo.tapiador@gmail.com)

**Idea original:** Justo Tapiador García y **Albert Zotkin**, en una
conversación sobre sistemas operativos, bootstrap de compiladores y qué
pasaría si una IA aprendiera a administrar una máquina en lugar de
ejecutar un programa escrito para administrarla. El primer prototipo —
un MLP con un diccionario `memory` — fue el punto de partida; IA-SO
Brooder lo convierte en un sistema completo con memoria asociativa,
percepción por canales, entorno de recompensas, incubadora PPO y
arranque real desde SSD.

Contacto: [justo.tapiador@gmail.com](mailto:justo.tapiador@gmail.com)

---

## Licencia

[MIT](LICENSE) — Copyright (c) 2026 Justo Tapiador García.
Incuba, modifica, nace, comparte.
