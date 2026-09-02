# El emulador web — la consola de IA-SO Brooder en el navegador

> Fase 2 · `py -m brooder servidor` · desde v0.3.0

## Qué es

Un servidor web **dentro del propio repo** que sirve una consola idéntica
a la de máquina real. Para probar IA-SO Brooder sin instalar nada más que
el repo clonado: cualquier navegador (móvil incluido) se convierte en el
teclado y el monitor de la IA-SO.

```
navegador ──HTTP/JSON──> servidor (ThreadingHTTPServer, 100 % stdlib)
                              │
                              ▼
                    SesionInteractiva  ← brooder/sesion.py
                    (el MISMO núcleo, la MISMA imagen SSD y el MISMO
                     sandbox que usa `brooder arrancar`)
```

La clave del diseño: **el emulador no es una máquina nueva — es un
teclado y una pantalla remotos**. La sesión vive en `brooder/sesion.py`,
compartida por el CLI (imprime en vivo) y el servidor (envía las mismas
líneas por JSON). No puede existir una consola web «desincronizada» de
la real porque ambas son la misma ruta de código: el test
`test_consola_web_igual_que_la_clasica` lo certifica byte a byte.

## Cómo se lanza

Desde la raíz del repo (ahí viven `config.json`, `ssd/` y el sandbox):

```
py -m brooder servidor
```

El servidor muestra qué está sirviendo y la URL de la consola
(`http://127.0.0.1:7800/` por defecto). `Ctrl+C` lo detiene. También
puede sobrescribirse en caliente:

```
py -m brooder servidor --puerto 8080        # solo esta vez
py -m brooder servidor --host 0.0.0.0       # ⚠ visible desde tu LAN
```

## config.json

Vive en la raíz del repo. La validación es **honesta** con la misma
convención que `pendrive.json`: un campo inválido se sustituye por su
valor por defecto y el cambio se anuncia en `avisos` (en la consola del
servidor y en el panel EMULADOR de la web). Nunca un crash.

```json
{
  "perfil": "D",
  "maquina": "virtual",
  "ssd": "ssd/brooder.img",
  "sandbox": "brooder_sandbox",
  "red": { "host": "127.0.0.1", "puerto": 7800 },
  "consola": { "tema": "cian" }
}
```

| Campo | Valores | Significado |
|---|---|---|
| `perfil` | `A`/`B`/`C`/`D` | el carácter de la máquina emulada (ver tabla) |
| `maquina` | `virtual` \| `real` | `real` usa `PCReal` con sandbox en disco: **el pendrive recuerda entre apagados, también desde el navegador** |
| `ssd` | ruta | qué imagen SSD se monta al arrancar |
| `sandbox` | ruta | raíz del sandbox de la máquina real |
| `red.host` | IP | `127.0.0.1` (solo tú) por defecto |
| `red.puerto` | 1..65535 | `7800` por defecto |
| `consola.columnas/filas` | 40–200 / 12–60 | tamaño del monitor (lo fija el perfil; sobreescribible) |
| `consola.tema` | `cian` \| `verde` \| `ambar` | fósforo del CRT |
| `consola.retardo_post_ms` | 0–2000 | cadencia de las líneas del POST |
| `consola.panel` | bool | mostrar el monitor del sistema lateral |

### Los perfiles (la tabla de hardware del README)

| Perfil | Monitor | POST | Extras |
|---|---|---|---|
| **A · PC reciclado** | 80×25 | 120 ms/línea — se siente el hardware viejo | — |
| **B · Sobremesa equilibrado** | 100×30 | 40 ms/línea | — |
| **C · Incubadora** | 120×32 | sin retardo | panel de métricas del cerebro (éxitos/trazado del manifiesto SSD) |
| **D · SSD viajero** | 100×30 | 60 ms/línea | el perfil natural: cualquier PC con Python 3.9+ lo hospeda |

El perfil simula el **carácter** de la máquina (monitor, cadencia,
métricas), no el rendimiento real: la inferencia corre donde corra el
servidor. Es la honestidad de siempre.

## Emular la historia del proyecto

La emulación «desde la primera feature hasta la última» se consigue por
donde toca: **el kernel es acumulativo y las imágenes SSD son
intercambiables**. Apunta el config a una imagen antigua:

```json
"ssd": "ssd/brooder-fase0.img"
```

y el POST mostrará el `[ AVISO ]` del guardián de contrato: la placa
nueva (kernel 26×23) encendiendo el cerebro de la Fase 0 (21×17), con
las primitivas y canales que le faltan visibles en el arranque. No es
una reconstrucción falsa: es cómo se comporta el hardware real con una
pieza antigua. Las imágenes históricas están en el historial git del
repo (`git log -- ssd/`).

## La API (para quien quiera su propio cliente)

| Ruta | Método | Qué hace |
|---|---|---|
| `/` | GET | la consola web (`brooder/web/consola.html`) |
| `/api/config` | GET | perfil, monitor, tema, avisos |
| `/api/estado` | GET | encendida + panel del monitor (diagnóstico del núcleo) |
| `/api/arrancar` | POST | enciende la IA-SO: POST + banner por JSON |
| `/api/linea` | POST `{"texto": "3+5"}` | atiende una línea (comando o solicitud) |
| `/api/apagar` | POST | botón de apagado (equivale a `:salir`) |

## Seguridad

* **Loopback por defecto** (`127.0.0.1`), coherente con la política «Red
  desactivada por seguridad» del POST. Exponer a LAN (`0.0.0.0`) exige
  editarlo a mano y se anuncia con alerta roja al arrancar.
* **Sin ejecución arbitraria**: `/api/linea` acepta una línea de texto
  que pasa por `Solicitud.desde_texto` igual que el teclado real. Cero
  `exec`, cero shell.
* **Sin archivos servidos**: la única ruta estática es la propia consola
  (vive dentro del paquete, `brooder/web/`). Nada del sandbox ni del
  repo se expone.
* La imagen SSD se carga con `weights_only=True` (el guard de seguridad
  de la Fase 0) — una imagen manipulada no ejecuta código.

## Límites honestos

* **Una sesión única** (una consola = una máquina física): el lock del
  emulador serializa el acceso. Dos pestañas comparten la MISMA máquina
  — lo que una teclea, la otra lo ve. Multi-sesión es una fase futura.
* **`:recovery` sin input anidado**: en la web, el menú se imprime y la
  SIGUIENTE línea es la opción elegida (modo recovery). El CLI mantiene
  su menú interactivo clásico.
* El rendimiento de la inferencia es el del servidor, no el del
  navegador.
