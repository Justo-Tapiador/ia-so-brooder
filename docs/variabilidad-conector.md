# Variabilidad del conector — el fix OOD (v0.4.0)

> v0.4.0 · `brooder diagnostico` · la reencubación de 63 segundos

## El agujero

Hasta v0.3.0, el conector USB solo existía en los episodios
`DISPOSITIVO`: las cinco tareas clásicas (ECO, SUMA, GUARDAR,
RECORDAR, AVISO) nacían **siempre con el conector vacío**. El
cerebro incubado jamás vio los canales `[22]`–`[25]` de su
percepción (conectado / montado / puntero / escrituras) activos
durante una clásica. En términos de aprendizaje: una distribución de
entrada que nunca visitó — **fuera de distribución (OOD)**.

La medición con el cerebro publicado v0.3.0 (60 solicitudes por
tarea y estado) lo pone en números:

| Estado del conector | ECO | SUMA | GUARDAR | RECORDAR | AVISO | media |
|---------------------|-----|------|---------|----------|-------|-------|
| vacío               | 98 % | 100 % | 100 % | 100 % | 100 % | **99,7 %** |
| conectado           | 98 % | 100 % | 100 % | 100 % | 100 % | **99,7 %** |
| **montado** (datos residuales) | 68 % | **0 %** | 18 % | 55 % | 98 % | **48 %** |

SUMA al 0 % con el pendrive montado: dos canales que siempre estuvieron
a 0,0 pasan a 1,0 — y la política se desbarranca. El kernel no miente:
responde con su `[FALLO]` legítimo (presupuesto agotado o pantalla
errónea). La honestidad del sistema intacta; la robustez, rota.

## El fix: entrenar el mundo que faltaba

`EntornoBrooder` gana `estado_conector`:

| valor | significado |
|-------|-------------|
| `None` | régimen histórico: clásicas con el conector vacío, bit a bit (mismo flujo del rng; los tests con semilla fija de v0.3.0 siguen pasando sin tocar) |
| `"aleatorio"` | **entrenamiento**: cada clásica sortea su conector — 60 % vacío / 10 % conectado / 30 % montado |
| `"vacio"` / `"conectado"` / `"montado"` | **evaluación**: fuerza un estado y mide esa distribución por separado |

El estado «montado» modela el pendrive de una sesión anterior:
reenchufado, montado, con 1–3 ranuras de datos A–Z y **el cursor donde
lo dejaron** (`conectar_dispositivo` acepta `puntero`). El puntero
inicial aleatorio es deliberado — el kernel real rebobina su registro a
0 entre solicitudes, pero entrenar «más ancho» de lo que corre obliga a
la política a **leer** el canal `[24]` en vez de asumirlo (aleatorización
de dominio, estándar en robustez de políticas).

La cirugía es limpia porque el diseño antiguo ya lo era:

* El pendrive es **neutro en recompensa** para las clásicas
  (anti-señuelo: administrar hardware solo se premia en DISPOSITIVO).
* Las condiciones de éxito de las clásicas **no tocan** el
  dispositivo (pantalla, acumulador, disco, RAM — nunca ranuras).
* El **oráculo** ignora el pendrive en las clásicas: el test de
  resolubilidad al 100 % sigue pasando con variabilidad activa
  (`test_oraculo_resuelve_todo_con_variabilidad`).
* Los episodios DISPOSITIVO quedan **inmunes**: el pendrive llega por
  contrato de la solicitud, no por azar.

## El gate de invarianza

La incubadora ya no se cree el 99 % de una sola distribución.
`ConfiguracionPPO.conector_variable` (por defecto `True`) cambia dos
cosas:

1. **Entrena** con la mezcla 60/10/30 (`estado_conector="aleatorio"`).
2. **Evalúa cada clásica en los tres estados** y fusiona por
   mínimo: la cifra que ve el currículo es la del **peor** estado.
   Avanzar de etapa o converger exige resolver ECO igual de bien con el
   pendrive montado que con el conector vacío.

El trazado del registro se fusiona igual (mínimo entre estados), así
que la convergencia sigue exigiendo «resolver Y trazar» — ahora en
todos los mundos. `brooder incubar --conector-fijo` apaga todo el
mecanismo para reproducir el régimen de v0.3.0.

El registro de métricas (`metricas.jsonl`) y el resumen final ganan
`exito_eval_estado`: el detalle por tarea y estado, por si quieres
ver el agujero cerrándose en directo.

## El diagnóstico lo enseña

`brooder diagnostico` añade el bloque de invarianza — y con la imagen
v0.3.0 muestra el agujero tal cual era:

```text
Invarianza del conector (clásicas por estado del pendrive):
  ✔ vacio      media clásicas 99 %
  ✔ conectado  media clásicas 99 %
  ✘ montado    media clásicas 45 %

Veredicto: dominio parcial: falla la invarianza del conector (OOD)
```

Tras la reencubación, el mismo comando devuelve los tres estados al
verde y `DOMINIO COMPLETO` vuelve a significar lo que dice.

## La reencubación real (63 segundos)

Fine-tuning PPO desde el cerebro publicado v0.3.0 (`montar_ssd` →
`Incubadora`, sin trasplante: el contrato 26×23 no cambia), etapa 5,
gates activos, semilla 1234:

| | pasos | tiempo | resultado |
|--|-------|--------|-----------|
| reencubación | **20.585** | **63 s** (CPU) | convergido: invarianza + trazado |

La curva cuenta el cuento completo: en el primer checkpoint (paso
~10k) SUMA **cae al 15 %** en evaluación con el conector montado — la
entropía de exploración hace su trabajo re-muestrear la distribución
nueva — y la consolidación posterior la devuelve al 100 %. Pasar por
el valle y salir no es un accidente: es PPO funcionando con gates.

Resultado final (mínimo por tarea entre estados, 60 solicitudes/tarea):

| | vacío | conectado | montado | dispositivo |
|--|-------|-----------|---------|-------------|
| v0.3.0 | 99,7 % | 99,7 % | **48 %** | 100 % |
| v0.4.0 | **100 %** | **99,7 %** | **100 %** | **100 %** |

Trazado del registro (mínimo entre estados): GUARDAR 91 %,
RECORDAR 91 %, DISPOSITIVO 73 % — en el régimen del gate (≥ 70 %) y
sin regresión material frente a v0.3.0 (GUARDAR 100 %, RECORDAR 96 %,
DISPOSITIVO 74 % medidos solo en vacío).

`ssd/brooder.img` queda reemplazada por la imagen reencubada (la
vieja sigue disponible en el historial git, etiqueta `v0.3.0`).

## Archivos tocados

| archivo | cambio |
|---------|--------|
| `brooder/entorno.py` | constantes de la mezcla, `estado_conector`, `_enchufar_conector_clasica`, docstrings |
| `brooder/primitivas/base.py` | `conectar_dispositivo(contenido, puntero)`: el cursor no se rebobina |
| `brooder/incubadora.py` | `conector_variable`, `evaluar(estado_conector=...)`, `_evaluar_etapa` por estados + fusión mínima, métricas por estado |
| `brooder/cli.py` | `incubar --conector-fijo`, bloque de invarianza en `diagnostico`, resumen de invarianza en `incubar` |
| `tests/test_conector_variable.py` | 12 tests: régimen histórico intacto, mezcla, residuales, inmunidad DISPOSITIVO, oráculo con variabilidad, gate de invarianza, diagnóstico |
| `ssd/brooder.img` | imagen reencubada (v0.4.0) |

## Lecciones

* **El 100 % de una sola distribución no es el 100 %.** El cerebro
  v0.3.0 parecía perfecto porque el mundo de entrenamiento era una
  habitación con una sola luz encendida.
* **Cerrar un agujero OOD puede ser barato.** 20.585 pasos y 63 s de
  CPU bastaron: la política ya sabía hacer todo; solo tenía que
  aprender a ignorar (o leer) cuatro canales que nunca había visto
  activos.
* **La honestidad primero.** El kernel respondió «fallo» durante todo
  el agujero — nunca simuló éxito. Por eso el fix es una reencubación
  (cambiar la política) y no un parche al kernel (esconder el
  síntoma).
* **Gates o no hay historia.** Sin la evaluación por estados, la
  reencubación habría «convergido» igual — y el agujero seguiría ahí,
  ahora con más pasos de entrenamiento encima.
