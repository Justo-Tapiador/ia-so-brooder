"""
La incubadora — donde Brooder aprende a administrar una máquina
================================================================

Entrenamiento por refuerzo (PPO) sobre el entorno virtual de
solicitudes. Se ejecuta en la máquina "incubadora" (la tuya, o una
con GPU: da igual) y produce los pesos que después viajan en el
SSD al PC de nacimiento.

Currículo (se avanza cuando TODAS las tareas de la etapa superan el
umbral de éxito en evaluación determinista):

  etapa 1: ECO
  etapa 2: ECO, SUMA
  etapa 3: ECO, SUMA, GUARDAR, RECORDAR
  etapa 4: ECO, SUMA, GUARDAR, RECORDAR, AVISO

Detalles de implementación dignos de mención:

* Cada solicitud es un episodio: h y M (estado del cerebro) se
  reinician por solicitud, igual que hará el núcleo en producción.
* El rollout se corta SIEMPRE en fronteras de episodio, así el GAE
  no necesita bootstrapping con episodios truncados.
* La actualización PPO re-ejecuta cada episodio completo (BPTT)
  desde su h0/M0 almacenados, y acumula gradientes por grupos de
  episodios.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from brooder.cerebro import (
    CerebroBrooder,
    enmascarar_logits_argumento,
    entropia_conjunta,
)
from brooder.constantes import CURRICULO, OBS_DIM, Tarea
from brooder.entorno import EntornoBrooder


# ------------------------------------------------------------------
# configuración
# ------------------------------------------------------------------
@dataclass
class ConfiguracionPPO:
    # 1.5M pasos resuelven las 5 tareas con margen (una incubación
    # completa tarda ~25 min en una CPU moderna; menos en GPU).
    pasos_totales: int = 1_500_000
    pasos_por_rollout: int = 1024
    epocas: int = 6
    episodios_por_grupo: int = 16
    lr: float = 1e-3
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip: float = 0.3
    coef_valor: float = 0.5
    coef_entropia: float = 0.02
    coef_entropia_exploracion: float = 0.05  # mientras haya tareas sin resolver
    umbral_exploracion: float = 0.50          # éxito mínimo para dejar de explorar
    max_grad_norm: float = 0.5
    semilla: int = 1234
    eval_cada: int = 12_000
    solicitudes_eval: int = 60
    umbral_etapa: float = 0.85
    parar_al_converger: bool = True
    imprimir_cada_rollouts: int = 4


# ------------------------------------------------------------------
# episodios
# ------------------------------------------------------------------
@dataclass
class Episodio:
    obs: list = field(default_factory=list)
    prim: list = field(default_factory=list)
    arg: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    valor: list = field(default_factory=list)
    recompensa: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def tensores(self):
        return (
            torch.tensor(self.obs, dtype=torch.float32),          # [T, D]
            torch.tensor(self.prim, dtype=torch.long),            # [T]
            torch.tensor(self.arg, dtype=torch.long),             # [T]
            torch.tensor(self.logp, dtype=torch.float32),         # [T]
            torch.tensor(self.valor, dtype=torch.float32),        # [T]
            torch.tensor(self.recompensa, dtype=torch.float32),   # [T]
        )


def _gae(recompensas, valores, gamma: float, lam: float):
    """Ventajas generalizadas para un episodio que SIEMPRE termina."""
    T = len(recompensas)
    ventajas = torch.zeros(T)
    ultima = 0.0
    for t in reversed(range(T)):
        no_fin = 0.0 if t == T - 1 else 1.0
        v_prox = valores[t + 1].item() if t + 1 < T else 0.0
        delta = (
            recompensas[t].item()
            + gamma * v_prox * no_fin
            - valores[t].item()
        )
        ultima = delta + gamma * lam * no_fin * ultima
        ventajas[t] = ultima
    retornos = ventajas + valores
    return ventajas, retornos


# ------------------------------------------------------------------
# evaluación determinista (reutilizada por el CLI diagnostico)
# ------------------------------------------------------------------
def evaluar(
    cerebro: CerebroBrooder,
    tareas,
    n_solicitudes: int = 60,
    semilla: int = 99_999,
) -> dict:
    """Mide el éxito de la política (determinista) por tarea."""
    cerebro.eval()
    entorno = EntornoBrooder(tareas_activas=list(tareas), semilla=semilla)
    for tarea in tareas:
        entorno.fijar_tareas([tarea])
        for _ in range(n_solicitudes):
            obs = entorno.reiniciar()
            h, M = cerebro.estado_inicial()
            terminada = False
            while not terminada:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                prim, arg, _, h, M = cerebro.decidir(obs_t, h, M, determinista=True)
                obs, _, terminada, _ = entorno.paso(prim, arg)
    return {t: v[0] for t, v in entorno.tasa_exito().items()}


# ------------------------------------------------------------------
# la incubadora
# ------------------------------------------------------------------
class Incubadora:
    """Entrena a Brooder con PPO y gestiona el currículo."""

    def __init__(
        self,
        cfg: ConfiguracionPPO | None = None,
        cerebro: CerebroBrooder | None = None,
        dir_salida: str | Path = "entrenamiento",
        silencioso: bool = False,
    ):
        self.cfg = cfg or ConfiguracionPPO()
        self.dispositivo = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.cerebro = (cerebro or CerebroBrooder()).to(self.dispositivo)
        self.cerebro.train()
        self.optimizador = torch.optim.Adam(
            self.cerebro.parameters(), lr=self.cfg.lr
        )
        self.dir_salida = Path(dir_salida)
        self.dir_salida.mkdir(parents=True, exist_ok=True)
        self.silencioso = silencioso

        random.seed(self.cfg.semilla)
        np.random.seed(self.cfg.semilla)
        torch.manual_seed(self.cfg.semilla)

        self.entorno = EntornoBrooder(
            tareas_activas=CURRICULO[0], semilla=self.cfg.semilla
        )
        self.etapa = 0
        self.paso_global = 0
        self.mejor_exito = -1.0
        self.convergido = False
        self._coef_entropia_actual = self.cfg.coef_entropia
        self.ruta_metricas = self.dir_salida / "metricas.jsonl"

        # semilla de evaluación distinta de la de entrenamiento
        self._semilla_eval = self.cfg.semilla + 1

    # --------------------------------------------------
    # registro
    # --------------------------------------------------
    def _log(self, mensaje: str) -> None:
        if not self.silencioso:
            print(mensaje, flush=True)

    def _registrar(self, registro: dict) -> None:
        registro["paso"] = self.paso_global
        registro["etapa"] = self.etapa
        with open(self.ruta_metricas, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    # --------------------------------------------------
    # recolección de rollouts
    # --------------------------------------------------
    def _recolectar(self):
        episodios: list[Episodio] = []
        ep = Episodio()
        obs = self.entorno.reiniciar()
        h, M = self.cerebro.estado_inicial(dispositivo=self.dispositivo)
        recompensa_total = 0.0
        entropia_total = 0.0
        pasos_rollout = 0

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.dispositivo)
            logits_p, g, valor, h, M = self.cerebro.paso(obs_t, h, M)

            dist_p = torch.distributions.Categorical(logits=logits_p)
            prim = dist_p.sample()
            logits_a = enmascarar_logits_argumento(
                self.cerebro.logits_argumento(g, prim), prim
            )
            dist_a = torch.distributions.Categorical(logits=logits_a)
            arg = dist_a.sample()
            logp = (
                dist_p.log_prob(prim).item() + dist_a.log_prob(arg).item()
            )
            entropia_total += (
                dist_p.entropy().item() + dist_a.entropy().item()
            )

            obs, recompensa, terminada, info = self.entorno.paso(
                int(prim.item()), int(arg.item())
            )
            recompensa_total += recompensa
            ep.obs.append(list(obs_t.cpu().numpy()))
            ep.prim.append(int(prim.item()))
            ep.arg.append(int(arg.item()))
            ep.logp.append(logp)
            ep.valor.append(float(valor.item()))
            ep.recompensa.append(float(recompensa))
            self.paso_global += 1
            pasos_rollout += 1

            if terminada:
                episodios.append(ep)
                ep = Episodio()
                # el rollout se corta en frontera de episodio, pero
                # garantizando siempre un mínimo de datos para PPO
                if pasos_rollout >= self.cfg.pasos_por_rollout and len(episodios) >= 16:
                    break
                obs = self.entorno.reiniciar()
                h, M = self.cerebro.estado_inicial(dispositivo=self.dispositivo)

        n_pasos = sum(len(e) for e in episodios)
        return (
            episodios,
            recompensa_total / max(1, len(episodios)),
            entropia_total / max(1, n_pasos),
        )

    # --------------------------------------------------
    # actualización PPO
    # --------------------------------------------------
    def _actualizar(self, episodios: list[Episodio]) -> dict:
        # GAE por episodio + normalización global de ventajas
        ventajas, retornos = [], []
        for e in episodios:
            obs, prim, arg, logp, valor, recomp = e.tensores()
            adv, ret = _gae(recomp, valor, self.cfg.gamma, self.cfg.lambda_gae)
            ventajas.append(adv)
            retornos.append(ret)
        todas_adv = torch.cat(ventajas)
        media = todas_adv.mean()
        desviacion = todas_adv.std()
        if not torch.isfinite(desviacion) or desviacion < 1e-6:
            desviacion = torch.tensor(1.0)

        datos = []
        for e, adv, ret in zip(episodios, ventajas, retornos):
            obs, prim, arg, logp, valor, recomp = e.tensores()
            datos.append({
                "obs": obs.to(self.dispositivo),
                "prim": prim.to(self.dispositivo),
                "arg": arg.to(self.dispositivo),
                "logp": logp.to(self.dispositivo),
                "ret": ret.to(self.dispositivo),
                "adv": ((adv - media) / (desviacion + 1e-8)).to(self.dispositivo),
            })

        estadisticas = {"perdida_politica": 0.0, "perdida_valor": 0.0, "entropia": 0.0}
        denom = len(datos) * self.cfg.epocas

        for _ in range(self.cfg.epocas):
            # barajar y ordenar por longitud (agrupa episodios
            # parecidos: menos relleno inútil en el lote)
            orden = list(range(len(datos)))
            random.shuffle(orden)
            orden.sort(key=lambda j: len(datos[j]["prim"]))
            for i in range(0, len(orden), self.cfg.episodios_por_grupo):
                grupo = [datos[j] for j in orden[i : i + self.cfg.episodios_por_grupo]]
                if not grupo:
                    continue
                lote = self._empaquetar(grupo)
                self.optimizador.zero_grad()

                logits_p, g_seq, valores, _, _ = self.cerebro.secuencia(
                    lote["obs"], lote["h0"], lote["M0"]
                )
                # argumentos condicionados a las primitivas tomadas,
                # restringidos al contrato de tipos del kernel
                logits_a = enmascarar_logits_argumento(
                    self.cerebro.logits_argumento(g_seq, lote["prim"]),
                    lote["prim"],
                )
                mascara = lote["mascara"]  # [B, T] 1 real / 0 relleno

                logp_nuevo = (
                    F.log_softmax(logits_p, dim=-1)
                    .gather(2, lote["prim"].unsqueeze(2))
                    .squeeze(2)
                    + F.log_softmax(logits_a, dim=-1)
                    .gather(2, lote["arg"].unsqueeze(2))
                    .squeeze(2)
                )
                ratio = torch.exp(logp_nuevo - lote["logp"])
                surr1 = ratio * lote["adv"]
                surr2 = (
                    torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip)
                    * lote["adv"]
                )
                perdida_politica = -(
                    (torch.min(surr1, surr2) * mascara).sum() / mascara.sum()
                )
                dif = (valores - lote["ret"]) ** 2
                perdida_valor = (dif * mascara).sum() / mascara.sum()
                entropia = (
                    (entropia_conjunta(logits_p, logits_a) * mascara).sum()
                    / mascara.sum()
                )

                perdida = (
                    perdida_politica
                    + self.cfg.coef_valor * perdida_valor
                    - self._coef_entropia_actual * entropia
                )
                perdida.backward()

                estadisticas["perdida_politica"] += perdida_politica.item() / denom
                estadisticas["perdida_valor"] += perdida_valor.item() / denom
                estadisticas["entropia"] += entropia.item() / denom

                torch.nn.utils.clip_grad_norm_(
                    self.cerebro.parameters(), self.cfg.max_grad_norm
                )
                self.optimizador.step()

        self.cerebro.train()
        return estadisticas

    def _empaquetar(self, grupo: list) -> dict:
        """Convierte un grupo de episodios en un lote rellenado [B, T, ...]."""
        B = len(grupo)
        T = max(len(d["prim"]) for d in grupo)
        D = grupo[0]["obs"].shape[1]

        def rellenar(tensor, forma, valor=0):
            salida = torch.full(forma, valor, dtype=tensor.dtype,
                                device=self.dispositivo)
            salida[: tensor.shape[0]] = tensor
            return salida

        obs = torch.stack([rellenar(d["obs"], (T, D)) for d in grupo])
        mascara = torch.zeros(B, T, device=self.dispositivo)
        for b, d in enumerate(grupo):
            mascara[b, : len(d["prim"])] = 1.0

        h0 = torch.zeros(B, self.cerebro.oculto, device=self.dispositivo)
        M0 = (
            self.cerebro.memoria_inicial.detach()
            .unsqueeze(0)
            .expand(B, -1, -1)
            .contiguous()
        )
        return {
            "obs": obs,
            "prim": torch.stack([rellenar(d["prim"], (T,)) for d in grupo]),
            "arg": torch.stack([rellenar(d["arg"], (T,)) for d in grupo]),
            "logp": torch.stack([rellenar(d["logp"], (T,)) for d in grupo]),
            "ret": torch.stack([rellenar(d["ret"], (T,)) for d in grupo]),
            "adv": torch.stack([rellenar(d["adv"], (T,)) for d in grupo]),
            "mascara": mascara,
            "h0": h0,
            "M0": M0,
        }

    # --------------------------------------------------
    # evaluación y currículo
    # --------------------------------------------------
    def _evaluar_etapa(self) -> dict:
        tareas = CURRICULO[self.etapa]
        return evaluar(
            self.cerebro,
            tareas,
            n_solicitudes=self.cfg.solicitudes_eval,
            semilla=self._semilla_eval,
        )

    def _guardar(self, nombre: str, exito_medio: float | None = None) -> Path:
        ruta = self.dir_salida / nombre
        paquete = {
            "config": self.cerebro.configuracion(),
            "estado": self.cerebro.state_dict(),
            "optimizador": self.optimizador.state_dict(),
            "paso": self.paso_global,
            "etapa": self.etapa,
            "exito_medio_eval": (
                exito_medio if exito_medio is not None else self.mejor_exito
            ),
        }
        torch.save(paquete, ruta)
        return ruta

    def reanudar(self, ruta) -> int:
        """Reanuda la incubación desde un checkpoint (ultimo.pt).

        Devuelve el paso global guardado. Permite incubar en varias
        sesiones: 'brooder incubar' + 'brooder incubar --reanudar'.
        """
        paquete = torch.load(ruta, map_location=self.dispositivo, weights_only=False)
        self.cerebro.load_state_dict(paquete["estado"])
        if "optimizador" in paquete:  # checkpoints antiguos: sin estado de Adam
            self.optimizador.load_state_dict(paquete["optimizador"])
        self.paso_global = paquete.get("paso", 0)
        self.etapa = paquete.get("etapa", 0)
        self.entorno.fijar_tareas(CURRICULO[min(self.etapa, len(CURRICULO) - 1)])
        self.cerebro.train()
        return self.paso_global

    # --------------------------------------------------
    # bucle principal
    # --------------------------------------------------
    def entrenar(self, reanudar: bool = False) -> dict:
        if reanudar:
            checkpoint = self.dir_salida / "ultimo.pt"
            if checkpoint.exists():
                paso_previo = self.reanudar(checkpoint)
                self._log(f"Reanudando desde el paso {paso_previo} (etapa {self.etapa + 1}).")
            else:
                self._log("No hay checkpoint previo: incubación desde cero.")
        self._log(
            "┌────────────────────────────────────────────────────┐\n"
            "│  INCUBADORA DE BROODER — entrenamiento PPO         │\n"
            f"│  dispositivo: {str(self.dispositivo):37s} │\n"
            f"│  pasos objetivo: {self.cfg.pasos_totales:<33d} │\n"
            "└────────────────────────────────────────────────────┘"
        )
        t_inicio = time.time()
        n_rollout = 0
        ultimo_eval = -1

        while self.paso_global < self.cfg.pasos_totales and not self.convergido:
            t0 = time.time()
            episodios, recompensa_media, entropia_media = self._recolectar()
            stats = self._actualizar(episodios)
            n_rollout += 1
            duracion = time.time() - t0

            # métricas de entrenamiento (ventana móvil)
            ventana = self.entorno.resultados[-200:]
            exito_ent = {}
            for tarea, exito in ventana:
                a, t = exito_ent.get(tarea, (0, 0))
                exito_ent[tarea] = (a + int(exito), t + 1)
            exito_ent = {
                k: round(v[0] / v[1], 3) for k, v in exito_ent.items() if v[1] >= 5
            }

            # ENTROPÍA ADAPTATIVA: si alguna tarea de la etapa sigue
            # sin resolverse, la política podría colapsar a lo ya
            # aprendido y dejar de muestrear las primitivas nuevas
            # (exploración muerta). Se eleva el incentivo de entropía
            # hasta que todas las tareas activas superan el umbral.
            min_exito = min(exito_ent.values()) if exito_ent else 0.0
            self._coef_entropia_actual = (
                self.cfg.coef_entropia_exploracion
                if min_exito < self.cfg.umbral_exploracion
                else self.cfg.coef_entropia
            )

            if n_rollout % self.cfg.imprimir_cada_rollouts == 0:
                self._log(
                    f"[paso {self.paso_global:>7d}] etapa {self.etapa + 1} | "
                    f"recompensa {recompensa_media:+.2f} | "
                    f"entrenamiento {exito_ent} | "
                    f"entropía {entropia_media:.2f} | "
                    f"{duracion:.1f}s"
                )

            # ¿toca evaluar?
            if (
                self.paso_global - ultimo_eval >= self.cfg.eval_cada
                or self.paso_global >= self.cfg.pasos_totales
            ):
                ultimo_eval = self.paso_global
                exito_eval = self._evaluar_etapa()
                tareas_vistas = CURRICULO[self.etapa]
                exito_medio = sum(exito_eval.values()) / len(exito_eval)
                self._log(
                    f"  ↳ EVAL: "
                    + " | ".join(f"{t} {exito_eval[t]:.0%}" for t in exito_eval)
                    + f"  (media {exito_medio:.0%})"
                )

                # mejor checkpoint global
                if exito_medio > self.mejor_exito:
                    self.mejor_exito = exito_medio
                    self._guardar("mejor.pt", exito_medio)
                self._guardar("ultimo.pt", exito_medio)

                self._registrar({
                    "recompensa_media": round(recompensa_media, 4),
                    "entropia_media": round(entropia_media, 4),
                    "exito_entrenamiento": exito_ent,
                    "exito_eval": {k: round(v, 4) for k, v in exito_eval.items()},
                    "exito_medio_eval": round(exito_medio, 4),
                    "perdida_politica": round(stats["perdida_politica"], 5),
                    "perdida_valor": round(stats["perdida_valor"], 5),
                    "tareas_evaluadas": [t.name for t in tareas_vistas],
                    "segundos": round(time.time() - t_inicio, 1),
                })

                # ¿avanza de etapa / converge?
                if all(v >= self.cfg.umbral_etapa for v in exito_eval.values()):
                    if self.etapa < len(CURRICULO) - 1:
                        self.etapa += 1
                        self.entorno.fijar_tareas(CURRICULO[self.etapa])
                        self._log(
                            f"  ★ ETAPA {self.etapa + 1}: nuevas tareas "
                            f"{[t.name for t in CURRICULO[self.etapa]]}"
                        )
                        # re-evaluar con el repertorio ampliado
                        ultimo_eval = self.paso_global
                    elif self.cfg.parar_al_converger:
                        self.convergido = True
                        self._log(
                            "  ★ CONVERGIDO: todas las tareas dominadas "
                            f"(media {exito_medio:.0%}). Incubación completa."
                        )

        # guardado final
        exito_final = self._evaluar_etapa()
        exito_medio = sum(exito_final.values()) / max(1, len(exito_final))
        if exito_medio >= self.mejor_exito:
            self.mejor_exito = exito_medio
            self._guardar("mejor.pt", exito_medio)
        self._guardar("ultimo.pt", exito_medio)

        resumen = {
            "pasos": self.paso_global,
            "etapa_final": self.etapa + 1,
            "convergido": self.convergido,
            "exito_eval_final": {k: round(v, 4) for k, v in exito_final.items()},
            "exito_medio_final": round(exito_medio, 4),
            "segundos": round(time.time() - t_inicio, 1),
            "dispositivo": str(self.dispositivo),
        }
        self._registrar({"resumen_final": resumen})
        self._log(
            "Incubación terminada: "
            + " | ".join(f"{t} {v:.0%}" for t, v in exito_final.items())
        )
        return resumen
