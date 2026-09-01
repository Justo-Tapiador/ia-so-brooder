"""
Cerebro de Brooder — GRU + memoria asociativa + cabezas PPO
===========================================================

Sustituye al MLP del primer prototipo por una arquitectura con tres
ideas, tal como se acordó en la conversación original:

1. **Recurrencia (GRU)**: Brooder opera en el tiempo; cada ciclo
   depende de lo que ha hecho antes (leyó un dígito y aún no lo
   sumó, buscó una ranura y aún no escribió...).

2. **Memoria asociativa**: un banco de ranuras direccionables POR
   CONTENIDO (lectura y borrado/añadido con atención softmax, al
   estilo de las NTM/DNC simplificadas). No es el diccionario
   `memory` del prototipo: es un espacio vectorial que la propia red
   aprende a organizar.

3. **Decisión factorizada y condicionada**: la acción es el par
   (primitiva, argumento). Primero se elige la primitiva; el
   argumento se muestrea de una distribución QUE DEPENDE de la
   primitiva elegida. Así la red puede aprender, por ejemplo, que
   `mostrar_en_pantalla` casi siempre quiere el BUS mientras que
   `mover_cabezal_disco` quiere una dirección, sin que los errores
   de una contaminen a la otra.

El mismo módulo sirve para entrenar (muestreo estocástico) y para
operar (modo determinista).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from brooder.constantes import (
    ARG_BUS,
    MASCARA_ARGUMENTOS,
    N_ARGUMENTOS,
    N_PRIMITIVAS,
    OBS_DIM,
)

# máscara de tipos como tensor (ver constantes.MASCARA_ARGUMENTOS)
_TENSOR_MASCARA = torch.tensor(MASCARA_ARGUMENTOS, dtype=torch.bool)


def enmascarar_logits_argumento(logits, primitivas):
    """Aplica el contrato de tipos del núcleo a los logits.

    logits: [..., n_argumentos] | primitivas: [...] (long)
    Las combinaciones que el kernel rechazaría no se proponen
    siquiera: su logit pasa a -inf.
    """
    mascara = _TENSOR_MASCARA.to(logits.device)[primitivas]
    return logits.masked_fill(~mascara, float("-inf"))


class CerebroBrooder(nn.Module):
    """Red neuronal de IA-SO Brooder."""

    def __init__(
        self,
        dim_entrada: int = OBS_DIM,
        n_primitivas: int = N_PRIMITIVAS,
        n_argumentos: int = N_ARGUMENTOS,
        oculto: int = 96,
        mem_ranuras: int = 8,
        mem_dim: int = 16,
        dim_embebido_prim: int = 16,
    ):
        super().__init__()
        self.dim_entrada = dim_entrada
        self.n_primitivas = n_primitivas
        self.n_argumentos = n_argumentos
        self.oculto = oculto
        self.mem_ranuras = mem_ranuras
        self.mem_dim = mem_dim

        # percepción -> espacio oculto
        self.codificador = nn.Sequential(
            nn.Linear(dim_entrada, oculto),
            nn.Tanh(),
        )

        # recurrencia
        self.gru = nn.GRUCell(oculto, oculto)

        # ---- memoria asociativa (direccionable por contenido) ----
        self.consulta_lectura = nn.Linear(oculto, mem_dim)
        self.consulta_escritura = nn.Linear(oculto, mem_dim)
        self.puerta_borrado = nn.Linear(oculto, mem_dim)
        self.vector_escritura = nn.Linear(oculto, mem_dim)
        # estado inicial de la memoria: parámetro aprendido
        self.memoria_inicial = nn.Parameter(torch.zeros(mem_ranuras, mem_dim))

        # ---- cabezas de decisión ----
        dim_cabezas = oculto + mem_dim
        self.cabeza_primitiva = nn.Linear(dim_cabezas, n_primitivas)
        self.cabeza_valor = nn.Linear(dim_cabezas, 1)

        # el argumento se condiciona a la primitiva elegida:
        # P(a | prim) en vez de P(a) independiente
        self.embebido_primitiva = nn.Embedding(n_primitivas, dim_embebido_prim)
        self.cabeza_argumento = nn.Linear(
            dim_cabezas + dim_embebido_prim, n_argumentos
        )

        # Sesgo inductivo suave: en un ordenador casi todo dato que
        # se manipula procede del bus (acaba de leerse). Se le da
        # una pequeña ventaja inicial al argumento BUS.
        with torch.no_grad():
            self.cabeza_argumento.bias[ARG_BUS] += 2.0

        self._escala_atencion = 1.0 / math.sqrt(mem_dim)

    # --------------------------------------------------
    # estados iniciales
    # --------------------------------------------------
    def estado_inicial(self, lote: int = 1, dispositivo=None):
        """h y M con los que arranca cada solicitud."""
        h = torch.zeros(lote, self.oculto, device=dispositivo)
        M = self.memoria_inicial.detach().clone()
        M = M.unsqueeze(0).expand(lote, -1, -1).contiguous()
        return h, M

    # --------------------------------------------------
    # un paso de cálculo
    # --------------------------------------------------
    def paso(self, obs, h, M):
        """Un ciclo de percepción + memoria.

        obs: [B, dim_entrada] | h: [B, oculto] | M: [B, ranuras, dim]
        Devuelve (logits_primitiva, g, valor, h', M') donde g es el
        vector que alimenta las cabezas; los logits del argumento se
        calculan después, condicionados a la primitiva elegida.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z = self.codificador(obs)
        h = self.gru(z, h)

        # --- lectura por contenido ---
        consulta = self.consulta_lectura(h)                      # [B, dim]
        afinidad = torch.einsum("bd,bsd->bs", consulta, M) * self._escala_atencion
        atencion_lectura = F.softmax(afinidad, dim=-1)           # [B, ranuras]
        lectura = torch.einsum("bs,bsd->bd", atencion_lectura, M)  # [B, dim]

        # --- escritura por contenido (borrar + añadir) ---
        clave = self.consulta_escritura(h)
        afinidad_w = torch.einsum("bd,bsd->bs", clave, M) * self._escala_atencion
        atencion_escritura = F.softmax(afinidad_w, dim=-1)       # [B, ranuras]
        borrar = torch.sigmoid(self.puerta_borrado(h))           # [B, dim]
        anadir = torch.tanh(self.vector_escritura(h))            # [B, dim]
        M = M * (1.0 - atencion_escritura.unsqueeze(-1) * borrar.unsqueeze(1))
        M = M + atencion_escritura.unsqueeze(-1) * anadir.unsqueeze(1)

        # --- decisión ---
        g = torch.cat([h, lectura], dim=-1)
        logits_primitiva = self.cabeza_primitiva(g)
        valor = self.cabeza_valor(g).squeeze(-1)
        return logits_primitiva, g, valor, h, M

    def logits_argumento(self, g, primitivas):
        """Logits del argumento condicionados a la primitiva elegida.

        g: [B, G] o [B, T, G] | primitivas: [B] o [B, T] (long)
        """
        embebido = self.embebido_primitiva(primitivas)
        if g.dim() == 2:
            entrada = torch.cat([g, embebido], dim=-1)
        else:
            entrada = torch.cat([g, embebido], dim=-1)
        return self.cabeza_argumento(entrada)

    # --------------------------------------------------
    # secuencia completa (para BPTT del PPO)
    # --------------------------------------------------
    def secuencia(self, obs_seq, h0, M0):
        """Aplica `paso` a lo largo del tiempo.

        obs_seq: [B, T, dim] | h0: [B, oculto] | M0: [B, S, D]
        Devuelve (logits_prim [B,T,P], g [B,T,G], valores [B,T], h, M).
        """
        B, T, _ = obs_seq.shape
        lp, valores, gs = [], [], []
        h, M = h0, M0
        for t in range(T):
            logits_p, g, valor, h, M = self.paso(obs_seq[:, t], h, M)
            lp.append(logits_p)
            gs.append(g)
            valores.append(valor)
        return (
            torch.stack(lp, dim=1),
            torch.stack(gs, dim=1),
            torch.stack(valores, dim=1),
            h,
            M,
        )

    # --------------------------------------------------
    # decisión operativa
    # --------------------------------------------------
    @torch.no_grad()
    def decidir(self, obs, h, M, determinista: bool = True):
        """Elige (primitiva, argumento) para el ciclo actual.

        En producción es determinista (argmax): un sistema operativo
        debe comportarse de forma predecible. En entrenamiento se
        muestrea. El argumento se muestrea del espacio válido para
        la primitiva elegida (contrato de tipos del kernel).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits_p, g, valor, h, M = self.paso(obs, h, M)
        if determinista:
            prim = logits_p.argmax(dim=-1)
        else:
            prim = torch.distributions.Categorical(logits=logits_p).sample()
        logits_a = enmascarar_logits_argumento(
            self.logits_argumento(g, prim), prim
        )
        if determinista:
            arg = logits_a.argmax(dim=-1)
        else:
            arg = torch.distributions.Categorical(logits=logits_a).sample()
        return (
            int(prim.item()),
            int(arg.item()),
            float(valor.item()),
            h,
            M,
        )

    # --------------------------------------------------
    # persistencia
    # --------------------------------------------------
    def configuracion(self) -> dict:
        return {
            "dim_entrada": self.dim_entrada,
            "n_primitivas": self.n_primitivas,
            "n_argumentos": self.n_argumentos,
            "oculto": self.oculto,
            "mem_ranuras": self.mem_ranuras,
            "mem_dim": self.mem_dim,
        }

    def guardar(self, ruta) -> None:
        torch.save(
            {"config": self.configuracion(), "estado": self.state_dict()},
            ruta,
        )

    @staticmethod
    def cargar(ruta, dispositivo=None) -> "CerebroBrooder":
        # weights_only=True: un .pt es un pickle; cargarlo sin más
        # ejecutaría cualquier código embebido. Aquí solo deben vivir
        # config (ints) y pesos (tensores), así que el modo restringido
        # de torch es suficiente — y si no lo fuera, mejor que falle.
        paquete = torch.load(ruta, map_location=dispositivo, weights_only=True)
        cerebro = CerebroBrooder(**paquete["config"])
        cerebro.load_state_dict(paquete["estado"])
        cerebro.eval()
        return cerebro


def entropia_conjunta(logits_p, logits_a):
    """Entropía de la política factorizada (primitiva + argumento).

    Segura frente a logits enmascarados (-inf): las opciones
    inválidas tienen probabilidad 0 y no contribuyen.
    """
    lp = F.log_softmax(logits_p, dim=-1)
    la = F.log_softmax(logits_a, dim=-1)
    ent_p = -(lp.exp() * lp.clamp(min=-1e9)).sum(-1)
    ent_a = -(la.exp() * la.clamp(min=-1e9)).sum(-1)
    return ent_p + ent_a
