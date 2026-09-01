"""Tests de seguridad: la imagen SSD nunca ejecuta código.

Brooder carga artefactos que viajan por redes no confiables (la imagen
SSD, los checkpoints). Un .pt es un pickle: si se deserializa sin
restricciones, cualquier payload embebido se ejecuta en la máquina que
carga. Estos tests fijan la política inversa — weights_only=True en
todos los puntos de entrada y solo config+pesos en la imagen.
"""
import io
import pickle
from zipfile import ZipFile

import pytest
import torch

from brooder.cerebro import CerebroBrooder
from brooder.nucleo import exportar_ssd, montar_ssd


@pytest.fixture
def modelo(tmp_path):
    """Un cerebro inocuo guardado en disco (como mejor.pt)."""
    cerebro = CerebroBrooder(oculto=16, mem_ranuras=2, mem_dim=4)
    ruta = tmp_path / "mejor.pt"
    cerebro.guardar(ruta)
    return ruta


# ------------------------------------------------------------------
# carga restringida: un pickle malicioso NO se ejecuta
# ------------------------------------------------------------------
def test_cargar_rechaza_pickle_malicioso(tmp_path):
    """CerebroBrooder.cargar debe usar weights_only=True.

    El payload ejecutaría `os.system` al deserializarse; con
    weights_only=True la carga aborta antes de ejecutar nada.
    """
    marcador = tmp_path / "PWNED_cargar"
    ruta = tmp_path / "malicioso.pt"

    class Payload:
        def __reduce__(self):
            import os

            return (os.system, (f"touch {marcador}",))

    torch.save(Payload(), ruta)

    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        CerebroBrooder.cargar(ruta)
    assert not marcador.exists(), (
        "el payload se ejecutó: torch.load sin weights_only=True"
    )


def test_montar_ssd_rechaza_imagen_maliciosa(tmp_path):
    """montar_ssd es la puerta del PC de nacimiento: debe abortar."""
    marcador = tmp_path / "PWNED_montar"
    ruta_pt = tmp_path / "malicioso.pt"

    class Payload:
        def __reduce__(self):
            import os

            return (os.system, (f"touch {marcador}",))

    torch.save(Payload(), ruta_pt)

    imagen = tmp_path / "brooder.img"
    with ZipFile(imagen, "w") as z:
        z.writestr("brooder.pt", ruta_pt.read_bytes())
        z.writestr("manifiesto.json", "{}")

    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        montar_ssd(imagen)
    assert not marcador.exists(), (
        "el payload se ejecutó: montar_ssd deserializó código ajeno"
    )


# ------------------------------------------------------------------
# la imagen SSD solo contiene config + pesos
# ------------------------------------------------------------------
def test_exportar_excluye_el_optimizador(tmp_path):
    """Un checkpoint de entrenamiento lleva el estado de Adam;
    la imagen exportada NO debe transportarlo al PC de nacimiento.
    """
    cerebro = CerebroBrooder(oculto=16, mem_ranuras=2, mem_dim=4)
    optimizador = torch.optim.Adam(cerebro.parameters(), lr=1e-3)
    # poblar el estado de Adam con un paso falso
    dim_cabezas = cerebro.oculto + cerebro.mem_dim
    perdida = cerebro.cabeza_valor(torch.zeros(1, dim_cabezas)).sum()
    perdida.backward()
    optimizador.step()

    checkpoint = tmp_path / "mejor.pt"
    torch.save(
        {
            "config": cerebro.configuracion(),
            "estado": cerebro.state_dict(),
            "optimizador": optimizador.state_dict(),
            "paso": 1234,
            "etapa": 2,
            "exito_medio_eval": 0.9,
        },
        checkpoint,
    )

    imagen = tmp_path / "brooder.img"
    exportar_ssd(checkpoint, imagen)

    with ZipFile(imagen) as z:
        paquete = torch.load(
            io.BytesIO(z.read("brooder.pt")), weights_only=True
        )
    assert set(paquete) == {"config", "estado"}, (
        f"la imagen transporta más de lo necesario: {sorted(paquete)}"
    )
    assert isinstance(CerebroBrooder(**paquete["config"]), CerebroBrooder)


def test_imagen_existente_carga_en_modo_restringido(modelo, tmp_path):
    """Ida y vuelta: exportar y montar funcionan con weights_only=True."""
    imagen = tmp_path / "brooder.img"
    exportar_ssd(modelo, imagen, metricas={"exito_eval_final": {"ECO": 1.0}})

    cerebro, _estado, manifiesto = montar_ssd(imagen)
    assert isinstance(cerebro, CerebroBrooder)
    assert manifiesto["metricas"]["exito_eval_final"]["ECO"] == 1.0
