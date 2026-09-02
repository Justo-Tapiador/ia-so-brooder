"""
Tests del cargador de config.json (Fase 2 — emulador web)
=========================================================

La convención es la de pendrive.json: validación honesta campo a campo.
Un valor inválido se sustituye por su defecto y se anuncia en avisos.
Nunca un crash, nunca un silencio.
"""
from __future__ import annotations

import json
from pathlib import Path

from brooder.config import cargar_config


def _escribir(tmp_path, datos):
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


# ------------------------------------------------------------------
# por defecto (sin archivo)
# ------------------------------------------------------------------
def test_sin_config_usa_perfil_d_y_avisa(tmp_path):
    config = cargar_config(tmp_path / "config.json")
    assert config.perfil == "D"
    assert config.host == "127.0.0.1"
    assert config.puerto == 7800
    assert config.maquina == "virtual"
    assert config.columnas == 100 and config.filas == 30
    assert any("no existe" in aviso for aviso in config.avisos)


def test_config_corrupto_no_rompe(tmp_path):
    ruta = tmp_path / "config.json"
    ruta.write_text("{perfil: roto", encoding="utf-8")
    config = cargar_config(ruta)
    assert config.perfil == "D"
    assert any("ilegible" in aviso for aviso in config.avisos)


def test_config_con_raiz_no_objeto(tmp_path):
    ruta = _escribir(tmp_path, ["A", "B"])
    config = cargar_config(ruta)
    assert config.perfil == "D"
    assert any("raíz" in aviso for aviso in config.avisos)


# ------------------------------------------------------------------
# perfiles del README
# ------------------------------------------------------------------
def test_perfil_a_pc_reciclado(tmp_path):
    ruta = _escribir(tmp_path, {"perfil": "A"})
    config = cargar_config(ruta)
    assert config.columnas == 80 and config.filas == 25
    assert config.retardo_post_ms == 120
    assert config.nombre_perfil == "PC reciclado"


def test_perfil_c_incubadora_con_metricas(tmp_path):
    ruta = _escribir(tmp_path, {"perfil": "C"})
    config = cargar_config(ruta)
    assert config.columnas == 120 and config.filas == 32
    assert config.retardo_post_ms == 0
    assert config.metricas is True


def test_perfil_desconocido_cae_a_d_con_aviso(tmp_path):
    ruta = _escribir(tmp_path, {"perfil": "Z"})
    config = cargar_config(ruta)
    assert config.perfil == "D"
    assert any("perfil" in aviso.lower() for aviso in config.avisos)


# ------------------------------------------------------------------
# máquina / ssd / sandbox
# ------------------------------------------------------------------
def test_maquina_real(tmp_path):
    ruta = _escribir(tmp_path, {"maquina": "real"})
    config = cargar_config(ruta)
    assert config.maquina_real is True
    assert config.sandbox == Path("brooder_sandbox")


def test_maquina_invalida(tmp_path):
    ruta = _escribir(tmp_path, {"maquina": "cuántica"})
    config = cargar_config(ruta)
    assert config.maquina == "virtual"
    assert any("maquina" in aviso.lower() for aviso in config.avisos)


def test_ssd_inexistente_avisa(tmp_path):
    ruta = _escribir(tmp_path, {"ssd": "ssd/fantasma.img"})
    config = cargar_config(ruta)
    assert config.ssd == Path("ssd/fantasma.img")   # se conserva (el POST avisará)
    assert any("no existe" in aviso for aviso in config.avisos)


# ------------------------------------------------------------------
# red y consola
# ------------------------------------------------------------------
def test_red_invalida_usa_defectos(tmp_path):
    ruta = _escribir(tmp_path, {"red": {"host": "", "puerto": 99999}})
    config = cargar_config(ruta)
    assert config.host == "127.0.0.1"
    assert config.puerto == 7800
    assert any("puerto" in aviso for aviso in config.avisos)


def test_overrides_de_consola_sobre_el_perfil(tmp_path):
    ruta = _escribir(
        tmp_path,
        {"perfil": "A", "consola": {"columnas": 132, "tema": "verde"}},
    )
    config = cargar_config(ruta)
    assert config.columnas == 132          # override del 80 del perfil A
    assert config.filas == 25              # el resto del perfil se conserva
    assert config.tema == "verde"


def test_tema_desconocido(tmp_path):
    ruta = _escribir(tmp_path, {"consola": {"tema": "rosa"}})
    config = cargar_config(ruta)
    assert config.tema == "cian"
    assert any("tema" in aviso.lower() for aviso in config.avisos)


def test_consola_no_objeto_se_ignora(tmp_path):
    ruta = _escribir(tmp_path, {"consola": [1, 2]})
    config = cargar_config(ruta)
    assert config.columnas == 100
    assert any("consola" in aviso.lower() for aviso in config.avisos)


def test_panel_false(tmp_path):
    ruta = _escribir(tmp_path, {"consola": {"panel": False}})
    config = cargar_config(ruta)
    assert config.panel is False
