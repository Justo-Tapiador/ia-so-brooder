"""
Tests del servidor-emulador (Fase 2 — emulador web)
===================================================

Arranca un ThreadingHTTPServer real en un puerto efímero (0) sobre un
config de prueba y habla con la API JSON con urllib (stdlib): la misma
herramienta que tendría cualquier cliente.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from brooder.config import ConfigServidor
from brooder.servidor import crear_httpd

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def servidor():
    """Un emulador en 127.0.0.1:puerto-efímero, máquina virtual."""
    config = ConfigServidor()
    config.ssd = REPO / "ssd" / "brooder.img"
    config.host = "127.0.0.1"
    config.puerto = 0  # efímero: el SO elige
    httpd = crear_httpd(config)
    hilo = threading.Thread(target=httpd.serve_forever, daemon=True)
    hilo.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, httpd
    httpd.shutdown()
    httpd.server_close()


def _get(base, ruta):
    with urllib.request.urlopen(base + ruta, timeout=10) as respuesta:
        return respuesta.status, respuesta.read().decode("utf-8")


def _get_json(base, ruta):
    estado, texto = _get(base, ruta)
    return estado, json.loads(texto)


def _post(base, ruta, datos):
    cuerpo = json.dumps(datos).encode("utf-8")
    peticion = urllib.request.Request(
        base + ruta, data=cuerpo, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(peticion, timeout=30) as respuesta:
        return respuesta.status, json.loads(respuesta.read().decode("utf-8"))


# ------------------------------------------------------------------
# la consola y su configuración
# ------------------------------------------------------------------
def test_get_sirve_la_consola(servidor):
    base, _ = servidor
    estado, html = _get(base, "/")
    assert estado == 200
    assert "EMULADOR WEB" in html
    assert "brooder&gt;" in html          # el prompt de la consola
    assert "api/arrancar" in html         # el flujo de encendido


def test_api_config_expone_el_monitor(servidor):
    base, _ = servidor
    _, datos = _get_json(base, "/api/config")
    assert datos["perfil"] == "D"
    assert datos["consola"]["columnas"] == 100
    assert datos["consola"]["retardo_post_ms"] == 60
    assert datos["version"]


def test_ruta_desconocida_404(servidor):
    base, _ = servidor
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as info:
        _get(base, "/api/fantasma")
    assert info.value.code == 404


# ------------------------------------------------------------------
# ciclo de vida: arrancar -> atender -> apagar
# ------------------------------------------------------------------
def test_arrancar_devuelve_el_post_completo(servidor):
    base, _ = servidor
    estado, datos = _post(base, "/api/arrancar", {})
    assert estado == 200 and datos["ok"]
    texto = "\n".join(datos["lineas"])
    assert "IA-SO Brooder BIOS v" in texto
    assert "contrato 26x23" in texto
    assert "IA-SO BROODER EN MARCHA" in texto
    assert datos["encendida"] is True
    assert datos["resumen"] is not None


def test_arrancar_dos_veces_rechaza_con_mensaje(servidor):
    base, _ = servidor
    _post(base, "/api/arrancar", {})
    _, segunda = _post(base, "/api/arrancar", {})
    assert any("ya está encendida" in l for l in segunda["lineas"])
    assert segunda["encendida"] is True


def test_linea_suma_por_la_web(servidor):
    base, _ = servidor
    _post(base, "/api/arrancar", {})
    _, datos = _post(base, "/api/linea", {"texto": "3+5"})
    texto = "\n".join(datos["lineas"])
    assert "[ OK ]" in texto and "pantalla='8'" in texto
    assert datos["panel"]["dispositivos"]["acumulador"] == 8


def test_ciclo_completo_del_pendrive_por_la_web(servidor):
    base, _ = servidor
    _post(base, "/api/arrancar", {})

    # lectura con pendrive ausente: veredicto honesto del kernel
    _, fallo = _post(base, "/api/linea", {"texto": "leer 3P"})
    assert "[FALLO]" in "\n".join(fallo["lineas"])
    assert "pendrive=ausente" in "\n".join(fallo["lineas"])

    _, enchufar = _post(base, "/api/linea", {"texto": ":pendrive"})
    assert "pendrive conectado" in "\n".join(enchufar["lineas"])

    _, montar = _post(base, "/api/linea", {"texto": "montar"})
    assert "pendrive=montado" in "\n".join(montar["lineas"])

    _, escribir = _post(base, "/api/linea", {"texto": "escribir 3P"})
    assert "pendrive[3]='P'" in "\n".join(escribir["lineas"])

    _, leer = _post(base, "/api/linea", {"texto": "leer 3P"})
    assert "[ OK ]" in "\n".join(leer["lineas"])
    assert "pendrive[3]='P'" in "\n".join(leer["lineas"])

    # el panel del monitor refleja el estado del dispositivo
    _, estado = _get_json(base, "/api/estado")
    assert "montado" in estado["panel"]["dispositivos"]["pendrive"]


def test_recovery_por_la_web(servidor):
    base, _ = servidor
    _post(base, "/api/arrancar", {})
    _, menu = _post(base, "/api/linea", {"texto": ":recovery"})
    assert "RECOVERY" in "\n".join(menu["lineas"])
    assert menu["modo"] == "recovery"

    _, estado = _post(base, "/api/linea", {"texto": "E"})
    assert estado["modo"] == "normal"
    diag = json.loads("\n".join(estado["lineas"]))
    assert "dispositivos" in diag


def test_salir_apaga_y_los_posteriores_avisan(servidor):
    base, _ = servidor
    _post(base, "/api/arrancar", {})
    _, cierre = _post(base, "/api/linea", {"texto": ":salir"})
    texto = "\n".join(cierre["lineas"])
    assert "Apagando IA-SO... hasta pronto." in texto
    assert "Sesión cerrada." in texto
    assert cierre["encendida"] is False

    _, despues = _post(base, "/api/linea", {"texto": "3+5"})
    assert any("apagada" in l.lower() for l in despues["lineas"])


def test_boton_de_apagado(servidor):
    base, _ = servidor
    _post(base, "/api/arrancar", {})
    _, apagado = _post(base, "/api/apagar", {})
    assert "Sesión cerrada." in "\n".join(apagado["lineas"])
    assert apagado["encendida"] is False

    # apagar dos veces: sin líneas (ya está apagada)
    _, de_nuevo = _post(base, "/api/apagar", {})
    assert de_nuevo["lineas"] == []


def test_arrancar_de_nuevo_tras_apagar_crea_sesion_nueva(servidor):
    base, _ = servidor
    _post(base, "/api/arrancar", {})
    _post(base, "/api/linea", {"texto": "3+5"})   # la 1.ª sesión atiende algo
    _post(base, "/api/apagar", {})
    _, segunda = _post(base, "/api/arrancar", {})
    assert "IA-SO BROODER EN MARCHA" in "\n".join(segunda["lineas"])
    # cada arranque es una sesión nueva desde el SSD congelado: la 1.ª
    # atendió 1 solicitud y esta nace limpia (no hereda contadores)
    assert segunda["resumen"] == "sin solicitudes atendidas todavía"


def test_cuerpo_invalido_400(servidor):
    base, _ = servidor
    import urllib.error

    peticion = urllib.request.Request(
        base + "/api/linea", data=b"no-json", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(peticion, timeout=10)
    assert info.value.code == 400


# ------------------------------------------------------------------
# sesión única: el lock serializa el acceso concurrente
# ------------------------------------------------------------------
def test_lineas_concurrentes_se_serializan(servidor):
    base, _ = servidor
    _post(base, "/api/arrancar", {})
    resultados: list = []
    errores: list = []

    def golpear():
        try:
            _, datos = _post(base, "/api/linea", {"texto": "3+5"})
            resultados.append(datos)
        except Exception as exc:  # pragma: no cover
            errores.append(exc)

    hilos = [threading.Thread(target=golpear) for _ in range(6)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join(timeout=60)

    assert not errores
    assert len(resultados) == 6
    for datos in resultados:
        assert datos["ok"] and "[ OK ]" in "\n".join(datos["lineas"])
