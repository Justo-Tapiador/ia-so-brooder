"""Tests del contrato de primitivas y de la máquina virtual."""
import pytest

from brooder.constantes import (
    ARG_BUS,
    N_RANURAS_DISCO,
    Primitiva,
    TOKEN_A,
    TOKEN_MAS,
    texto_a_tokens,
    tokens_a_texto,
)
from brooder.primitivas.virtual import PCVirtual


@pytest.fixture
def maquina():
    m = PCVirtual()
    m.reiniciar_registros()
    return m


# ------------------------------------------------------------------
# tokens
# ------------------------------------------------------------------
def test_ida_y_vuelta_tokens():
    assert texto_a_tokens("HOLA+5!") == [
        ord("H") - 55, ord("O") - 55, ord("L") - 55, ord("A") - 55,
        TOKEN_MAS, 5, 37,
    ]
    assert tokens_a_texto(texto_a_tokens("ABC123")) == "ABC123"


# ------------------------------------------------------------------
# teclado y bus
# ------------------------------------------------------------------
def test_leer_teclado_extrae_en_orden(maquina):
    maquina.escribir_teclado([1, 2, 3])
    assert maquina.leer_teclado()
    instante = maquina.instante()
    assert instante.bus_valor == 1 and instante.bus_valido
    assert maquina.leer_teclado()
    assert maquina.instante().bus_valor == 2


def test_leer_teclado_vacio_da_error(maquina):
    assert not maquina.leer_teclado()
    assert maquina.instante().ultimo_error


# ------------------------------------------------------------------
# pantalla
# ------------------------------------------------------------------
def test_mostrar_desde_bus(maquina):
    maquina.escribir_teclado([TOKEN_A])
    maquina.leer_teclado()
    assert maquina.mostrar_en_pantalla(ARG_BUS)
    assert maquina.instante().pantalla == [TOKEN_A]


def test_mostrar_literal_rechazado_por_el_kernel(maquina):
    # las primitivas de datos exigen BUS: contrato de tipos
    assert not maquina.ejecutar(Primitiva.MOSTRAR_EN_PANTALLA, TOKEN_A)
    assert "BUS" in maquina.instante().ultimo_error


def test_usar_gpu_vacia_la_pantalla(maquina):
    maquina.escribir_teclado([TOKEN_A])
    maquina.leer_teclado()
    maquina.mostrar_en_pantalla(ARG_BUS)
    assert maquina.instante().pantalla == [TOKEN_A]
    assert maquina.usar_gpu()
    assert maquina.instante().pantalla == []


# ------------------------------------------------------------------
# CPU
# ------------------------------------------------------------------
def test_cpu_suma_con_bus(maquina):
    for digito in (3, 5):
        maquina.escribir_teclado([digito])
        maquina.leer_teclado()
        assert maquina.cpu_sumar(ARG_BUS)
    assert maquina.instante().acumulador == 8


def test_cpu_cociente_y_resto(maquina):
    maquina.escribir_teclado([7, 8])
    maquina.leer_teclado()
    maquina.cpu_sumar(ARG_BUS)
    maquina.leer_teclado()
    maquina.cpu_sumar(ARG_BUS)
    assert maquina.instante().acumulador == 15
    assert maquina.cpu_cociente()
    assert maquina.instante().bus_valor == 1
    assert maquina.cpu_resto()
    assert maquina.instante().bus_valor == 5


# ------------------------------------------------------------------
# disco
# ------------------------------------------------------------------
def test_disco_escribir_y_leer(maquina):
    maquina.escribir_teclado([4, TOKEN_A])
    maquina.leer_teclado()
    assert maquina.mover_cabezal_disco(ARG_BUS)  # cabezal -> 4
    maquina.leer_teclado()
    assert maquina.escribir_disco(ARG_BUS)       # disco[4] = 'A'
    instante = maquina.instante()
    assert instante.disco_contenido[4] == TOKEN_A
    assert maquina.leer_disco()
    assert maquina.instante().bus_valor == TOKEN_A


def test_disco_direccion_invalida(maquina):
    assert not maquina.mover_cabezal_disco(N_RANURAS_DISCO + 5)


def test_direccion_literal_permitida(maquina):
    # direccionamiento acepta literales (0..9): contrato de tipos
    assert maquina.mover_cabezal_disco(7)
    assert maquina.instante().disco_cabezal == 7


# ------------------------------------------------------------------
# memoria
# ------------------------------------------------------------------
def test_memoria_escribir_y_leer(maquina):
    maquina.escribir_teclado([2])
    maquina.leer_teclado()
    assert maquina.mover_puntero_memoria(ARG_BUS)
    # la validación de tipos la hace el KERNEL (ejecutar), como un syscall
    assert not maquina.ejecutar(Primitiva.ESCRIBIR_MEMORIA, 9)
    assert maquina.instante().ultimo_error
    maquina.escribir_teclado([TOKEN_A])
    maquina.leer_teclado()
    assert maquina.escribir_memoria(ARG_BUS)
    assert maquina.leer_memoria()
    assert maquina.instante().bus_valor == TOKEN_A


# ------------------------------------------------------------------
# reinicios
# ------------------------------------------------------------------
def test_reiniciar_registros_conserva_disco(maquina):
    maquina.mover_cabezal_disco(3)
    maquina.escribir_teclado([TOKEN_A])
    maquina.leer_teclado()
    maquina.escribir_disco(ARG_BUS)
    maquina.reiniciar_registros()
    instante = maquina.instante()
    assert instante.acumulador == 0
    assert instante.pantalla == []
    assert instante.disco_contenido[3] == TOKEN_A  # persiste


# ------------------------------------------------------------------
# red desactivada (seguridad)
# ------------------------------------------------------------------
def test_red_desactivada(maquina):
    assert not maquina.leer_red()
    assert "red" in maquina.instante().ultimo_error.lower()
