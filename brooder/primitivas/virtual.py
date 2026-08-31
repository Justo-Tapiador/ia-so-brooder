"""
PC Virtual — la máquina simulada de nacimiento
==============================================

Implementación de `InterfazPrimitivas` completamente en memoria.
Es la máquina sobre la que la incubadora entrena a Brooder y la
máquina por defecto al arrancar la IA-SO en modo demostración.

Semánticamente es IDÉNTICA a `PCReal`: mismas validaciones, mismos
errores, mismos límites. Por eso una IA incubada contra esta
máquina puede arrancar después contra la real.
"""
from __future__ import annotations

from brooder.primitivas.base import InstanteMaquina, MaquinaBase


class PCVirtual(MaquinaBase):
    """Un ordenador simulado con CPU, RAM, disco, pantalla, teclado,
    audio, GPU y una interfaz de red desactivada."""

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            "PCVirtual("
            f"acumulador={self._acumulador}, "
            f"pantalla={len(self._pantalla)} tk, "
            f"teclado={len(self._teclado)} tk, "
            f"cabezal_disco={self._disco_cabezal})"
        )
