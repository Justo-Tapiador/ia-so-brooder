"""Capa de primitivas de hardware de IA-SO Brooder.

La IA nunca accede al hardware: solicita primitivas y el núcleo las
ejecuta a través de una implementación de `InterfazPrimitivas`.
"""
from brooder.primitivas.base import InstanteMaquina, InterfazPrimitivas, MaquinaBase
from brooder.primitivas.reales import PCReal
from brooder.primitivas.virtual import PCVirtual

__all__ = [
    "InstanteMaquina",
    "InterfazPrimitivas",
    "MaquinaBase",
    "PCReal",
    "PCVirtual",
]
