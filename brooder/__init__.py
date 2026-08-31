"""
IA-SO Brooder
=============

Un sistema operativo gobernado por una red neuronal que se
*incuba* (entrena por refuerzo en una máquina potente), se exporta
en una imagen SSD y *nace* en un PC con CPU y GPU, donde atiende
solicitudes del usuario administrando el hardware a través de un
contrato de primitivas.
"""
from brooder.constantes import NOMBRE_PROYECTO, VERSION

__version__ = VERSION
__all__ = ["NOMBRE_PROYECTO", "VERSION"]
