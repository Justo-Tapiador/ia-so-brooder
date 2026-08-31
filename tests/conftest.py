"""Fixtures compartidas de la suite de tests."""
import sys
from pathlib import Path

import pytest

# permitir ejecutar los tests desde la raíz del repo sin instalar
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import random  # noqa: E402

from brooder.constantes import Tarea  # noqa: E402


@pytest.fixture
def rng():
    return random.Random(20240707)


@pytest.fixture
def semillas_todas_las_tareas():
    return list(Tarea)
