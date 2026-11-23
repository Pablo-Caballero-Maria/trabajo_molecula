"""Esquema tipado de las colecciones de MongoDB del trabajo y acceso tipado a ellas.

Cada colección se usa como un ``Collection[...]`` de PyMongo parametrizado con su tipo de documento
(un ``TypedDict``), de modo que ``ty`` comprueba las inserciones y los campos que se leen de los
resultados de cada consulta.

* Fase 1, base ``empresa``: ``dept`` (:class:`Dept`) y ``emp`` (:class:`Emp`), con los mismos
  campos que exigen sus validadores ``$jsonSchema`` (fase1_emp_dept.py).
* Fase 2, bases ``CDS16`` y ``CDS29``: ``molecules`` (:class:`Molecula`) y ``mfp_counts``
  (:class:`FrecuenciaBit`), las del volcado de la asignatura (instrucciones/dump.tar).
"""

# Sin `from __future__ import annotations`: TypedDict debe ver NotRequired al crear la clase.
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict, cast

from pymongo.collection import Collection
from pymongo.database import Database


class Dept(TypedDict):
    _id: int  # DEPTNO
    dname: str
    loc: str


class Emp(TypedDict):
    _id: int  # EMPNO
    ename: str
    job: str
    hiredate: datetime
    sal: int | float
    # Atributos opcionales: si no tienen valor, no se guardan
    comm: NotRequired[int | float]
    deptno: NotRequired[int]  # referencia a dept._id
    mgr: NotRequired[int]  # referencia a emp._id del jefe


class Huella(TypedDict):
    """Morgan fingerprint de 1024 bits: las posiciones a 1 y cuántas son."""

    bits: list[int]
    count: int


# Una molécula del volcado. Su clase de actividad está en el campo 'class', una palabra reservada
# de Python: por eso se declara con la sintaxis funcional de TypedDict.
Molecula = TypedDict(
    'Molecula',
    {'_id': str, 'smiles': str, 'class': Literal['Active', 'Inactive'], 'mfp': Huella},
)


class Similar(TypedDict):
    """Una molécula encontrada por el cribado, con su similitud de Tanimoto con la consulta."""

    _id: str
    tanimoto: float
    smiles: str


class FrecuenciaBit(TypedDict):
    """Cuántas moléculas tienen a 1 cada bit (``_id``): ordena los bits requeridos del cribado."""

    _id: int
    count: int


#: Una base de datos de MongoDB (sus documentos sin tipar: cada colección se tipa al pedirla).
type BD = Database[dict[str, Any]]


def coleccion[T: Mapping[str, Any]](
    db: Database[dict[str, Any]], nombre: str, _documento: type[T]
) -> Collection[T]:
    """La colección ``nombre`` con documentos de tipo ``T`` (las de PyMongo son genéricas)."""
    return cast('Collection[T]', db.get_collection(nombre))


def departamentos(db: BD) -> Collection[Dept]:
    """La colección dept de la base empresa."""
    return coleccion(db, 'dept', Dept)


def empleados(db: BD) -> Collection[Emp]:
    """La colección emp de la base empresa."""
    return coleccion(db, 'emp', Emp)


def moleculas(db: BD) -> Collection[Molecula]:
    """La colección molecules de CDS16 o CDS29."""
    return coleccion(db, 'molecules', Molecula)


def frecuencias(db: BD) -> Collection[FrecuenciaBit]:
    """La colección mfp_counts de CDS16 o CDS29."""
    return coleccion(db, 'mfp_counts', FrecuenciaBit)
