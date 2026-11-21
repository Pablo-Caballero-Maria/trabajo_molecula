"""Fixtures compartidas.

Los tests marcados con ``mongo`` usan un servidor real (``MOLECULA_TEST_URI``, por defecto
``mongodb://localhost:27017``) y bases de datos temporales que se borran al terminar; si no hay
servidor, se saltan. Las bases moleculares CDS16 y CDS29 se restauran desde la copia de
seguridad de la asignatura (``instrucciones/dump.tar``) leyendo el BSON con PyMongo, sin
necesidad de ``mongorestore``.
"""

import json
import os
import tarfile
import uuid
from pathlib import Path

import bson
import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

TEST_URI = os.environ.get('MOLECULA_TEST_URI', 'mongodb://localhost:27017')
RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / 'src'
DUMP = RAIZ / 'instrucciones' / 'dump.tar'


def resultados(nombre):
    """Un fichero de resultados entregado (src/*_resultados.json)."""
    return json.loads((SRC / nombre).read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def mongo_client():
    client = MongoClient(TEST_URI, serverSelectionTimeoutMS=1500)
    try:
        client.admin.command('ping')
    except PyMongoError:
        pytest.skip(f'No hay servidor MongoDB en {TEST_URI}')
    yield client
    client.close()


@pytest.fixture
def db_temporal(mongo_client):
    """Base de datos vacía con un nombre único, que se borra al terminar el test."""
    nombre = f'molecula_test_{uuid.uuid4().hex[:8]}'
    yield mongo_client[nombre]
    mongo_client.drop_database(nombre)


@pytest.fixture(scope='session')
def bases_moleculares(mongo_client):
    """CDS16 y CDS29 restauradas del dump en bases temporales: {'CDS16': nombre, ...}."""
    sufijo = uuid.uuid4().hex[:8]
    nombres = {}
    with tarfile.open(DUMP) as tar:
        for miembro in tar.getmembers():
            if not miembro.name.endswith('.bson'):
                continue
            _, base, fichero = miembro.name.split('/')
            contenido = tar.extractfile(miembro)
            assert contenido is not None
            nombres[base] = f'{base}_test_{sufijo}'
            documentos = bson.decode_all(contenido.read())
            mongo_client[nombres[base]][fichero.removesuffix('.bson')].insert_many(documentos)
    yield nombres
    for nombre in nombres.values():
        mongo_client.drop_database(nombre)
