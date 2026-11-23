"""El esquema tipado (src/documentos.py): coincide con los validadores y con el volcado."""

from typing import Any, cast, get_type_hints

from pymongo import MongoClient

import fase1_emp_dept as f1
from documentos import Dept, Emp, Molecula, departamentos, empleados, frecuencias, moleculas


def test_emp_y_dept_coinciden_con_sus_validadores():
    for tipo, esquema in ((Dept, f1.DEPT_SCHEMA), (Emp, f1.EMP_SCHEMA)):
        js = cast('dict[str, Any]', esquema['$jsonSchema'])
        assert set(get_type_hints(tipo)) == set(js['properties'])
        assert tipo.__required_keys__ == set(js['required'])


def test_los_documentos_de_la_carga_son_del_tipo_declarado():
    for fila in f1.EMPS_RAW:
        doc = f1.documento_emp(*fila)
        assert Emp.__required_keys__ <= set(doc) <= set(get_type_hints(Emp))
    assert all(set(d) == set(get_type_hints(Dept)) for d in f1.DEPTS)


def test_cada_coleccion_con_su_nombre():
    client: MongoClient = MongoClient(connect=False)
    try:
        assert [c.name for c in (departamentos(client.e), empleados(client.e))] == ['dept', 'emp']
        assert [c.name for c in (moleculas(client.m), frecuencias(client.m))] == [
            'molecules',
            'mfp_counts',
        ]
    finally:
        client.close()
    assert set(get_type_hints(Molecula)) == {'_id', 'smiles', 'class', 'mfp'}
