"""Fase 1: esquema DEPT/EMP en MongoDB, carga, validación e informes Q1-Q11."""

from datetime import datetime

import pytest
from conftest import SRC, TEST_URI
from pymongo.errors import WriteError

import fase1_emp_dept as f1

EMPNOS = {fila[0] for fila in f1.EMPS_RAW}
DEPTNOS = {d['_id'] for d in f1.DEPTS}


# ---------------------------------------------------------------------------
# Datos y utilidades (sin servidor)
# ---------------------------------------------------------------------------
def test_d_convierte_fechas_iso():
    assert f1.d('1981-11-17') == datetime(1981, 11, 17)


def test_documento_emp_omite_los_atributos_ausentes():
    rey = f1.documento_emp(7839, 'KING', 'PRESIDENT', None, '1981-11-17', 5000, None, 10)
    assert rey == {
        '_id': 7839,
        'ename': 'KING',
        'job': 'PRESIDENT',
        'hiredate': datetime(1981, 11, 17),
        'sal': 5000,
        'deptno': 10,
    }
    vendedor = f1.documento_emp(7844, 'TURNER', 'SALESMAN', 7698, '1981-09-08', 1500, 0, 30)
    assert vendedor['comm'] == 0  # una comisión de 0 sí es un valor
    assert vendedor['mgr'] == 7698


def test_los_datos_de_prueba_son_coherentes():
    assert len(EMPNOS) == len(f1.EMPS_RAW) == 18
    for _, _, _, mgr, _, _, _, deptno in f1.EMPS_RAW:
        assert mgr is None or mgr in EMPNOS
        assert deptno is None or deptno in DEPTNOS
    # casos límite de las cardinalidades mínimas 0
    assert [fila[1] for fila in f1.EMPS_RAW if fila[7] is None] == ['LOPEZ']
    assert [fila[1] for fila in f1.EMPS_RAW if fila[3] is None] == ['KING']


@pytest.mark.parametrize(
    ('esquema', 'requeridos'),
    [
        (f1.DEPT_SCHEMA, ['_id', 'dname', 'loc']),
        (f1.EMP_SCHEMA, ['_id', 'ename', 'job', 'hiredate', 'sal']),
    ],
)
def test_esquemas_estrictos(esquema, requeridos):
    regla = esquema['$jsonSchema']
    assert regla['required'] == requeridos
    assert regla['additionalProperties'] is False
    assert set(requeridos) <= set(regla['properties'])


def test_tabla_alinea_las_columnas(capsys):
    f1.tabla([{'a': 1, 'b': 'xyz'}, {'a': 22}], ['a', 'b'])
    assert capsys.readouterr().out.splitlines() == ['a   b  ', '--  ---', '1   xyz', '22     ', '']


def test_titulo_enmarca_el_texto(capsys):
    f1.titulo('Q0')
    assert capsys.readouterr().out == '=' * 78 + '\nQ0\n' + '=' * 78 + '\n'


def test_concatenar_usa_reduce():
    expr = f1.concatenar('$x')['$reduce']
    assert expr['input'] == '$x'
    assert expr['initialValue'] == ''


# ---------------------------------------------------------------------------
# Contra un servidor MongoDB
# ---------------------------------------------------------------------------
@pytest.fixture
def empresa(db_temporal):
    f1.crear_colecciones(db_temporal)
    f1.cargar_datos(db_temporal)
    return db_temporal


@pytest.mark.mongo
def test_carga_y_validacion(empresa):
    assert empresa.dept.count_documents({}) == 5
    assert empresa.emp.count_documents({}) == 18
    assert {'deptno_1', 'mgr_1', 'job_1'} <= set(empresa.emp.index_information())
    mensaje, detalle = f1.prueba_validacion(empresa)
    assert mensaje == 'Document failed validation'
    assert "propiedad 'sal'" in detalle
    assert empresa.emp.count_documents({'_id': 9999}) == 0


@pytest.mark.mongo
@pytest.mark.parametrize(
    'documento',
    [
        {'_id': 1, 'ename': 'X', 'job': 'CLERK', 'sal': 10},  # falta hiredate
        {
            '_id': 2,
            'ename': 'X',
            'job': 'CLERK',
            'hiredate': datetime(2020, 1, 1),
            'sal': 1,
            'extra': 1,
        },
        {
            '_id': 3,
            'ename': 'NOMBRE_LARGO',
            'job': 'CLERK',
            'hiredate': datetime(2020, 1, 1),
            'sal': 1,
        },
        {'_id': '4', 'ename': 'X', 'job': 'CLERK', 'hiredate': datetime(2020, 1, 1), 'sal': 1},
    ],
)
def test_el_esquema_rechaza_documentos_invalidos(empresa, documento):
    with pytest.raises(WriteError):
        empresa.emp.insert_one(documento)


@pytest.mark.mongo
def test_sin_referencias_huerfanas(empresa):
    assert f1.referencias_huerfanas(empresa) == ([], [])
    huerfano = f1.documento_emp(1, 'X', 'CLERK', 2, '2020-01-01', 1, None, 99)
    empresa.emp.insert_one(huerfano)
    assert f1.referencias_huerfanas(empresa) == ([1], [1])


@pytest.mark.mongo
def test_q1_une_cada_empleado_con_su_departamento(empresa):
    filas = f1.q1(empresa)
    assert len(filas) == 18
    lopez = next(f for f in filas if f['ename'] == 'LOPEZ')
    assert (lopez['dname'], lopez['loc']) == ('(sin dept.)', '-')
    assert filas[0]['dname'] == '(sin dept.)'  # el orden por nombre de departamento


@pytest.mark.mongo
def test_q2_incluye_los_departamentos_vacios(empresa):
    filas = {f['dname']: f for f in f1.q2(empresa)}
    assert filas['OPERATIONS']['n_emp'] == 0
    assert filas['OPERATIONS']['sal_medio'] == 0
    assert filas['RESEARCH']['masa_salarial'] == 800 + 2975 + 3000 + 1100 + 3000
    assert sum(f['n_emp'] for f in filas.values()) == 17  # LOPEZ no tiene departamento


@pytest.mark.mongo
def test_q3_y_q4_resuelven_la_relacion_reflexiva(empresa):
    jefes = {f['ename']: f['jefe'] for f in f1.q3(empresa)}
    assert jefes['KING'] == '(ninguno)'
    assert jefes['BLAKE'] == 'KING'
    q4 = f1.q4(empresa)
    assert (q4[0]['jefe'], q4[0]['n_subordinados']) == ('BLAKE', 5)
    assert q4[0]['subordinados'].count(', ') == 4


@pytest.mark.mongo
def test_q5_recorre_toda_la_jerarquia(empresa):
    equipo = f1.q5(empresa)
    assert [(f['nivel'], f['ename']) for f in equipo] == [
        (1, 'FORD'),
        (1, 'SCOTT'),
        (2, 'ADAMS'),
        (2, 'SMITH'),
    ]


@pytest.mark.mongo
def test_q6_sube_hasta_el_presidente(empresa):
    assert [f['ename'] for f in f1.q6(empresa)] == ['SCOTT', 'JONES', 'KING']


@pytest.mark.mongo
def test_q7_q8_q9(empresa):
    assert {f['ename'] for f in f1.q7(empresa)} == {'SCOTT', 'FORD'}
    totales = {f['ename']: f['total'] for f in f1.q8(empresa)}
    assert totales == {'MARTIN': 2650, 'WARD': 1750, 'ALLEN': 1900, 'TURNER': 1500}
    anios = {f['anio']: f['n'] for f in f1.q9(empresa)}
    assert anios[1981] == 10
    assert sum(anios.values()) == 18


@pytest.mark.mongo
def test_q10_y_q11(empresa):
    assert f1.q10(empresa) == (['LOPEZ'], ['KING'], ['OPERATIONS'])
    modificados, filas = f1.q11(empresa)
    assert modificados == 2
    assert {f['ename']: f['sal'] for f in filas} == {'SMITH': 840.0, 'ADAMS': 1155.0}


@pytest.mark.mongo
def test_la_busqueda_por_departamento_usa_el_indice(empresa):
    assert f1.etapa_indice(empresa, {'deptno': 30}) == ('IXSCAN', 'deptno_1')


@pytest.mark.mongo
def test_main_reproduce_la_salida_entregada(mongo_client, capsys):
    nombre = 'molecula_test_empresa'
    try:
        f1.main(TEST_URI, nombre)
    finally:
        mongo_client.drop_database(nombre)
    esperado = (SRC / 'fase1_salida.txt').read_text(encoding='utf-8')
    assert capsys.readouterr().out == esperado
