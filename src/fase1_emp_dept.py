#!/usr/bin/env python3
"""
ADP - Prácticas (Apartado I) - Fase 1: Solución NoSQL MongoDB
Esquema EMP / DEPT con la relación reflexiva "Es jefe / Subordinado de".

Autor: Pablo Caballero María

Uso:  uv run python fase1_emp_dept.py   (desde src/; MONGO_URI permite usar otro servidor)
Crea la base de datos 'empresa', sus colecciones con validación,
carga los datos de prueba y ejecuta las consultas-informe.
"""

import os
from datetime import datetime

import pymongo
from pymongo.errors import WriteError

CONNECTION = os.environ.get('MONGO_URI', 'mongodb://adp:adp@127.0.0.1:27017/?authSource=admin')
DB_NAME = 'empresa'


# ---------------------------------------------------------------------------
# 1. ESTRUCTURA DE COLECCIONES (con validación $jsonSchema)
# ---------------------------------------------------------------------------
DEPT_SCHEMA = {
    '$jsonSchema': {
        'bsonType': 'object',
        'required': ['_id', 'dname', 'loc'],
        'additionalProperties': False,
        'properties': {
            '_id': {'bsonType': 'int', 'description': 'DEPTNO (clave primaria)'},
            'dname': {'bsonType': 'string', 'maxLength': 14},
            'loc': {'bsonType': 'string', 'maxLength': 13},
        },
    }
}

EMP_SCHEMA = {
    '$jsonSchema': {
        'bsonType': 'object',
        'required': ['_id', 'ename', 'job', 'hiredate', 'sal'],
        'additionalProperties': False,
        'properties': {
            '_id': {'bsonType': 'int', 'description': 'EMPNO (clave primaria)'},
            'ename': {'bsonType': 'string', 'maxLength': 10},
            'job': {'bsonType': 'string', 'maxLength': 9},
            'hiredate': {'bsonType': 'date'},
            'sal': {'bsonType': ['double', 'int'], 'minimum': 0},
            'comm': {'bsonType': ['double', 'int'], 'minimum': 0},
            'deptno': {
                'bsonType': 'int',
                'description': 'referencia a dept._id, cardinalidad (0,1)',
            },
            'mgr': {
                'bsonType': 'int',
                'description': 'referencia a emp._id del jefe, cardinalidad (0,1)',
            },
        },
    }
}


def crear_colecciones(db):
    """Colecciones dept y emp con validación estricta e índices sobre las referencias."""
    db.create_collection('dept', validator=DEPT_SCHEMA, validationLevel='strict')
    db.create_collection('emp', validator=EMP_SCHEMA, validationLevel='strict')
    # Índices sobre las referencias: aceleran $lookup y $graphLookup
    db.emp.create_index('deptno')
    db.emp.create_index('mgr')
    db.emp.create_index('job')


# ---------------------------------------------------------------------------
# 2. CARGA DE DATOS DE PRUEBA
# ---------------------------------------------------------------------------
def d(s):
    return datetime.strptime(s, '%Y-%m-%d')


DEPTS = [
    {'_id': 10, 'dname': 'ACCOUNTING', 'loc': 'NEW YORK'},
    {'_id': 20, 'dname': 'RESEARCH', 'loc': 'DALLAS'},
    {'_id': 30, 'dname': 'SALES', 'loc': 'CHICAGO'},
    {'_id': 40, 'dname': 'OPERATIONS', 'loc': 'BOSTON'},  # sin empleados
    {'_id': 50, 'dname': 'IOT LAB', 'loc': 'SEVILLA'},
]

# (empno, ename, job, mgr, hiredate, sal, comm, deptno); None = atributo ausente
EMPS_RAW = [
    (7839, 'KING', 'PRESIDENT', None, '1981-11-17', 5000, None, 10),
    (7566, 'JONES', 'MANAGER', 7839, '1981-04-02', 2975, None, 20),
    (7698, 'BLAKE', 'MANAGER', 7839, '1981-05-01', 2850, None, 30),
    (7782, 'CLARK', 'MANAGER', 7839, '1981-06-09', 2450, None, 10),
    (7788, 'SCOTT', 'ANALYST', 7566, '1987-04-19', 3000, None, 20),
    (7902, 'FORD', 'ANALYST', 7566, '1981-12-03', 3000, None, 20),
    (7369, 'SMITH', 'CLERK', 7902, '1980-12-17', 800, None, 20),
    (7876, 'ADAMS', 'CLERK', 7788, '1987-05-23', 1100, None, 20),
    (7499, 'ALLEN', 'SALESMAN', 7698, '1981-02-20', 1600, 300, 30),
    (7521, 'WARD', 'SALESMAN', 7698, '1981-02-22', 1250, 500, 30),
    (7654, 'MARTIN', 'SALESMAN', 7698, '1981-09-28', 1250, 1400, 30),
    (7844, 'TURNER', 'SALESMAN', 7698, '1981-09-08', 1500, 0, 30),
    (7900, 'JAMES', 'CLERK', 7698, '1981-12-03', 950, None, 30),
    (7934, 'MILLER', 'CLERK', 7782, '1982-01-23', 1300, None, 10),
    (8001, 'MORENO', 'MANAGER', 7839, '2022-09-01', 3100, None, 50),
    (8002, 'GARCIA', 'ANALYST', 8001, '2023-02-15', 2600, None, 50),
    (8003, 'ROMERO', 'ANALYST', 8001, '2024-10-07', 2400, None, 50),
    (8004, 'LOPEZ', 'CLERK', 7782, '2025-03-03', 1150, None, None),  # sin departamento
]


def documento_emp(empno, ename, job, mgr, hd, sal, comm, deptno):
    """Documento de un empleado: los atributos opcionales ausentes no se guardan."""
    doc = {'_id': empno, 'ename': ename, 'job': job, 'hiredate': d(hd), 'sal': sal}
    if comm is not None:
        doc['comm'] = comm
    if deptno is not None:
        doc['deptno'] = deptno
    if mgr is not None:
        doc['mgr'] = mgr
    return doc


def cargar_datos(db):
    db.dept.insert_many(DEPTS)
    db.emp.insert_many([documento_emp(*fila) for fila in EMPS_RAW])


def prueba_validacion(db):
    """Inserta un empleado con salario negativo y explica por qué lo rechaza el esquema."""
    try:
        db.emp.insert_one(
            {'_id': 9999, 'ename': 'ERROR', 'job': 'CLERK', 'hiredate': d('2025-01-01'), 'sal': -10}
        )
    except WriteError as e:
        detalles = e.details or {}
        regla = detalles['errInfo']['details']['schemaRulesNotSatisfied'][0]
        prop = regla['propertiesNotSatisfied'][0]
        return (
            detalles.get('errmsg'),
            f"(regla '{regla['operatorName']}', propiedad '{prop['propertyName']}': "
            f'{prop["details"][0]["operatorName"]} = {prop["details"][0]["specifiedAs"]})\n',
        )
    return None


def referencias_huerfanas(db):
    """Integridad referencial (MongoDB no la impone): comprobación explícita."""
    ids_dept = set(db.dept.distinct('_id'))
    ids_emp = set(db.emp.distinct('_id'))
    bad_dept = [
        e['_id'] for e in db.emp.find({'deptno': {'$exists': True}}) if e['deptno'] not in ids_dept
    ]
    bad_mgr = [e['_id'] for e in db.emp.find({'mgr': {'$exists': True}}) if e['mgr'] not in ids_emp]
    return bad_dept, bad_mgr


# ---------------------------------------------------------------------------
# 3. CONSULTAS - INFORMES
# ---------------------------------------------------------------------------
def titulo(t):
    print('=' * 78)
    print(t)
    print('=' * 78)


def tabla(filas, cols):
    anchos = [max(len(str(c)), *(len(str(f.get(c, ''))) for f in filas)) for c in cols]
    print('  '.join(str(c).ljust(a) for c, a in zip(cols, anchos, strict=True)))
    print('  '.join('-' * a for a in anchos))
    for f in filas:
        print('  '.join(str(f.get(c, '')).ljust(a) for c, a in zip(cols, anchos, strict=True)))
    print()


def concatenar(campo):
    """Expresión que une las cadenas de un array con ', ' ($reduce)."""
    return {
        '$reduce': {
            'input': campo,
            'initialValue': '',
            'in': {
                '$concat': [
                    '$$value',
                    {'$cond': [{'$eq': ['$$value', '']}, '', ', ']},
                    '$$this',
                ]
            },
        }
    }


def q1(db):
    """Q1. Listado de empleados con el nombre y la localidad de su departamento."""
    return list(
        db.emp.aggregate(
            [
                {
                    '$lookup': {
                        'from': 'dept',
                        'localField': 'deptno',
                        'foreignField': '_id',
                        'as': 'd',
                    }
                },
                {'$unwind': {'path': '$d', 'preserveNullAndEmptyArrays': True}},
                {
                    '$project': {
                        '_id': 0,
                        'empno': '$_id',
                        'ename': 1,
                        'job': 1,
                        'sal': 1,
                        'dname': {'$ifNull': ['$d.dname', '(sin dept.)']},
                        'loc': {'$ifNull': ['$d.loc', '-']},
                    }
                },
                {'$sort': {'dname': 1, 'sal': -1}},
            ]
        )
    )


def q2(db):
    """Q2. Informe por departamento (incluye los departamentos vacíos)."""
    return list(
        db.dept.aggregate(
            [
                {
                    '$lookup': {
                        'from': 'emp',
                        'localField': '_id',
                        'foreignField': 'deptno',
                        'as': 'emps',
                    }
                },
                {
                    '$project': {
                        '_id': 0,
                        'deptno': '$_id',
                        'dname': 1,
                        'loc': 1,
                        'n_emp': {'$size': '$emps'},
                        'masa_salarial': {'$sum': '$emps.sal'},
                        'sal_medio': {'$round': [{'$ifNull': [{'$avg': '$emps.sal'}, 0]}, 2]},
                        'sal_max': {'$ifNull': [{'$max': '$emps.sal'}, 0]},
                    }
                },
                {'$sort': {'deptno': 1}},
            ]
        )
    )


def q3(db):
    """Q3. Empleados con su jefe directo (auto-referencia)."""
    return list(
        db.emp.aggregate(
            [
                {
                    '$lookup': {
                        'from': 'emp',
                        'localField': 'mgr',
                        'foreignField': '_id',
                        'as': 'jefe',
                    }
                },
                {'$unwind': {'path': '$jefe', 'preserveNullAndEmptyArrays': True}},
                {
                    '$project': {
                        '_id': 0,
                        'empno': '$_id',
                        'ename': 1,
                        'job': 1,
                        'jefe': {'$ifNull': ['$jefe.ename', '(ninguno)']},
                        'job_jefe': {'$ifNull': ['$jefe.job', '-']},
                    }
                },
                {'$sort': {'jefe': 1, 'ename': 1}},
            ]
        )
    )


def q4(db):
    """Q4. Número de subordinados directos de cada jefe."""
    return list(
        db.emp.aggregate(
            [
                {'$match': {'mgr': {'$exists': True}}},
                {
                    '$group': {
                        '_id': '$mgr',
                        'n_subordinados': {'$sum': 1},
                        'subordinados': {'$push': '$ename'},
                    }
                },
                {'$lookup': {'from': 'emp', 'localField': '_id', 'foreignField': '_id', 'as': 'j'}},
                {'$unwind': '$j'},
                {
                    '$project': {
                        '_id': 0,
                        'empno': '$_id',
                        'jefe': '$j.ename',
                        'n_subordinados': 1,
                        'subordinados': concatenar('$subordinados'),
                    }
                },
                {'$sort': {'n_subordinados': -1, 'jefe': 1}},
            ]
        )
    )


def q5(db, ename='JONES'):
    """Q5. Jerarquía completa bajo un jefe con $graphLookup (subordinados a cualquier nivel)."""
    return list(
        db.emp.aggregate(
            [
                {'$match': {'ename': ename}},
                {
                    '$graphLookup': {
                        'from': 'emp',
                        'startWith': '$_id',
                        'connectFromField': '_id',
                        'connectToField': 'mgr',
                        'as': 'equipo',
                        'depthField': 'nivel',
                    }
                },
                {'$unwind': '$equipo'},
                {
                    '$project': {
                        '_id': 0,
                        'empno': '$equipo._id',
                        'ename': '$equipo.ename',
                        'job': '$equipo.job',
                        'nivel': {'$add': ['$equipo.nivel', 1]},
                    }
                },
                {'$sort': {'nivel': 1, 'ename': 1}},
            ]
        )
    )


def q6(db, ename='ADAMS'):
    """Q6. Cadena de mando ascendente de un empleado, hasta el presidente."""
    return list(
        db.emp.aggregate(
            [
                {'$match': {'ename': ename}},
                {
                    '$graphLookup': {
                        'from': 'emp',
                        'startWith': '$mgr',
                        'connectFromField': 'mgr',
                        'connectToField': '_id',
                        'as': 'cadena',
                        'depthField': 'nivel',
                    }
                },
                {'$unwind': '$cadena'},
                {
                    '$project': {
                        '_id': 0,
                        'nivel': {'$add': ['$cadena.nivel', 1]},
                        'ename': '$cadena.ename',
                        'job': '$cadena.job',
                    }
                },
                {'$sort': {'nivel': 1}},
            ]
        )
    )


def q7(db):
    """Q7. Empleados que ganan lo mismo o más que su jefe."""
    return list(
        db.emp.aggregate(
            [
                {
                    '$lookup': {
                        'from': 'emp',
                        'localField': 'mgr',
                        'foreignField': '_id',
                        'as': 'jefe',
                    }
                },
                {'$unwind': '$jefe'},
                {'$match': {'$expr': {'$gte': ['$sal', '$jefe.sal']}}},
                {
                    '$project': {
                        '_id': 0,
                        'ename': 1,
                        'sal': 1,
                        'jefe': '$jefe.ename',
                        'sal_jefe': '$jefe.sal',
                    }
                },
            ]
        )
    )


def q8(db):
    """Q8. Retribución total (sal + comm) de los vendedores."""
    vendedores = db.emp.find({'job': 'SALESMAN'}, {'_id': 0, 'ename': 1, 'sal': 1, 'comm': 1}).sort(
        'comm', -1
    )
    return [{**r, 'total': r['sal'] + r.get('comm', 0)} for r in vendedores]


def q9(db):
    """Q9. Contrataciones por año y puesto."""
    return list(
        db.emp.aggregate(
            [
                {
                    '$group': {
                        '_id': {'$year': '$hiredate'},
                        'n': {'$sum': 1},
                        'puestos': {'$addToSet': '$job'},
                    }
                },
                {
                    '$project': {
                        '_id': 0,
                        'anio': '$_id',
                        'n': 1,
                        'puestos': concatenar({'$sortArray': {'input': '$puestos', 'sortBy': 1}}),
                    }
                },
                {'$sort': {'anio': 1}},
            ]
        )
    )


def q10(db):
    """Q10. Casos límite de las cardinalidades mínimas 0: sin departamento, sin jefe, vacíos."""
    sin_dept = [e['ename'] for e in db.emp.find({'deptno': {'$exists': False}})]
    sin_jefe = [e['ename'] for e in db.emp.find({'mgr': {'$exists': False}})]
    dept_vacios = [
        r['dname']
        for r in db.dept.aggregate(
            [
                {
                    '$lookup': {
                        'from': 'emp',
                        'localField': '_id',
                        'foreignField': 'deptno',
                        'as': 'e',
                    }
                },
                {'$match': {'e': {'$size': 0}}},
            ]
        )
    ]
    return sin_dept, sin_jefe, dept_vacios


def q11(db):
    """Q11. Subida del 5 % a los CLERK de RESEARCH (updateMany con pipeline)."""
    res = db.emp.update_many(
        {'deptno': 20, 'job': 'CLERK'},
        [{'$set': {'sal': {'$round': [{'$multiply': ['$sal', 1.05]}, 2]}}}],
    )
    filas = list(db.emp.find({'deptno': 20, 'job': 'CLERK'}, {'_id': 0, 'ename': 1, 'sal': 1}))
    return res.modified_count, filas


def etapa_indice(db, filtro):
    """Etapa del plan ganador de una búsqueda: IXSCAN (con su índice) si usa uno."""
    stage = db.emp.find(filtro).explain()['queryPlanner']['winningPlan']
    while 'inputStage' in stage:
        if stage.get('stage') == 'IXSCAN':
            break
        stage = stage['inputStage']
    return stage.get('stage'), stage.get('indexName', '')


def main(connection=CONNECTION, db_name=DB_NAME):
    client = pymongo.MongoClient(connection)
    client.drop_database(db_name)  # la práctica se puede repetir desde cero
    db = client[db_name]

    crear_colecciones(db)
    cargar_datos(db)
    print(
        f'Cargados {db.dept.count_documents({})} departamentos y '
        f'{db.emp.count_documents({})} empleados.\n'
    )
    rechazo = prueba_validacion(db)
    if rechazo is not None:
        print('Prueba de validación -> inserción rechazada:', *rechazo)
    bad_dept, bad_mgr = referencias_huerfanas(db)
    print('Referencias deptno huérfanas:', bad_dept, '| referencias mgr huérfanas:', bad_mgr, '\n')

    titulo('Q1. Empleados con su departamento ($lookup emp -> dept)')
    tabla(q1(db), ['empno', 'ename', 'job', 'sal', 'dname', 'loc'])

    titulo('Q2. Resumen salarial por departamento ($lookup dept -> emp)')
    tabla(q2(db), ['deptno', 'dname', 'loc', 'n_emp', 'masa_salarial', 'sal_medio', 'sal_max'])

    titulo('Q3. Empleado y jefe directo (relación reflexiva "Subordinado de")')
    tabla(q3(db), ['empno', 'ename', 'job', 'jefe', 'job_jefe'])

    titulo('Q4. Jefes y número de subordinados directos ("Es jefe")')
    tabla(q4(db), ['empno', 'jefe', 'n_subordinados', 'subordinados'])

    titulo('Q5. Organigrama recursivo bajo JONES ($graphLookup)')
    tabla(q5(db), ['nivel', 'empno', 'ename', 'job'])

    titulo('Q6. Cadena de mando de ADAMS hasta la cima ($graphLookup ascendente)')
    tabla(q6(db), ['nivel', 'ename', 'job'])

    titulo('Q7. Empleados cuyo salario es igual o superior al de su jefe')
    tabla(q7(db), ['ename', 'sal', 'jefe', 'sal_jefe'])

    titulo('Q8. Retribución total de los SALESMAN (sal + comm)')
    tabla(q8(db), ['ename', 'sal', 'comm', 'total'])

    titulo('Q9. Contrataciones por año ($year + $group)')
    tabla(q9(db), ['anio', 'n', 'puestos'])

    titulo('Q10. Casos límite de las cardinalidades (0,1) y (0,N)')
    sin_dept, sin_jefe, dept_vacios = q10(db)
    print('Empleados sin departamento :', sin_dept)
    print('Empleados sin jefe         :', sin_jefe)
    print('Departamentos sin empleados:', dept_vacios, '\n')

    titulo('Q11. updateMany: +5 % de salario a los CLERK del departamento 20')
    modificados, filas = q11(db)
    print('Documentos modificados:', modificados)
    tabla(filas, ['ename', 'sal'])

    # Plan de ejecución: la búsqueda por deptno usa el índice
    print('Plan ganador para {deptno: 30}:', *etapa_indice(db, {'deptno': 30}))
    client.close()


if __name__ == '__main__':
    main()
