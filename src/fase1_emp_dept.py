#!/usr/bin/env python3
"""
ADP - Prácticas (Apartado I) - Fase 1: Solución NoSQL MongoDB
Esquema EMP / DEPT con la relación reflexiva "Es jefe / Subordinado de".

Autor: Pablo Caballero María

Uso:  python3 fase1_emp_dept.py
Crea la base de datos 'empresa', sus colecciones con validación,
carga los datos de prueba y ejecuta las consultas-informe.
"""

from datetime import datetime

import pymongo
from pymongo.errors import WriteError

CONNECTION = 'mongodb://adp:adp@127.0.0.1:27017/?authSource=admin'
DB_NAME = 'empresa'

client = pymongo.MongoClient(CONNECTION)
client.drop_database(DB_NAME)  # la práctica se puede repetir desde cero
db = client[DB_NAME]

# ---------------------------------------------------------------------------
# 1. ESTRUCTURA DE COLECCIONES (con validación $jsonSchema)
# ---------------------------------------------------------------------------
dept_schema = {
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

emp_schema = {
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

db.create_collection('dept', validator=dept_schema, validationLevel='strict')
db.create_collection('emp', validator=emp_schema, validationLevel='strict')

# Índices sobre las referencias: aceleran $lookup y $graphLookup
db.emp.create_index('deptno')
db.emp.create_index('mgr')
db.emp.create_index('job')


# ---------------------------------------------------------------------------
# 2. CARGA DE DATOS DE PRUEBA
# ---------------------------------------------------------------------------
def d(s):
    return datetime.strptime(s, '%Y-%m-%d')


depts = [
    {'_id': 10, 'dname': 'ACCOUNTING', 'loc': 'NEW YORK'},
    {'_id': 20, 'dname': 'RESEARCH', 'loc': 'DALLAS'},
    {'_id': 30, 'dname': 'SALES', 'loc': 'CHICAGO'},
    {'_id': 40, 'dname': 'OPERATIONS', 'loc': 'BOSTON'},  # sin empleados
    {'_id': 50, 'dname': 'IOT LAB', 'loc': 'SEVILLA'},
]

# (empno, ename, job, mgr, hiredate, sal, comm, deptno); None = atributo ausente
emps_raw = [
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

emps = []
for empno, ename, job, mgr, hd, sal, comm, deptno in emps_raw:
    doc = {'_id': empno, 'ename': ename, 'job': job, 'hiredate': d(hd), 'sal': sal}
    if comm is not None:
        doc['comm'] = comm
    if deptno is not None:
        doc['deptno'] = deptno
    if mgr is not None:
        doc['mgr'] = mgr
    emps.append(doc)

db.dept.insert_many(depts)
db.emp.insert_many(emps)
print(
    f'Cargados {db.dept.count_documents({})} departamentos y '
    f'{db.emp.count_documents({})} empleados.\n'
)

# Validación: un documento que incumple el esquema es rechazado
try:
    db.emp.insert_one(
        {'_id': 9999, 'ename': 'ERROR', 'job': 'CLERK', 'hiredate': d('2025-01-01'), 'sal': -10}
    )
except WriteError as e:
    detalles = e.details or {}
    regla = detalles['errInfo']['details']['schemaRulesNotSatisfied'][0]
    prop = regla['propertiesNotSatisfied'][0]
    print(
        'Prueba de validación -> inserción rechazada:',
        detalles.get('errmsg'),
        f"(regla '{regla['operatorName']}', propiedad '{prop['propertyName']}': "
        f'{prop["details"][0]["operatorName"]} = {prop["details"][0]["specifiedAs"]})\n',
    )

# Integridad referencial (MongoDB no la impone): comprobación explícita
ids_dept = set(db.dept.distinct('_id'))
ids_emp = set(db.emp.distinct('_id'))
bad_dept = [
    e['_id'] for e in db.emp.find({'deptno': {'$exists': True}}) if e['deptno'] not in ids_dept
]
bad_mgr = [e['_id'] for e in db.emp.find({'mgr': {'$exists': True}}) if e['mgr'] not in ids_emp]
print('Referencias deptno huérfanas:', bad_dept, '| referencias mgr huérfanas:', bad_mgr, '\n')


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


# Q1. Listado de empleados con el nombre y la localidad de su departamento
titulo('Q1. Empleados con su departamento ($lookup emp -> dept)')
q1 = db.emp.aggregate(
    [
        {'$lookup': {'from': 'dept', 'localField': 'deptno', 'foreignField': '_id', 'as': 'd'}},
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
tabla(list(q1), ['empno', 'ename', 'job', 'sal', 'dname', 'loc'])

# Q2. Informe por departamento (incluye los departamentos vacíos)
titulo('Q2. Resumen salarial por departamento ($lookup dept -> emp)')
q2 = db.dept.aggregate(
    [
        {'$lookup': {'from': 'emp', 'localField': '_id', 'foreignField': 'deptno', 'as': 'emps'}},
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
tabla(list(q2), ['deptno', 'dname', 'loc', 'n_emp', 'masa_salarial', 'sal_medio', 'sal_max'])

# Q3. Empleados con su jefe directo (auto-referencia)
titulo('Q3. Empleado y jefe directo (relación reflexiva "Subordinado de")')
q3 = db.emp.aggregate(
    [
        {'$lookup': {'from': 'emp', 'localField': 'mgr', 'foreignField': '_id', 'as': 'jefe'}},
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
tabla(list(q3), ['empno', 'ename', 'job', 'jefe', 'job_jefe'])

# Q4. Número de subordinados directos de cada jefe
titulo('Q4. Jefes y número de subordinados directos ("Es jefe")')
q4 = db.emp.aggregate(
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
                'subordinados': {
                    '$reduce': {
                        'input': '$subordinados',
                        'initialValue': '',
                        'in': {
                            '$concat': [
                                '$$value',
                                {'$cond': [{'$eq': ['$$value', '']}, '', ', ']},
                                '$$this',
                            ]
                        },
                    }
                },
            }
        },
        {'$sort': {'n_subordinados': -1, 'jefe': 1}},
    ]
)
tabla(list(q4), ['empno', 'jefe', 'n_subordinados', 'subordinados'])

# Q5. Jerarquía completa bajo JONES con $graphLookup (subordinados a cualquier nivel)
titulo('Q5. Organigrama recursivo bajo JONES ($graphLookup)')
q5 = list(
    db.emp.aggregate(
        [
            {'$match': {'ename': 'JONES'}},
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
tabla(q5, ['nivel', 'empno', 'ename', 'job'])

# Q6. Cadena de mando ascendente de un empleado (de ADAMS hasta el presidente)
titulo('Q6. Cadena de mando de ADAMS hasta la cima ($graphLookup ascendente)')
q6 = list(
    db.emp.aggregate(
        [
            {'$match': {'ename': 'ADAMS'}},
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
tabla(q6, ['nivel', 'ename', 'job'])

# Q7. Empleados que ganan más que su jefe
titulo('Q7. Empleados cuyo salario es igual o superior al de su jefe')
q7 = db.emp.aggregate(
    [
        {'$lookup': {'from': 'emp', 'localField': 'mgr', 'foreignField': '_id', 'as': 'jefe'}},
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
tabla(list(q7), ['ename', 'sal', 'jefe', 'sal_jefe'])

# Q8. Retribución total (sal + comm) de los vendedores
titulo('Q8. Retribución total de los SALESMAN (sal + comm)')
q8 = db.emp.find({'job': 'SALESMAN'}, {'_id': 0, 'ename': 1, 'sal': 1, 'comm': 1}).sort('comm', -1)
filas = [{**r, 'total': r['sal'] + r.get('comm', 0)} for r in q8]
tabla(filas, ['ename', 'sal', 'comm', 'total'])

# Q9. Contrataciones por año y puesto
titulo('Q9. Contrataciones por año ($year + $group)')
q9 = db.emp.aggregate(
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
                'puestos': {
                    '$reduce': {
                        'input': {'$sortArray': {'input': '$puestos', 'sortBy': 1}},
                        'initialValue': '',
                        'in': {
                            '$concat': [
                                '$$value',
                                {'$cond': [{'$eq': ['$$value', '']}, '', ', ']},
                                '$$this',
                            ]
                        },
                    }
                },
            }
        },
        {'$sort': {'anio': 1}},
    ]
)
tabla(list(q9), ['anio', 'n', 'puestos'])

# Q10. Comprobación de cardinalidades mínimas 0
titulo('Q10. Casos límite de las cardinalidades (0,1) y (0,N)')
sin_dept = [e['ename'] for e in db.emp.find({'deptno': {'$exists': False}})]
sin_jefe = [e['ename'] for e in db.emp.find({'mgr': {'$exists': False}})]
dept_vacios = [
    r['dname']
    for r in db.dept.aggregate(
        [
            {'$lookup': {'from': 'emp', 'localField': '_id', 'foreignField': 'deptno', 'as': 'e'}},
            {'$match': {'e': {'$size': 0}}},
        ]
    )
]
print('Empleados sin departamento :', sin_dept)
print('Empleados sin jefe         :', sin_jefe)
print('Departamentos sin empleados:', dept_vacios, '\n')

# Q11. Actualización: subida del 5 % a los CLERK de RESEARCH y verificación
titulo('Q11. updateMany: +5 % de salario a los CLERK del departamento 20')
res = db.emp.update_many(
    {'deptno': 20, 'job': 'CLERK'},
    [{'$set': {'sal': {'$round': [{'$multiply': ['$sal', 1.05]}, 2]}}}],
)
print('Documentos modificados:', res.modified_count)
tabla(
    list(db.emp.find({'deptno': 20, 'job': 'CLERK'}, {'_id': 0, 'ename': 1, 'sal': 1})),
    ['ename', 'sal'],
)

# Plan de ejecución: la búsqueda por deptno usa el índice
plan = db.emp.find({'deptno': 30}).explain()
stage = plan['queryPlanner']['winningPlan']
while 'inputStage' in stage:
    if stage.get('stage') == 'IXSCAN':
        break
    stage = stage['inputStage']
print('Plan ganador para {deptno: 30}:', stage.get('stage'), stage.get('indexName', ''))

client.close()
