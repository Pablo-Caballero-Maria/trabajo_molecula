#!/usr/bin/env python3
"""
ADP - Practicas (Apartado I)
Ampliacion del ejemplo screening_Mol-database-only_mongo_p.py:
busqueda por similitud (Tanimoto) con el framework de agregacion de MongoDB,
sobre CDS16 y CDS29, con varios umbrales y comparacion contra la busqueda
exhaustiva (fuerza bruta) para validar el resultado y medir la mejora.

Autor: Pablo Caballero Maria
"""

import json
from time import time

import pymongo

CONNECTION = 'mongodb://adp:adp@127.0.0.1:27017/?authSource=admin'
QMOL = 'Nc1nc2c(ncn2C2OC(CO)C(O)C2O)c(=O)[nH]1'  # guanosina (molecula de consulta)
QFP = [
    71,
    75,
    80,
    90,
    128,
    147,
    152,
    194,
    210,
    222,
    233,
    314,
    334,
    345,
    356,
    378,
    428,
    433,
    454,
    463,
    492,
    508,
    519,
    577,
    610,
    633,
    650,
    656,
    672,
    697,
    700,
    721,
    740,
    750,
    774,
    807,
    820,
    831,
    849,
    897,
    932,
    935,
    1019,
]


def similares_agregacion(db, qfp, threshold):
    """Pipeline del ejemplo de la asignatura: count bounds + required bits + Tanimoto."""
    qn = len(qfp)
    qmin = int(qn * threshold)
    qmax = int(qn / threshold)
    reqbits = [
        c['_id']
        for c in db.mfp_counts.find({'_id': {'$in': qfp}}).sort('count', 1).limit(qn - qmin + 1)
    ]
    pipeline = [
        {'$match': {'mfp.count': {'$gte': qmin, '$lte': qmax}, 'mfp.bits': {'$in': reqbits}}},
        {
            '$project': {
                'tanimoto': {
                    '$let': {
                        'vars': {'common': {'$size': {'$setIntersection': ['$mfp.bits', qfp]}}},
                        'in': {
                            '$divide': [
                                '$$common',
                                {'$subtract': [{'$add': [qn, '$mfp.count']}, '$$common']},
                            ]
                        },
                    }
                },
                'smiles': 1,
                'class': 1,
            }
        },
        {'$match': {'tanimoto': {'$gte': threshold}}},
        {'$sort': {'tanimoto': -1}},
    ]
    return list(db.molecules.aggregate(pipeline)), len(reqbits), qmin, qmax


def similares_fuerza_bruta(db, qfp, threshold):
    """Recorrido completo de la coleccion, sin filtros previos: referencia de correccion."""
    q = set(qfp)
    res = []
    for doc in db.molecules.find({}, {'_id': 1, 'smiles': 1, 'mfp': 1}):
        b = set(doc['mfp']['bits'])
        common = len(q & b)
        t = common / (len(q) + len(b) - common)
        if t >= threshold:
            res.append({'_id': doc['_id'], 'tanimoto': t, 'smiles': doc['smiles']})
    res.sort(key=lambda r: -r['tanimoto'])
    return res


client = pymongo.MongoClient(CONNECTION)
resumen = []
print('Molecula de consulta:', QMOL, '| bits a 1:', len(QFP), '\n')

for db_name in ['CDS16', 'CDS29']:
    db = client[db_name]
    n_total = db.molecules.count_documents({})
    print('=' * 78)
    print('%s  (%d moleculas)' % (db_name, n_total))
    print('=' * 78)
    for threshold in [0.8, 0.6, 0.4, 0.3]:
        t0 = time()
        res, n_reqbits, qmin, qmax = similares_agregacion(db, QFP, threshold)
        t_agg = time() - t0

        t0 = time()
        res_fb = similares_fuerza_bruta(db, QFP, threshold)
        t_fb = time() - t0

        # numero de documentos que superan el filtro previo (candidatos evaluados)
        reqbits = [
            c['_id']
            for c in db.mfp_counts.find({'_id': {'$in': QFP}})
            .sort('count', 1)
            .limit(len(QFP) - qmin + 1)
        ]
        n_cand = db.molecules.count_documents(
            {'mfp.count': {'$gte': qmin, '$lte': qmax}, 'mfp.bits': {'$in': reqbits}}
        )

        coincide = [r['_id'] for r in res] == [r['_id'] for r in res_fb]
        resumen.append(
            {
                'db': db_name,
                'n_total': n_total,
                'threshold': threshold,
                'qmin': qmin,
                'qmax': qmax,
                'req_bits': n_reqbits,
                'candidatos': n_cand,
                'recuperadas': len(res),
                't_agregacion_ms': t_agg * 1000,
                't_fuerza_bruta_ms': t_fb * 1000,
                'mismo_resultado': coincide,
            }
        )
        print(
            '\n-- umbral T = %.1f  |  cota de bits [%d, %d]  |  required bits = %d'
            % (threshold, qmin, qmax, n_reqbits)
        )
        print(
            '   candidatos tras el filtro: %d de %d (%.1f %%)'
            % (n_cand, n_total, 100.0 * n_cand / n_total)
        )
        print(
            '   moleculas recuperadas: %d  |  agregacion %.1f ms  vs  fuerza bruta %.1f ms'
            ' (x%.1f)  |  mismo resultado: %s'
            % (len(res), t_agg * 1000, t_fb * 1000, t_fb / max(t_agg, 1e-9), coincide)
        )
        for r in res[:5]:
            print('     %.4f  %-10s %s' % (r['tanimoto'], r['_id'], r['smiles']))
        if len(res) > 5:
            print('     ... (%d mas)' % (len(res) - 5))

with open('screening_resultados.json', 'w') as f:
    json.dump(resumen, f, indent=1)
print('\nResumen guardado en screening_resultados.json')
client.close()
