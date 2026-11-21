"""Ampliación: cribado virtual por similitud de Tanimoto (screening_ampliado.py)."""

import json

import numpy as np
import pytest
from conftest import TEST_URI, resultados

import figuras_screening as fig_s
import screening_ampliado as sa

RESUMEN = resultados('screening_resultados.json')


def test_la_figura_tiene_una_serie_por_base_y_panel():
    fig = fig_s.figura(RESUMEN)
    assert len(fig.data) == 4
    assert {t.name for t in fig.data} == {'CDS16', 'CDS29'}
    assert list(fig.data[0].x) == [0.3, 0.4, 0.6, 0.8]


def test_main_de_la_figura_escribe_los_ficheros(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'r.json').write_text(json.dumps(RESUMEN))
    monkeypatch.setattr(fig_s.go.Figure, 'write_image', lambda *a, **k: None)  # sin Chrome/Kaleido
    fig_s.main('r.json')
    assert (tmp_path / 'fig_screening.html').exists()


@pytest.fixture
def coleccion_sintetica(db_temporal):
    """300 moléculas con fingerprints aleatorios y su colección de frecuencias mfp_counts."""
    rng = np.random.default_rng(5)
    moleculas, fingerprints = [], []
    for i in range(300):
        bits = sorted({int(b) for b in rng.choice(64, size=rng.integers(4, 20), replace=False)})
        fingerprints.append(bits)
        moleculas.append(
            {
                '_id': f'M{i}',
                'smiles': 'C' * (i % 7 + 1),
                'class': 'Active',
                'mfp': {'bits': bits, 'count': len(bits)},
            }
        )
    db_temporal.molecules.insert_many(moleculas)
    todos = [b for bits in fingerprints for b in bits]
    cuentas = np.bincount(todos, minlength=64)
    db_temporal.mfp_counts.insert_many([{'_id': b, 'count': int(c)} for b, c in enumerate(cuentas)])
    return db_temporal, moleculas


@pytest.mark.mongo
@pytest.mark.parametrize('umbral', [0.9, 0.7, 0.5, 0.3, 0.2])
def test_la_agregacion_coincide_con_la_busqueda_exhaustiva(coleccion_sintetica, umbral):
    db, moleculas = coleccion_sintetica
    for consulta in moleculas[:15]:
        qfp = consulta['mfp']['bits']
        res, n_req, qmin, qmax = sa.similares_agregacion(db, qfp, umbral)
        exhaustiva = sa.similares_fuerza_bruta(db, qfp, umbral)
        assert {r['_id'] for r in res} == {r['_id'] for r in exhaustiva}
        assert consulta['_id'] in {r['_id'] for r in res}  # cada molécula se encuentra a sí misma
        assert n_req == len(qfp) - qmin + 1
        assert sa.candidatos(db, qfp, qmin, qmax) >= len(res)


@pytest.mark.mongo
def test_los_bits_requeridos_son_los_menos_frecuentes(coleccion_sintetica):
    db, _ = coleccion_sintetica
    qfp = list(range(10))
    req = sa.bits_requeridos(db, qfp, qmin=7)
    frecuencias = {c['_id']: c['count'] for c in db.mfp_counts.find({'_id': {'$in': qfp}})}
    assert len(req) == 4
    assert max(frecuencias[b] for b in req) <= min(frecuencias[b] for b in set(qfp) - set(req))


@pytest.mark.mongo
def test_main_reproduce_el_cribado_entregado(bases_moleculares, tmp_path, capsys):
    bases = [bases_moleculares['CDS16'], bases_moleculares['CDS29']]
    resumen = sa.main(TEST_URI, bases, salida=str(tmp_path / 'r.json'))
    assert json.loads((tmp_path / 'r.json').read_text()) == resumen
    claves = [
        'n_total',
        'threshold',
        'qmin',
        'qmax',
        'req_bits',
        'candidatos',
        'recuperadas',
        'mismo_resultado',
    ]
    assert [{k: r[k] for k in claves} for r in resumen] == [
        {k: r[k] for k in claves} for r in RESUMEN
    ]
    assert 'mismo resultado: True' in capsys.readouterr().out
