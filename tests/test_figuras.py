"""Figuras de la fase 2, reconstruidas a partir de los resultados entregados (src/*.json).

`guardar` se sustituye por una función que se queda con la figura, así que no se escribe nada ni
hace falta Chrome para exportar los PNG.
"""

import numpy as np
import optuna
import pytest
from conftest import resultados

import fase2_clasificacion as f2
import fase2_seleccion as fs
import fase2_shap as fsh

FASE2 = resultados('fase2_resultados.json')
SELECCION = resultados('seleccion_resultados.json')


@pytest.fixture
def guardadas(monkeypatch):
    figuras = {}

    def guardar(fig, nombre, ancho, alto):
        figuras[nombre] = (fig, ancho, alto)

    for modulo in (f2, fs, fsh):
        monkeypatch.setattr(modulo, 'guardar', guardar)
    return figuras


def test_figura_cv(guardadas):
    f2.figura_cv({base: FASE2[base] for base in ('CDS16', 'CDS29')})
    fig, ancho, alto = guardadas['fig_cv_modelos']
    assert (ancho, alto) == (640, 300)
    assert len(fig.data) == 4  # dos bases × (GMean, AUROC)
    rf = list(f2.modelos()).index('Random Forest')
    assert fig.data[0].x[rf] == pytest.approx(0.881, abs=5e-4)  # CDS16, GMean del Random Forest


def test_figura_optuna(guardadas):
    estudios = {}
    for nombre, res in FASE2['CDS29_holdout']['resultados'].items():
        estudio = optuna.create_study(direction='maximize')
        for valor in res['historial']:
            estado = (
                optuna.trial.TrialState.COMPLETE
                if valor is not None
                else optuna.trial.TrialState.PRUNED
            )
            estudio.add_trial(optuna.trial.create_trial(state=estado, value=valor))
        estudios[nombre] = estudio
    f2.figura_optuna(estudios)
    fig = guardadas['fig_optuna_cds29'][0]
    assert len(fig.data) == 2 * len(estudios)
    mejores = fig.data[1::2]
    for traza, res in zip(mejores, FASE2['CDS29_holdout']['resultados'].values(), strict=True):
        assert traza.y[-1] == pytest.approx(res['gmean_cv_interno'])  # el escalón acaba en el mejor
        assert list(traza.y) == sorted(traza.y)


def test_figura_roc(guardadas):
    curvas = {'A': (np.array([0, 0.5, 1]), np.array([0, 0.8, 1]), 0.65)}
    f2.figura_roc(curvas)
    fig = guardadas['fig_roc_cds29'][0]
    assert fig.data[1].name == 'A (AUC = 0.650)'
    assert fig.layout.yaxis.scaleanchor == 'x'


def test_figura_seleccion(guardadas):
    curvas = {}
    for clave, valor in SELECCION['seleccion'].items():
        modelo, criterio, bits = clave.split(' | ')
        curvas[(modelo, criterio, int(bits.split()[0]))] = (valor['gmean'], valor['gmean_std'])
    base = {m: (SELECCION['ingenieria'][f'{m} | {fs.BASE}']['gmean'], 0.0) for m in fs.MODELOS}
    criterios = list(SELECCION['bits_estables'])
    fs.figura_seleccion(curvas, base, criterios)
    fig = guardadas['fig_seleccion_cds29'][0]
    assert len(fig.data) == len(fs.MODELOS) * len(criterios)
    assert list(fig.data[0].x) == fs.KS
    textos = [a.text for a in fig.layout.annotations]
    assert 'Los 1024 bits: 0.625' in textos


def test_figura_ingenieria(guardadas):
    gm, dimensiones, nombres = {}, {}, []
    for clave, valor in SELECCION['ingenieria'].items():
        modelo, nombre = clave.split(' | ')
        gm[(modelo, nombre)] = (valor['gmean'], valor['gmean_std'])
        dimensiones[nombre] = valor['n_variables']
        if nombre not in nombres:
            nombres.append(nombre)
    fs.figura_ingenieria(gm, nombres, dimensiones)
    fig = guardadas['fig_ingenieria_cds29'][0]
    assert [t.name for t in fig.data] == fs.MODELOS
    lr = fig.data[1]
    assert max(lr.x) == pytest.approx(gm[('LogReg (L2)', 'Bits + similitud')][0])  # la mejor


def test_figura_beeswarm(guardadas):
    rng = np.random.default_rng(0)
    X = (rng.random((50, 6)) < 0.4).astype(np.uint8)
    sv = rng.normal(size=(50, 6)) * X
    nombres = np.array([f'FP{i}' for i in range(6)])
    ids = np.array([f'M{i}' for i in range(50)])
    fsh.figura_beeswarm(sv, X, nombres, ids, top=[3, 1])
    fig, _, alto = guardadas['fig_shap_cds29']
    assert alto == 120 + 26 * 2
    assert sum(len(t.x) for t in fig.data) == 2 * 50  # cada molécula una vez por bit mostrado
    assert list(fig.layout.yaxis.ticktext) == ['FP3', 'FP1']
