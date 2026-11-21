"""Fase 2: selección e ingeniería de características (fase2_seleccion.py)."""

import numpy as np
import pytest
from conftest import resultados
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import fase2_seleccion as fs


@pytest.fixture(scope='module')
def datos():
    rng = np.random.default_rng(2)
    y = (rng.random(120) < 0.3).astype(int)
    X = (rng.random((120, 32)) < 0.15).astype(np.uint8)
    X[:, 0] = y  # un bit que separa las clases
    X[:, 1] = X[:, 2]  # dos bits redundantes
    return X, y


def test_tanimoto():
    A = np.array([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    B = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.uint8)
    S = fs.tanimoto(A, B)
    np.testing.assert_allclose(S, [[2 / 3, 1.0], [0.0, 0.0]])


def test_caracteristicas_similitud(datos):
    X, y = datos
    S = fs.caracteristicas_similitud(X, y, X, es_entrenamiento=True)
    assert S.shape == (len(X), 5) == (len(X), len(fs.nombres_similitud()))
    assert np.all(S[:, :4] <= 1)
    np.testing.assert_allclose(S[:, 4], S[:, 0] - S[:, 2])
    # sin descartar la diagonal, cada molécula sería idéntica a sí misma
    S_diag = fs.caracteristicas_similitud(X, y, X, es_entrenamiento=False)
    con_bits = X.sum(axis=1) > 0
    assert np.all(np.maximum(S_diag[con_bits, 0], S_diag[con_bits, 2]) == pytest.approx(1.0))


def test_cuantiles_imputa_con_la_mediana_del_entrenamiento():
    rng = np.random.default_rng(3)
    Da = rng.lognormal(size=(300, 3))
    Db = rng.lognormal(size=(50, 3))
    Db[0, 1] = np.nan
    Ta, Tb = fs.cuantiles(Da, Db)
    assert Ta.min() >= 0
    assert Ta.max() <= 1
    assert Tb[0, 1] == pytest.approx(0.5, abs=0.02)  # la mediana cae en el cuantil 0,5


def test_filtros_de_frecuencia_y_redundancia(datos):
    X, _ = datos
    X = X.copy()
    X[:, 5] = 0  # un bit que nunca aparece
    frecuentes = fs.filtro_frecuencia(X)
    assert 5 not in frecuentes
    no_redundantes = fs.filtro_redundancia(X, frecuentes)
    assert 1 in no_redundantes
    assert 2 not in no_redundantes  # duplicado de 1: se conserva solo el primero


def test_rankings_son_permutaciones_completas(datos):
    X, y = datos
    rf = RandomForestClassifier(n_estimators=30, random_state=0)
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
    orden, n_l1 = fs.rankings(X, y, rf, lr)
    assert set(orden) == {
        'Chi cuadrado',
        'Información mutua',
        'Importancia RF',
        'Logística L1',
        'RFE (envolvente)',
    }
    for ranking in orden.values():
        assert sorted(ranking) == list(range(X.shape[1]))
        assert ranking[0] == 0 or 0 in ranking[:3]  # el bit que separa las clases, arriba
    assert 0 < n_l1 <= X.shape[1]


def test_construir_espacios(datos):
    X, y = datos
    rng = np.random.default_rng(4)
    a, v = np.arange(90), np.arange(90, 120)
    completos = {
        'X': X,
        'C': X.astype(np.int16) * 2,
        'D': rng.normal(size=(len(X), 6)),
        'M': (rng.random((len(X), 10)) < 0.3).astype(np.uint8),
    }
    espacios = fs.construir_espacios(completos, a, v, y[a])
    assert len(espacios) == 13
    for nombre, (Ea, Ev) in espacios.items():
        assert Ea.shape[0] == len(a), nombre
        assert Ev.shape[0] == len(v), nombre
        assert Ea.shape[1] == Ev.shape[1], nombre
    assert espacios['Bits + similitud'][0].shape[1] == X.shape[1] + 5


def test_los_resultados_entregados_confirman_la_seleccion():
    res = resultados('seleccion_resultados.json')
    assert res, 'seleccion_resultados.json no debe estar vacío'


@pytest.mark.mongo
def test_rdkit_reproduce_los_fingerprints_de_la_bd(mongo_client, bases_moleculares):
    datos = fs.representaciones(mongo_client, bases_moleculares['CDS16'])
    n = len(datos['y'])
    assert n == 82
    assert datos['C'].shape == (n, 1024)
    assert datos['M'].shape == (n, 167)
    assert datos['D'].shape == (n, len(datos['nombres_desc']))
    # En CDS16 se eliminan los 252 bits que no aparecen en ninguna molécula: el Morgan de RDKit
    # coincide con mfp.bits en los 772 restantes y deja a cero los eliminados.
    columnas = np.array([int(nombre[2:]) - 1 for nombre in datos['nombres_bits']])
    presentes = (datos['C'] > 0).astype(np.uint8)
    np.testing.assert_array_equal(presentes[:, columnas], datos['X'])
    assert presentes[:, np.setdiff1d(np.arange(1024), columnas)].sum() == 0
