"""Fase 2: inyección de ruido y eliminación de outliers (fase2_ruido_outliers.py)."""

import numpy as np
import pytest

import fase2_ruido_outliers as ro


@pytest.fixture
def X():
    rng = np.random.default_rng(1)
    return (rng.random((400, 64)) < 0.1).astype(np.uint8)


def test_hay_doce_combinaciones():
    assert len(ro.RUIDOS) * len(ro.OUTLIERS) == 12
    assert ro.RUIDOS['Sin ruido'] is None
    assert ro.OUTLIERS['Sin eliminar outliers'] is None


def test_ruido_gaussiano(X):
    ruidoso = ro.ruido_gaussiano(X, np.random.default_rng(0))
    assert ruidoso.shape == X.shape
    assert np.std(ruidoso - X) == pytest.approx(ro.SIGMA, rel=0.05)


def test_ruido_volteo_invierte_una_fraccion_p(X):
    ruidoso = ro.ruido_volteo(X, np.random.default_rng(0))
    assert set(np.unique(ruidoso)) <= {0, 1}
    assert np.mean(ruidoso != X) == pytest.approx(ro.P_VOLTEO, rel=0.3)


def test_ruido_enmascarado_solo_apaga_bits(X):
    ruidoso = ro.ruido_enmascarado(X, np.random.default_rng(0))
    assert np.all(ruidoso <= X)  # nunca enciende un bit
    apagados = (X == 1) & (ruidoso == 0)
    assert apagados.sum() / X.sum() == pytest.approx(ro.P_MASCARA, rel=0.3)


def test_aumentar_duplica_el_entrenamiento(X):
    y = np.arange(len(X)) % 2
    assert ro.aumentar(None, X, y, np.random.default_rng(0))[0] is X
    Xa, ya = ro.aumentar(ro.ruido_volteo, X, y, np.random.default_rng(0))
    assert Xa.shape == (2 * len(X), X.shape[1])
    np.testing.assert_array_equal(ya, np.concatenate([y, y]))
    np.testing.assert_array_equal(Xa[: len(X)], X)


def test_distancias_jaccard():
    A = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    D = ro.distancias_jaccard(A)
    np.testing.assert_allclose(np.diag(D)[:2], 0, atol=1e-7)
    np.testing.assert_allclose(D, D.T)
    assert D[0, 1] == pytest.approx(1 - 1 / 3)  # 1 bit en común de 3 distintos
    assert D[0, 2] == pytest.approx(1.0)


@pytest.mark.parametrize('detector', [ro.outliers_iforest, ro.outliers_lof])
def test_los_detectores_marcan_la_contaminacion_prevista(X, detector):
    assert detector(X).mean() == pytest.approx(ro.CONTAMINACION, abs=0.01)


def test_conservadas_elimina_por_clase(X):
    y = np.r_[np.ones(40, dtype=int), np.zeros(360, dtype=int)]
    assert ro.conservadas(None, X, y).all()
    mascara = ro.conservadas(ro.outliers_iforest, X, y)
    # el 5 % de cada clase por separado: la minoritaria nunca se vacía
    assert (~mascara[:40]).sum() == 2
    assert (~mascara[40:]).sum() == 18
