"""Fase 2: disposición de los puntos del gráfico de enjambre (fase2_shap.py)."""

import numpy as np

import fase2_shap as fsh


def test_enjambre_apila_alrededor_de_la_fila():
    valores = np.r_[np.zeros(9), np.linspace(1, 2, 5)]
    capa = fsh.enjambre(valores, alto=0.4)
    assert capa.shape == valores.shape
    assert np.abs(capa).max() == 0.4
    # los nueve valores iguales se reparten por encima y por debajo de la fila
    assert (capa[:9] > 0).sum() == 4
    assert (capa[:9] < 0).sum() == 4


def test_enjambre_es_determinista_y_admite_un_solo_valor():
    valores = np.random.default_rng(0).normal(size=200)
    np.testing.assert_array_equal(fsh.enjambre(valores), fsh.enjambre(valores))
    np.testing.assert_array_equal(fsh.enjambre(np.array([3.0])), [0.0])
