"""Fase 2: métricas, modelos, validación cruzada y Optuna (fase2_clasificacion.py)."""

import numpy as np
import optuna
import pytest
from conftest import resultados
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import fase2_clasificacion as f2

RESULTADOS = resultados('fase2_resultados.json')


@pytest.fixture(scope='module')
def datos():
    """Fingerprints sintéticos desbalanceados (25 % de activas) con señal en los primeros bits."""
    rng = np.random.default_rng(0)
    y = (rng.random(160) < 0.25).astype(int)
    X = (rng.random((160, 40)) < 0.08).astype(np.uint8)
    X[:, :4] |= (rng.random((160, 4)) < 0.7 * y[:, None]).astype(np.uint8)
    return X, y


def test_gmean():
    assert f2.gmean([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert f2.gmean([0, 0, 1, 1], [1, 1, 1, 1]) == 0.0  # sensibilidad 1, especificidad 0
    assert f2.gmean([0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0]) == pytest.approx(np.sqrt(0.5 * 0.5))


def test_metricas_de_una_matriz_de_confusion_conocida():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    m = f2.metricas(y_true, y_pred, y_pred.astype(float))
    assert m['Sensitivity'] == pytest.approx(0.75)
    assert m['Specificity'] == pytest.approx(4 / 6)
    assert m['GMean'] == pytest.approx(np.sqrt(0.75 * 4 / 6))
    assert m['Accuracy'] == pytest.approx(0.7)
    assert m['BalAcc'] == pytest.approx((0.75 + 4 / 6) / 2)
    assert set(m) == {
        'Accuracy',
        'BalAcc',
        'Sensitivity',
        'Specificity',
        'GMean',
        'Kappa',
        'MCC',
        'F1',
        'AUROC',
    }


def test_puntuacion_usa_probabilidad_o_funcion_de_decision(datos):
    X, y = datos
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    assert np.all((f2.puntuacion(rf, X) >= 0) & (f2.puntuacion(rf, X) <= 1))
    svm = SVC().fit(X, y)  # sin probability=True: solo decision_function
    np.testing.assert_allclose(f2.puntuacion(svm, X), svm.decision_function(X))


def test_mlp_balanceada_compensa_el_desbalance(datos):
    X, y = datos
    mlp = f2.MLPBalanceada(hidden_layer_sizes=(8,), max_iter=300, random_state=0).fit(X, y)
    assert set(np.unique(mlp.predict(X))) == {0, 1}  # no se limita a predecir la clase mayoritaria


def test_los_siete_modelos_ponderan_las_clases():
    modelos = f2.modelos()
    assert len(modelos) == 7
    for nombre in ('LogReg (L2)', 'SVM (RBF)', 'Random Forest', 'Extra Trees'):
        assert modelos[nombre].get_params()['class_weight'] in {'balanced', 'balanced_subsample'}
    assert modelos['k-NN (Jaccard)'].get_params()['metric'] == 'jaccard'


@pytest.mark.parametrize(
    'nombre', ['LogReg (L2)', 'SVM (RBF)', 'Random Forest', 'Red neuronal (MLP)']
)
def test_espacio_reconstruye_el_modelo_ganador(nombre):
    params = RESULTADOS['CDS29_holdout']['resultados'][nombre]['best_params']
    modelo = f2.espacio(nombre, optuna.trial.FixedTrial(params))
    ajustes = modelo.get_params()
    for clave, valor in params.items():
        if clave in ajustes:
            assert ajustes[clave] == valor
    if nombre == 'Red neuronal (MLP)':
        assert ajustes['hidden_layer_sizes'] == (params['neuronas'],) * params['capas_ocultas']


def test_espacio_rechaza_modelos_desconocidos():
    with pytest.raises(ValueError, match='k-NN'):
        f2.espacio('k-NN (Jaccard)', optuna.trial.FixedTrial({}))


def test_particion_holdout_estratificada(datos):
    X, y = datos
    X_tr, X_te, y_tr, y_te = f2.particion_holdout(X, y)
    assert len(y_te) == 32
    assert len(y_tr) + len(y_te) == len(y)
    assert abs(y_te.mean() - y.mean()) < 0.03
    # determinista: la misma semilla da la misma partición en todos los scripts
    np.testing.assert_array_equal(f2.particion_holdout(X, y)[3], y_te)


def test_validacion_cruzada_resume_cada_modelo(datos, monkeypatch, capsys):
    X, y = datos
    rapidos = {
        'LogReg (L2)': LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000),
        'Random Forest': RandomForestClassifier(
            n_estimators=20, class_weight='balanced', random_state=0
        ),
    }
    monkeypatch.setattr(f2, 'modelos', lambda: rapidos)
    res = f2.validacion_cruzada(X, y, n_splits=3, n_repeats=2)
    assert set(res) == set(rapidos)
    for resumen in res.values():
        assert 0.5 < resumen['media']['GMean'] <= 1
        assert resumen['std']['GMean'] >= 0
    assert 'GMean=' in capsys.readouterr().out


def test_optimizar_maximiza_el_gmean(datos, monkeypatch):
    X, y = datos
    monkeypatch.setattr(f2, 'N_TRIALS', 5)
    folds = list(f2.StratifiedKFold(n_splits=3, shuffle=True, random_state=f2.SEED).split(X, y))
    estudio = f2.optimizar('LogReg (L2)', X, y, folds)
    assert len(estudio.trials) == 5
    assert estudio.best_value == max(t.value for t in estudio.trials if t.value is not None)
    assert 1e-3 <= estudio.best_params['C'] <= 10


def test_estilo_aplica_la_plantilla():
    import plotly.graph_objects as go

    fig = f2.estilo(go.Figure())
    assert fig.layout.paper_bgcolor == 'white'
    assert fig.layout.legend.orientation == 'h'


# ---------------------------------------------------------------------------
# Con las bases de datos de la asignatura (restauradas del dump)
# ---------------------------------------------------------------------------
@pytest.mark.mongo
@pytest.mark.parametrize('base', ['CDS16', 'CDS29'])
def test_cargar_dataset_reconstruye_los_fingerprints(mongo_client, bases_moleculares, base):
    df, n_antes = f2.cargar_dataset(mongo_client, bases_moleculares[base])
    info = RESULTADOS[base]['info']
    X = df.drop(columns='class').to_numpy()
    assert n_antes == info['bits_total'] == 1024
    assert X.shape == (info['n_molecules'], info['bits_utiles'])
    assert int(df['class'].sum()) == info['n_active']
    assert X.sum(axis=1).mean() == pytest.approx(info['bits_por_mol_media'])
    assert X.sum(axis=0).min() > 0  # las columnas todo '0' se eliminan
    # cada fila coincide con las posiciones guardadas en mfp.bits
    doc = mongo_client[bases_moleculares[base]].molecules.find_one()
    fila = df.loc[doc['_id']]
    assert sorted(
        int(c[2:]) - 1 for c in fila.index[(fila == 1).to_numpy()] if c != 'class'
    ) == sorted(doc['mfp']['bits'])


@pytest.mark.mongo
def test_el_random_forest_de_optuna_reproduce_el_test(mongo_client, bases_moleculares):
    df, _ = f2.cargar_dataset(mongo_client, bases_moleculares['CDS29'])
    X = df.drop(columns='class').to_numpy()
    y = df['class'].to_numpy()
    X_tr, X_te, y_tr, y_te = f2.particion_holdout(X, y)
    holdout = RESULTADOS['CDS29_holdout']
    assert (len(y_tr), len(y_te), int(y_te.sum())) == (
        holdout['n_train'],
        holdout['n_test'],
        holdout['test_active'],
    )

    rf = holdout['resultados']['Random Forest']
    modelo = f2.espacio('Random Forest', optuna.trial.FixedTrial(rf['best_params'])).fit(X_tr, y_tr)
    m = f2.metricas(y_te, modelo.predict(X_te).astype(int), f2.puntuacion(modelo, X_te))
    assert m['GMean'] == pytest.approx(rf['test']['GMean'])
    assert m['AUROC'] == pytest.approx(rf['test']['AUROC'])
