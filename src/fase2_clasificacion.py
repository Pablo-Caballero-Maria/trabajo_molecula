#!/usr/bin/env python3
"""
ADP - Practicas (Apartado I) - Fase 2
Prediccion (clasificacion) de la actividad biologica (Active=1 / Inactive=0)
a partir del fingerprint molecular almacenado en MongoDB.

Autor: Pablo Caballero Maria
Uso: uv run python fase2_clasificacion.py   (desde src/)
     La variable de entorno MONGO_URI permite usar otro servidor (por defecto, el local).
"""

import json
import os
import warnings
from time import time

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import pymongo
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from documentos import moleculas

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
CONNECTION = os.environ.get('MONGO_URI', 'mongodb://adp:adp@127.0.0.1:27017/?authSource=admin')
N_BITS = 1024
N_TRIALS = 40  # pruebas de Optuna por modelo


class MLPBalanceada(MLPClassifier):
    """Perceptron multicapa con pesos de muestra inversos a la frecuencia de cada clase.

    MLPClassifier no admite class_weight; con compute_sample_weight('balanced') se obtiene
    la misma ponderacion que usan el resto de modelos.
    """

    def fit(self, X, y):
        return super().fit(X, y, sample_weight=compute_sample_weight('balanced', y))


def cargar_dataset(client, db_name):
    """Matriz binaria n x p de fingerprints; se eliminan las columnas todo '0'."""
    cur = moleculas(client[db_name]).find({}, {'_id': 1, 'class': 1, 'mfp.bits': 1})
    ids, filas, y = [], [], []
    for doc in cur:
        ids.append(doc['_id'])
        filas.append(doc['mfp']['bits'])
        y.append(1 if doc['class'] == 'Active' else 0)
    X = np.zeros((len(filas), N_BITS), dtype=np.uint8)
    for i, bits in enumerate(filas):  # en la BD solo estan las posiciones con valor 1
        X[i, bits] = 1
    df = pd.DataFrame(X, columns=['FP%d' % (j + 1) for j in range(N_BITS)], index=ids)
    n_antes = df.shape[1]
    df = df.loc[:, df.sum(axis=0) > 0]
    df['class'] = y
    return df, n_antes


def particion_holdout(X, y):
    """Particion 80/20 estratificada de CDS29 (X_tr, X_te, y_tr, y_te).

    Los scripts de ruido/outliers y de SHAP la reutilizan para trabajar con las mismas
    moleculas de entrenamiento y de test que el ajuste con Optuna.
    """
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def gmean(y_true, y_pred):
    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    return float(np.sqrt(sens * spec))


def metricas(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'BalAcc': balanced_accuracy_score(y_true, y_pred),
        'Sensitivity': tp / (tp + fn),
        'Specificity': tn / (tn + fp),
        'GMean': gmean(y_true, y_pred),
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, score),
    }


def puntuacion(modelo, X):
    if hasattr(modelo, 'predict_proba'):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def modelos():
    return {
        'LogReg (L2)': LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000),
        'SVM (RBF)': SVC(
            C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', random_state=SEED
        ),
        'k-NN (Jaccard)': KNeighborsClassifier(
            n_neighbors=5, metric='jaccard', weights='distance', n_jobs=-1
        ),
        'Bernoulli NB': BernoulliNB(alpha=1.0),
        'Random Forest': RandomForestClassifier(
            n_estimators=500, n_jobs=-1, class_weight='balanced_subsample', random_state=SEED
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=500, n_jobs=-1, class_weight='balanced_subsample', random_state=SEED
        ),
        'Red neuronal (MLP)': MLPBalanceada(
            hidden_layer_sizes=(100,), alpha=1.0, max_iter=500, random_state=SEED
        ),
    }


def validacion_cruzada(X, y, n_splits, n_repeats):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    res = {}
    for nombre, base in modelos().items():
        t0 = time()
        filas = []
        for tr, te in cv.split(X, y):
            m = clone(base).fit(X[tr], y[tr])
            filas.append(metricas(y[te], m.predict(X[te]).astype(int), puntuacion(m, X[te])))
        d = pd.DataFrame(filas)
        res[nombre] = {
            'media': d.mean().to_dict(),
            'std': d.std().to_dict(),
            'tiempo_s': time() - t0,
        }
        print(
            '  %-18s GMean=%.3f+-%.3f  Kappa=%.3f  Acc=%.3f  AUROC=%.3f  (%.1f s)'
            % (
                nombre,
                d.GMean.mean(),
                d.GMean.std(),
                d.Kappa.mean(),
                d.Accuracy.mean(),
                d.AUROC.mean(),
                time() - t0,
            )
        )
    return res


# ---------------------------------------------------------------------------
# Espacios de busqueda de Optuna. Con optuna.trial.FixedTrial(best_params) la misma
# funcion reconstruye el modelo ganador para entrenarlo sobre todo el entrenamiento.
# ---------------------------------------------------------------------------
def espacio(nombre, trial):
    if nombre == 'LogReg (L2)':
        return LogisticRegression(
            C=trial.suggest_float('C', 1e-3, 10, log=True), class_weight='balanced', max_iter=5000
        )
    if nombre == 'SVM (RBF)':
        return SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=SEED,
            C=trial.suggest_float('C', 0.1, 30, log=True),
            gamma=trial.suggest_float('gamma', 1e-3, 1e-1, log=True),
        )
    if nombre == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
            class_weight='balanced_subsample',
            random_state=SEED,
            max_features=trial.suggest_float('max_features', 0.01, 0.5, log=True),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 50, log=True),
        )
    if nombre == 'Red neuronal (MLP)':
        capas = trial.suggest_int('capas_ocultas', 1, 2)
        neuronas = trial.suggest_int('neuronas', 16, 256, log=True)
        return MLPBalanceada(
            hidden_layer_sizes=(neuronas,) * capas,
            max_iter=500,
            random_state=SEED,
            alpha=trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            learning_rate_init=trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        )
    raise ValueError(nombre)


def optimizar(nombre, X, y, folds):
    """Estudio de Optuna (TPE + poda por mediana) que maximiza el GMean medio de la CV interna."""

    def objetivo(trial):
        modelo = espacio(nombre, trial)
        valores = []
        for paso, (tr, va) in enumerate(folds):
            m = clone(modelo).fit(X[tr], y[tr])
            valores.append(gmean(y[va], m.predict(X[va]).astype(int)))
            trial.report(float(np.mean(valores)), paso)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(valores))

    estudio = optuna.create_study(
        direction='maximize',
        study_name=nombre,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    estudio.optimize(objetivo, n_trials=N_TRIALS)
    return estudio


# ---------------------------------------------------------------------------
# Figuras (Plotly): PNG para la memoria y HTML interactivo
# ---------------------------------------------------------------------------
SERIE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # orden categorico fijo
TINTA, TINTA_2, REJILLA, EJE = '#0b0b0b', '#52514e', '#e6e5e1', '#b9b8b3'


def estilo(fig):
    fig.update_layout(
        template='simple_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial', size=13, color=TINTA_2),
        legend=dict(
            orientation='h', x=0, xanchor='left', y=1.02, yanchor='bottom', font=dict(color=TINTA)
        ),
        margin=dict(l=10, r=15, t=50, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=REJILLA, linecolor=EJE, ticks='', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=REJILLA, linecolor=EJE, ticks='', zeroline=False)
    return fig


def guardar(fig, nombre, ancho, alto):
    fig.write_html(nombre + '.html', include_plotlyjs='cdn')
    fig.write_image(nombre + '.png', width=ancho, height=alto, scale=3)


def figura_cv(salida):
    nombres = list(modelos())
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=('GMean', 'AUROC'),
    )
    for k, (db, desplaz) in enumerate([('CDS16', -0.17), ('CDS29', 0.17)]):
        res = salida[db]['resultados']
        for col, met in [(1, 'GMean'), (2, 'AUROC')]:
            fig.add_trace(
                go.Scatter(
                    x=[res[n]['media'][met] for n in nombres],
                    y=[i + desplaz for i in range(len(nombres))],
                    error_x=dict(
                        type='data',
                        array=[res[n]['std'][met] for n in nombres],
                        color=SERIE[k],
                        thickness=1.5,
                        width=0,
                    ),
                    mode='markers',
                    marker=dict(color=SERIE[k], size=9, line=dict(color='white', width=1)),
                    name=db,
                    legendgroup=db,
                    showlegend=(col == 1),
                    customdata=[[n, res[n]['std'][met]] for n in nombres],
                    hovertemplate='%{customdata[0]} · '
                    + db
                    + '<br>'
                    + met
                    + ' = %{x:.3f} ± %{customdata[1]:.3f}<extra></extra>',
                ),
                row=1,
                col=col,
            )
    fig.update_yaxes(
        tickvals=list(range(len(nombres))), ticktext=nombres, autorange='reversed', showgrid=False
    )
    fig.update_xaxes(range=[0.25, 1.02], dtick=0.1)
    estilo(fig)
    fig.update_annotations(font=dict(size=13, color=TINTA))
    fig.update_layout(legend=dict(y=1.13), margin=dict(t=62))
    guardar(fig, 'fig_cv_modelos', 640, 300)


def figura_optuna(estudios):
    fig = go.Figure()
    for k, (nombre, est) in enumerate(estudios.items()):
        hechos = [t for t in est.trials if t.state == optuna.trial.TrialState.COMPLETE]
        fig.add_trace(
            go.Scatter(
                x=[t.number + 1 for t in hechos],
                y=[t.value for t in hechos],
                mode='markers',
                marker=dict(color=SERIE[k], size=6, opacity=0.35),
                legendgroup=nombre,
                showlegend=False,
                hovertemplate=nombre + ' · prueba %{x}<br>GMean = %{y:.3f}<extra></extra>',
            )
        )
        mejor, xs, ys = -np.inf, [], []
        for t in est.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                mejor = max(mejor, t.value)
            if np.isfinite(mejor):
                xs.append(t.number + 1)
                ys.append(mejor)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                line=dict(color=SERIE[k], width=2, shape='hv'),
                name=nombre,
                legendgroup=nombre,
                hovertemplate=nombre + ' · prueba %{x}<br>mejor GMean = %{y:.3f}<extra></extra>',
            )
        )
    fig.update_xaxes(title_text='Prueba de Optuna', range=[0.5, N_TRIALS + 0.5])
    fig.update_yaxes(title_text='GMean (CV interna)')
    estilo(fig)
    guardar(fig, 'fig_optuna_cds29', 640, 290)


def figura_roc(curvas):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            showlegend=False,
            line=dict(color=EJE, width=1),
            hoverinfo='skip',
        )
    )
    for k, (nombre, (fpr, tpr, auc)) in enumerate(curvas.items()):
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                line=dict(color=SERIE[k], width=2),
                name='%s (AUC = %.3f)' % (nombre, auc),
                hovertemplate=nombre + '<br>FPR = %{x:.3f}<br>TPR = %{y:.3f}<extra></extra>',
            )
        )
    fig.update_xaxes(
        title_text='1 - Especificidad (FPR)', range=[0, 1], dtick=0.2, constrain='domain'
    )
    fig.update_yaxes(
        title_text='Sensibilidad (TPR)', range=[0, 1], dtick=0.2, scaleanchor='x', scaleratio=1
    )
    estilo(fig)
    fig.update_layout(
        legend=dict(
            orientation='v',
            x=0.98,
            xanchor='right',
            y=0.03,
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.85)',
        ),
        margin=dict(t=10),
    )
    guardar(fig, 'fig_roc_cds29', 470, 420)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    client = pymongo.MongoClient(CONNECTION)
    salida = {}
    config_cv = {'CDS16': (10, 5), 'CDS29': (5, 2)}  # (folds, repeticiones)

    for db_name in ['CDS16', 'CDS29']:
        df, n_antes = cargar_dataset(client, db_name)
        X = df.drop(columns='class').values
        y = df['class'].values
        bpm = X.sum(axis=1)
        info = {
            'n_molecules': int(len(y)),
            'n_active': int(y.sum()),
            'n_inactive': int((y == 0).sum()),
            'bits_total': n_antes,
            'bits_utiles': int(X.shape[1]),
            'bits_eliminados': int(n_antes - X.shape[1]),
            'bits_por_mol_media': float(bpm.mean()),
            'bits_por_mol_min': int(bpm.min()),
            'bits_por_mol_max': int(bpm.max()),
            'densidad': float(X.mean()),
        }
        print('\n### %s: %s' % (db_name, info))
        print(df.iloc[:3, list(range(6)) + [-1]].to_string())
        k, r = config_cv[db_name]
        print('Validacion cruzada estratificada %d folds x %d repeticiones' % (k, r))
        salida[db_name] = {
            'info': info,
            'cv': {'folds': k, 'repeats': r},
            'resultados': validacion_cruzada(X, y, k, r),
        }

    figura_cv(salida)

    # --- CDS29: ajuste de hiperparametros con Optuna y evaluacion sobre particion de test ---
    print('\n### CDS29: Optuna (TPE, %d pruebas, metrica GMean) y test hold-out 80/20' % N_TRIALS)
    df, _ = cargar_dataset(client, 'CDS29')
    X = df.drop(columns='class').values
    y = df['class'].values
    X_tr, X_te, y_tr, y_te = particion_holdout(X, y)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X_tr, y_tr))

    test, estudios, curvas, ajustados = {}, {}, {}, {}
    for nombre in ['LogReg (L2)', 'SVM (RBF)', 'Random Forest', 'Red neuronal (MLP)']:
        t0 = time()
        est = optimizar(nombre, X_tr, y_tr, folds)
        t_opt = time() - t0
        estudios[nombre] = est
        best = espacio(nombre, optuna.trial.FixedTrial(est.best_params)).fit(X_tr, y_tr)
        ajustados[nombre] = best
        sc = puntuacion(best, X_te)
        pred = best.predict(X_te).astype(int)
        m = metricas(y_te, pred, sc)
        cm = confusion_matrix(y_te, pred, labels=[1, 0])
        estados = [t.state for t in est.trials]
        test[nombre] = {
            'best_params': est.best_params,
            'gmean_cv_interno': est.best_value,
            'mejor_prueba': est.best_trial.number + 1,
            'pruebas_completas': estados.count(optuna.trial.TrialState.COMPLETE),
            'pruebas_podadas': estados.count(optuna.trial.TrialState.PRUNED),
            'tiempo_optuna_s': t_opt,
            'test': m,
            'confusion': cm.tolist(),
            'historial': [t.value for t in est.trials],
        }
        fpr, tpr, _ = roc_curve(y_te, sc)
        curvas[nombre] = (fpr, tpr, m['AUROC'])
        params = {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in est.best_params.items()
        }
        print(
            '  %-18s best=%s  CV-GMean=%.3f (prueba %d; %d podadas; %.0f s)\n'
            '  %-18s test: GMean=%.3f Kappa=%.3f Acc=%.3f AUROC=%.3f  CM=%s'
            % (
                nombre,
                params,
                est.best_value,
                est.best_trial.number + 1,
                test[nombre]['pruebas_podadas'],
                t_opt,
                '',
                m['GMean'],
                m['Kappa'],
                m['Accuracy'],
                m['AUROC'],
                cm.tolist(),
            )
        )

    figura_optuna(estudios)
    figura_roc(curvas)

    salida['CDS29_holdout'] = {
        'n_train': int(len(y_tr)),
        'n_test': int(len(y_te)),
        'test_active': int(y_te.sum()),
        'optuna_trials': N_TRIALS,
        'resultados': test,
    }

    cols = df.drop(columns='class').columns
    imp = (
        pd.Series(ajustados['Random Forest'].feature_importances_, index=cols)
        .sort_values(ascending=False)
        .head(10)
    )
    salida['CDS29_top_bits'] = {k: float(v) for k, v in imp.items()}
    print('  Bits mas relevantes (RF):', list(imp.index))

    with open('fase2_resultados.json', 'w') as f:
        json.dump(salida, f, indent=1, default=float)
    print('\nResultados guardados en fase2_resultados.json y figuras fig_*.png / fig_*.html')
    client.close()


if __name__ == '__main__':
    main()
