#!/usr/bin/env python3
"""
ADP - Practicas (Apartado I) - Fase 2 (ampliacion)
Validacion cruzada de combinaciones de inyeccion de ruido y eliminacion de outliers en CDS29,
con la frontera de decision de cada combinacion dibujada sobre las dos primeras componentes
principales del fingerprint.

No repite el ajuste con Optuna: el Random Forest usa los hiperparametros guardados por
fase2_clasificacion.py en fase2_resultados.json, y se trabaja sobre la misma particion de
entrenamiento (80 %), de modo que el test sigue sin tocarse hasta la evaluacion final.

Autor: Pablo Caballero Maria
Uso: uv run python fase2_ruido_outliers.py   (desde src/, despues de fase2_clasificacion.py)
"""

import json
from time import time

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import pymongo
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor

from fase2_clasificacion import (
    CONNECTION,
    EJE,
    SEED,
    TINTA,
    TINTA_2,
    cargar_dataset,
    espacio,
    estilo,
    gmean,
    metricas,
    particion_holdout,
)

MODELO = 'Random Forest'
N_FOLDS = 5
FOLD_FIGURA = 0  # fold cuya frontera y cuyas moleculas de validacion se dibujan
CONTAMINACION = 0.05  # fraccion de cada clase que se marca como outlier
SIGMA = 0.15  # ruido gaussiano
P_VOLTEO = 0.005  # probabilidad de invertir cada bit (~5 bits por molecula)
P_MASCARA = 0.10  # probabilidad de apagar cada bit encendido (~4-5 bits por molecula)
RESOLUCION = 150  # celdas por eje de la rejilla del fondo

INACTIVA, ACTIVA = '#2a78d6', '#e34948'  # polos del par divergente azul <-> rojo
FONDO = [[0.0, '#9ec5f4'], [0.5, '#f0efec'], [1.0, '#f1a4a3']]  # mismos polos, aclarados


# ---------------------------------------------------------------------------
# Inyeccion de ruido: cada metodo genera una copia ruidosa del entrenamiento que se anade
# a los datos originales (aumento de datos), con las mismas etiquetas.
# ---------------------------------------------------------------------------
def ruido_gaussiano(X, rng):
    return X + rng.normal(0.0, SIGMA, X.shape).astype(np.float32)


def ruido_volteo(X, rng):
    """Ruido de sal y pimienta: cada bit se invierte con probabilidad P_VOLTEO."""
    return np.where(rng.random(X.shape) < P_VOLTEO, 1 - X, X)


def ruido_enmascarado(X, rng):
    """Ruido de enmascaramiento: se apagan bits encendidos (subestructuras no detectadas)."""
    return X * (rng.random(X.shape) >= P_MASCARA)


RUIDOS = {
    'Sin ruido': None,
    'Gaussiano (σ = %.2f)' % SIGMA: ruido_gaussiano,
    'Volteo de bits (p = %.3f)' % P_VOLTEO: ruido_volteo,
    'Enmascarado (p = %.2f)' % P_MASCARA: ruido_enmascarado,
}


# ---------------------------------------------------------------------------
# Eliminacion de outliers: cada detector devuelve la mascara de outliers de una clase.
# Se aplica a cada clase por separado para no vaciar la clase minoritaria (activas).
# ---------------------------------------------------------------------------
def distancias_jaccard(X):
    """Matriz de distancias de Jaccard (1 - Tanimoto) entre fingerprints binarios."""
    Z = X.astype(np.float32)
    inter = Z @ Z.T
    n = Z.sum(axis=1)
    return 1.0 - inter / np.maximum(n[:, None] + n[None, :] - inter, 1.0)


def outliers_iforest(X):
    detector = IsolationForest(
        n_estimators=200, contamination=CONTAMINACION, random_state=SEED, n_jobs=-1
    )
    return detector.fit_predict(X) == -1


def outliers_lof(X):
    detector = LocalOutlierFactor(n_neighbors=20, metric='precomputed', contamination=CONTAMINACION)
    return detector.fit_predict(distancias_jaccard(X)) == -1


OUTLIERS = {
    'Sin eliminar outliers': None,
    'Isolation Forest': outliers_iforest,
    'LOF (Jaccard)': outliers_lof,
}


def conservadas(detector, X, y):
    """Mascara de las moleculas de entrenamiento que sobreviven al detector."""
    mascara = np.ones(len(y), dtype=bool)
    if detector is not None:
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            mascara[idx[detector(X[idx])]] = False
    return mascara


def aumentar(ruido, X, y, rng):
    if ruido is None:
        return X, y
    return np.vstack([X, ruido(X, rng)]), np.concatenate([y, y])


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
def figura_fronteras(fig_datos, resumen, var, rejilla):
    filas, columnas = list(RUIDOS), list(OUTLIERS)
    gx, gy = rejilla
    fig = make_subplots(
        rows=len(filas),
        cols=len(columnas),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        vertical_spacing=0.022,
        column_titles=columnas,
        row_titles=filas,
    )
    for i, ruido in enumerate(filas, start=1):
        for j, det in enumerate(columnas, start=1):
            d = fig_datos[(ruido, det)]
            fig.add_trace(
                go.Heatmap(
                    x=gx,
                    y=gy,
                    z=d['fondo'],
                    coloraxis='coloraxis',
                    zsmooth='best',
                    hovertemplate='CP1 = %{x:.2f} · CP2 = %{y:.2f}<br>'
                    'P(activa) = %{z:.2f}<extra></extra>',
                ),
                row=i,
                col=j,
            )
            fig.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=d['fondo'],
                    contours=dict(start=0.5, end=0.5, size=1, coloring='lines'),
                    colorscale=[[0, TINTA_2], [1, TINTA_2]],
                    line=dict(width=1),
                    showscale=False,
                    hoverinfo='skip',
                ),
                row=i,
                col=j,
            )
            for clase, color, nombre in [(0, INACTIVA, 'Inactiva'), (1, ACTIVA, 'Activa')]:
                sel = d['y'] == clase
                fig.add_trace(
                    go.Scatter(
                        x=d['Z'][sel, 0],
                        y=d['Z'][sel, 1],
                        mode='markers',
                        marker=dict(color=color, size=4, line=dict(color='#fcfcfb', width=0.5)),
                        name='%s (clase real)' % nombre,
                        legendgroup=nombre,
                        showlegend=(i == 1 and j == 1),
                        customdata=np.column_stack([d['ids'][sel], d['p2'][sel], d['p1024'][sel]]),
                        hovertemplate='%{customdata[0]} · ' + nombre + '<br>'
                        'P(activa), modelo 2 CP = %{customdata[1]:.2f}<br>'
                        'P(activa), modelo 1024 bits = %{customdata[2]:.2f}<extra></extra>',
                    ),
                    row=i,
                    col=j,
                )
            r = resumen[(ruido, det)]
            n = (i - 1) * len(columnas) + j
            eje = '' if n == 1 else str(n)
            fig.add_annotation(
                text='GMean 1024 bits %.3f ± %.3f<br>GMean 2 CP %.3f'
                % (r['gmean'], r['gmean_std'], r['gmean_2cp']),
                xref='x%s domain' % eje,
                yref='y%s domain' % eje,
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                align='left',
                showarrow=False,
                font=dict(size=10, color=TINTA),
                bgcolor='rgba(252,252,251,0.82)',
                borderpad=3,
            )
    estilo(fig)
    fig.update_xaxes(range=[gx[0], gx[-1]], showgrid=False, linecolor=EJE, mirror=True)
    fig.update_yaxes(range=[gy[0], gy[-1]], showgrid=False, linecolor=EJE, mirror=True)
    for j in range(1, len(columnas) + 1):
        fig.update_xaxes(title_text='CP1 (%.1f %% var.)' % (100 * var[0]), row=len(filas), col=j)
    for i in range(1, len(filas) + 1):
        fig.update_yaxes(title_text='CP2 (%.1f %% var.)' % (100 * var[1]), row=i, col=1)
    for t in columnas + filas:
        fig.update_annotations(selector=dict(text=t), font=dict(size=13, color=TINTA))
    fig.update_layout(
        coloraxis=dict(
            colorscale=FONDO,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=dict(text='P(activa)', side='top'),
                orientation='h',
                x=1.0,
                xanchor='right',
                y=1.035,
                yanchor='bottom',
                len=0.3,
                thickness=10,
                tickvals=[0, 0.5, 1],
                outlinewidth=0,
            ),
        ),
        legend=dict(x=0, y=1.035, itemsizing='constant'),
        margin=dict(l=10, r=10, t=95, b=10),
    )
    fig.write_html('fig_ruido_outliers.html', include_plotlyjs='cdn')
    fig.write_image('fig_ruido_outliers.png', width=900, height=1120, scale=2)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    with open('fase2_resultados.json') as fh:
        params = json.load(fh)['CDS29_holdout']['resultados'][MODELO]['best_params']

    def modelo():
        return espacio(MODELO, optuna.trial.FixedTrial(params))

    client = pymongo.MongoClient(CONNECTION)
    df, _ = cargar_dataset(client, 'CDS29')
    client.close()
    X = df.drop(columns='class').to_numpy()
    y = df['class'].to_numpy()
    ids = df.index.astype(str).to_numpy()
    i_tr, i_te, y_tr, y_te = particion_holdout(np.arange(len(y)), y)
    X_tr, X_te = X[i_tr], X[i_te]
    print(
        '### CDS29: %s con los hiperparametros de Optuna %s\n'
        'Validacion cruzada estratificada de %d folds sobre la particion de entrenamiento '
        '(%d moleculas, %d activas)' % (MODELO, params, N_FOLDS, len(y_tr), int(y_tr.sum()))
    )
    print(
        'Ruido (copia ruidosa anadida al entrenamiento): %s\n'
        'Outliers (%.0f %% de cada clase, detectados antes de anadir ruido): %s\n'
        % (', '.join(RUIDOS), 100 * CONTAMINACION, ', '.join(OUTLIERS))
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    filas, filas_2cp, eliminadas, tiempos = {}, {}, {}, {}
    fig_datos, var, rejilla, malla = {}, None, None, np.empty((0, 2))
    for k, (tr, va) in enumerate(cv.split(X_tr, y_tr)):
        t_fold = time()
        Xa, ya, Xv, yv = X_tr[tr], y_tr[tr], X_tr[va], y_tr[va]
        # Proyeccion comun a todas las combinaciones del fold: PCA del entrenamiento original
        pca = PCA(n_components=2, random_state=SEED).fit(Xa)
        Zv = pca.transform(Xv)
        if k == FOLD_FIGURA:
            var = pca.explained_variance_ratio_
            margen = 0.06 * (Zv.max(axis=0) - Zv.min(axis=0))
            lo, hi = Zv.min(axis=0) - margen, Zv.max(axis=0) + margen
            rejilla = (
                np.linspace(lo[0], hi[0], RESOLUCION),
                np.linspace(lo[1], hi[1], RESOLUCION),
            )
            malla = np.column_stack([m.ravel() for m in np.meshgrid(*rejilla)])
        for det_nombre, detector in OUTLIERS.items():
            keep = conservadas(detector, Xa, ya)
            for r_idx, (ruido_nombre, ruido) in enumerate(RUIDOS.items()):
                clave = (ruido_nombre, det_nombre)
                t0 = time()
                rng = np.random.default_rng([SEED, k, r_idx])
                Xp, yp = aumentar(ruido, Xa[keep], ya[keep], rng)
                m = modelo().fit(Xp, yp)
                p1024 = m.predict_proba(Xv)[:, 1]
                filas.setdefault(clave, []).append(metricas(yv, m.predict(Xv).astype(int), p1024))
                # Mismo modelo y mismos datos, pero reentrenado sobre las 2 CP: su frontera
                # es exactamente la que se dibuja, no una aproximacion
                m2 = modelo().fit(pca.transform(Xp), yp)
                filas_2cp.setdefault(clave, []).append(gmean(yv, m2.predict(Zv).astype(int)))
                eliminadas.setdefault(clave, []).append(int((~keep).sum()))
                tiempos[clave] = tiempos.get(clave, 0.0) + time() - t0
                if k == FOLD_FIGURA:
                    fig_datos[clave] = {
                        'fondo': m2.predict_proba(malla)[:, 1].reshape(RESOLUCION, RESOLUCION),
                        'Z': Zv,
                        'y': yv,
                        'ids': ids[i_tr[va]],
                        'p2': m2.predict_proba(Zv)[:, 1],
                        'p1024': p1024,
                    }
        print('  fold %d/%d completado (%.0f s)' % (k + 1, N_FOLDS, time() - t_fold))

    # --- Resumen de la validacion cruzada --------------------------------------------------
    resumen, combinaciones = {}, {}
    print(
        '\n%-28s %-22s %-15s %-7s %-7s %-7s %-9s %-6s'
        % ('Ruido', 'Outliers', 'GMean', 'Kappa', 'MCC', 'AUROC', 'GMean2CP', 'Elim.')
    )
    for clave, f in filas.items():
        d = pd.DataFrame(f)
        resumen[clave] = {
            'gmean': d.GMean.mean(),
            'gmean_std': d.GMean.std(),
            'gmean_2cp': float(np.mean(filas_2cp[clave])),
        }
        combinaciones[' + '.join(clave)] = {
            'ruido': clave[0],
            'outliers': clave[1],
            'media': d.mean().to_dict(),
            'std': d.std().to_dict(),
            'gmean_2cp_media': float(np.mean(filas_2cp[clave])),
            'gmean_2cp_std': float(np.std(filas_2cp[clave], ddof=1)),
            'eliminadas_por_fold': float(np.mean(eliminadas[clave])),
            'tiempo_s': tiempos[clave],
        }
        print(
            '%-28s %-22s %.3f ± %.3f  %.3f   %.3f   %.3f   %.3f     %.0f'
            % (
                clave[0],
                clave[1],
                d.GMean.mean(),
                d.GMean.std(),
                d.Kappa.mean(),
                d.MCC.mean(),
                d.AUROC.mean(),
                np.mean(filas_2cp[clave]),
                np.mean(eliminadas[clave]),
            )
        )

    # --- Test: combinacion base frente a la mejor de la validacion cruzada ---------------
    base = (list(RUIDOS)[0], list(OUTLIERS)[0])
    mejor = max(resumen, key=lambda c: resumen[c]['gmean'])
    print('\nMejor combinacion en CV (GMean): %s' % ' + '.join(mejor))
    test = {}
    for etiqueta, (ruido_nombre, det_nombre) in [('base', base), ('mejor', mejor)]:
        if etiqueta == 'mejor' and mejor == base:
            test['mejor'] = test['base']
            break
        keep = conservadas(OUTLIERS[det_nombre], X_tr, y_tr)
        rng = np.random.default_rng([SEED, N_FOLDS, list(RUIDOS).index(ruido_nombre)])
        Xp, yp = aumentar(RUIDOS[ruido_nombre], X_tr[keep], y_tr[keep], rng)
        m = modelo().fit(Xp, yp)
        met = metricas(y_te, m.predict(X_te).astype(int), m.predict_proba(X_te)[:, 1])
        test[etiqueta] = {'combinacion': ' + '.join((ruido_nombre, det_nombre)), 'test': met}
        print(
            '  test %-5s %-50s GMean=%.3f Kappa=%.3f Acc=%.3f AUROC=%.3f'
            % (
                etiqueta,
                test[etiqueta]['combinacion'],
                met['GMean'],
                met['Kappa'],
                met['Accuracy'],
                met['AUROC'],
            )
        )

    assert var is not None and rejilla is not None
    figura_fronteras(fig_datos, resumen, var, rejilla)
    salida = {
        'modelo': MODELO,
        'best_params': params,
        'folds': N_FOLDS,
        'n_train': int(len(y_tr)),
        'n_test': int(len(y_te)),
        'contaminacion': CONTAMINACION,
        'sigma_gaussiano': SIGMA,
        'p_volteo': P_VOLTEO,
        'p_mascara': P_MASCARA,
        'fold_figura': FOLD_FIGURA + 1,
        'pca_varianza_explicada_fold_figura': [float(v) for v in var],
        'combinaciones': combinaciones,
        'test': test,
    }
    with open('ruido_outliers_resultados.json', 'w') as fh:
        json.dump(salida, fh, indent=1, default=float, ensure_ascii=False)
    print(
        '\nResultados guardados en ruido_outliers_resultados.json y figura '
        'fig_ruido_outliers.png / .html'
    )


if __name__ == '__main__':
    main()
