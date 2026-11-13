#!/usr/bin/env python3
"""
ADP - Practicas (Apartado I) - Fase 2 (explicabilidad)
Valores SHAP del Random Forest ajustado con Optuna sobre CDS29 y grafico de enjambre
(beeswarm) de los bits del fingerprint que mas influyen en la prediccion de actividad.

No repite el ajuste con Optuna: los hiperparametros se leen de fase2_resultados.json y el
modelo se reentrena una sola vez sobre la misma particion de entrenamiento (con la misma
semilla, de modo que es el mismo bosque que se evaluo en test). Las explicaciones se calculan
sobre las moleculas de test con shap.TreeExplainer, que es exacto para bosques de arboles.

Autor: Pablo Caballero Maria
Uso: uv run python fase2_shap.py   (desde src/, despues de fase2_clasificacion.py)
"""

import json
from time import time

import numpy as np
import optuna
import plotly.graph_objects as go
import pymongo
import shap

from fase2_clasificacion import (
    CONNECTION,
    EJE,
    SEED,
    TINTA,
    cargar_dataset,
    espacio,
    estilo,
    guardar,
    metricas,
    particion_holdout,
)

MODELO = 'Random Forest'
N_TOP = 15  # bits que se muestran en el grafico
BIT_ON, BIT_OFF = '#6250d6', '#b9b8b3'  # bit encendido (subestructura presente) / apagado


def enjambre(valores, n_bins=120, alto=0.4):
    """Desplazamiento vertical de cada punto de una fila del beeswarm.

    Igual que shap.plots.beeswarm: los valores se agrupan en intervalos del eje x y los puntos
    de un mismo intervalo se apilan alternando por encima y por debajo de la fila.
    """
    rng = np.random.default_rng(SEED)
    orden = np.argsort(valores + rng.normal(0, 1e-9, len(valores)))
    ancho = (valores.max() - valores.min()) / n_bins or 1.0
    bins = np.floor((valores[orden] - valores.min()) / ancho).astype(int)
    capa = np.zeros(len(valores))
    anterior, k = None, 0
    for pos, b in enumerate(bins):
        k = k + 1 if b == anterior else 0
        anterior = b
        capa[orden[pos]] = (k + 1) // 2 * (1 if k % 2 else -1)
    return capa * alto / max(np.abs(capa).max(), 1.0)


def figura_beeswarm(sv, X, nombres, ids, top):
    fig = go.Figure()
    desplaz = {j: enjambre(sv[:, j]) for j in top}
    for valor, color, texto in [
        (0, BIT_OFF, 'Bit = 0 (ausente)'),
        (1, BIT_ON, 'Bit = 1 (presente)'),
    ]:
        xs, ys, cd = [], [], []
        for fila, j in enumerate(top):
            dy = desplaz[j]
            sel = X[:, j] == valor
            xs.append(sv[sel, j])
            ys.append(fila + dy[sel])
            cd.append(np.column_stack([ids[sel], np.repeat(nombres[j], sel.sum())]))
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(xs),
                y=np.concatenate(ys),
                mode='markers',
                marker=dict(color=color, size=4, opacity=0.75),
                name=texto,
                customdata=np.vstack(cd),
                hovertemplate='%{customdata[0]} · %{customdata[1]} = '
                + str(valor)
                + '<br>SHAP = %{x:+.4f}<extra></extra>',
            )
        )
    fig.add_vline(x=0, line=dict(color=EJE, width=1))
    fig.update_yaxes(
        tickvals=list(range(len(top))),
        ticktext=[nombres[j] for j in top],
        autorange='reversed',
        showgrid=False,
        tickfont=dict(color=TINTA),
    )
    fig.update_xaxes(title_text='Valor SHAP (contribución a P(activa))', zeroline=False)
    estilo(fig)
    fig.update_layout(legend=dict(itemsizing='constant'))
    guardar(fig, 'fig_shap_cds29', 640, 120 + 26 * len(top))


def main():
    with open('fase2_resultados.json') as fh:
        previos = json.load(fh)
    res_rf = previos['CDS29_holdout']['resultados'][MODELO]
    params = res_rf['best_params']

    client = pymongo.MongoClient(CONNECTION)
    df, _ = cargar_dataset(client, 'CDS29')
    client.close()
    nombres = df.columns.drop('class').to_numpy()
    X = df.drop(columns='class').to_numpy()
    y = df['class'].to_numpy()
    ids = df.index.astype(str).to_numpy()
    i_tr, i_te, y_tr, y_te = particion_holdout(np.arange(len(y)), y)
    X_tr, X_te = X[i_tr], X[i_te]

    print('### CDS29: SHAP (TreeExplainer) del %s con %s' % (MODELO, params))
    t0 = time()
    modelo = espacio(MODELO, optuna.trial.FixedTrial(params)).fit(X_tr, y_tr)
    p_te = modelo.predict_proba(X_te)[:, 1]
    met = metricas(y_te, modelo.predict(X_te).astype(int), p_te)
    print(
        'Modelo reentrenado en %.1f s; test GMean=%.3f AUROC=%.3f (fase2_resultados.json: '
        'GMean=%.3f AUROC=%.3f)'
        % (
            time() - t0,
            met['GMean'],
            met['AUROC'],
            res_rf['test']['GMean'],
            res_rf['test']['AUROC'],
        )
    )

    t0 = time()
    explicacion = shap.TreeExplainer(modelo)(X_te)
    sv = explicacion.values[:, :, 1]  # contribuciones a P(activa)
    base = float(explicacion.base_values[0, 1])
    error = float(np.abs(base + sv.sum(axis=1) - p_te).max())
    print(
        'Valores SHAP de %d moleculas de test x %d bits en %.0f s; valor base E[P(activa)] = '
        '%.3f; error maximo de aditividad = %.1e'
        % (sv.shape[0], sv.shape[1], time() - t0, base, error)
    )

    importancia = np.abs(sv).mean(axis=0)
    top = np.argsort(importancia)[::-1][:N_TOP]
    figura_beeswarm(sv, X_te, nombres, ids[i_te], top)

    print(
        '\n%-6s %-8s %-10s %-12s %-12s %-10s %-10s'
        % (
            'Rango',
            'Bit',
            'media|SHAP|',
            'SHAP(bit=1)',
            'SHAP(bit=0)',
            '%act(bit=1)',
            '%act(bit=0)',
        )
    )
    tabla = []
    for r, j in enumerate(top, start=1):
        on = X_te[:, j] == 1
        fila = {
            'bit': str(nombres[j]),
            'media_abs_shap': float(importancia[j]),
            'shap_medio_bit1': float(sv[on, j].mean()) if on.any() else None,
            'shap_medio_bit0': float(sv[~on, j].mean()),
            'moleculas_test_bit1': int(on.sum()),
            'activas_entre_bit1': float(y_tr[X_tr[:, j] == 1].mean()),
            'activas_entre_bit0': float(y_tr[X_tr[:, j] == 0].mean()),
        }
        tabla.append(fila)
        print(
            '%-6d %-8s %.4f     %+.4f      %+.4f      %5.1f %%     %5.1f %%'
            % (
                r,
                fila['bit'],
                fila['media_abs_shap'],
                fila['shap_medio_bit1'] or 0.0,
                fila['shap_medio_bit0'],
                100 * fila['activas_entre_bit1'],
                100 * fila['activas_entre_bit0'],
            )
        )

    top_impureza = list(previos['CDS29_top_bits'])
    top10_shap = [str(nombres[j]) for j in top[:10]]
    comunes = [b for b in top10_shap if b in top_impureza]
    print(
        '\nTop-10 SHAP: %s\nTop-10 por impureza (fase 2): %s\nBits comunes: %d de 10 (%s)'
        % (top10_shap, top_impureza, len(comunes), ', '.join(comunes))
    )

    salida = {
        'modelo': MODELO,
        'best_params': params,
        'explicador': 'shap.TreeExplainer (tree_path_dependent), shap %s' % shap.__version__,
        'n_test': int(len(y_te)),
        'test': met,
        'valor_base': base,
        'error_aditividad_max': error,
        'top_bits': tabla,
        'top10_shap': top10_shap,
        'top10_impureza': top_impureza,
        'comunes_top10': comunes,
    }
    with open('shap_resultados.json', 'w') as fh:
        json.dump(salida, fh, indent=1, default=float, ensure_ascii=False)
    print('\nResultados guardados en shap_resultados.json y figura fig_shap_cds29.png / .html')


if __name__ == '__main__':
    main()
