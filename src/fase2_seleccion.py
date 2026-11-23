#!/usr/bin/env python3
"""
ADP - Practicas (Apartado I) - Fase 2 (ampliacion)
Seleccion e ingenieria de caracteristicas sobre CDS29.

Dos preguntas, las dos resueltas con validacion cruzada estratificada de 5 particiones sobre
la particion de entrenamiento (80 %) de CDS29, sin tocar el test hasta la medicion final:

  1. Seleccion: cuantos de los 1024 bits del fingerprint hacen falta y que criterio los elige
     mejor (filtrado univariante, importancia embebida y eliminacion recursiva).
  2. Ingenieria: si representaciones alternativas o adicionales al fingerprint binario
     (cuentas de subestructuras, descriptores fisico-quimicos, claves MACCS y caracteristicas
     de similitud) mejoran la prediccion.

Todo el preprocesado (rankings, umbrales, transformaciones y caracteristicas de similitud) se
ajusta dentro de cada particion de entrenamiento, de modo que la validacion no ve informacion
de las moleculas que evalua.

No repite el ajuste con Optuna: los hiperparametros se leen de fase2_resultados.json.

Autor: Pablo Caballero Maria
Uso: uv run python fase2_seleccion.py   (desde src/, despues de fase2_clasificacion.py)
"""

import json
from time import time

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import pymongo
from plotly.subplots import make_subplots
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as fpgen
from sklearn.feature_selection import RFE, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from documentos import moleculas
from fase2_clasificacion import (
    CONNECTION,
    N_BITS,
    SEED,
    SERIE,
    TINTA,
    TINTA_2,
    cargar_dataset,
    espacio,
    estilo,
    gmean,
    guardar,
    metricas,
    particion_holdout,
)

# RDKit construye sus modulos en tiempo de ejecucion, asi que ty no ve estas dos funciones.
RDLogger.DisableLog('rdApp.*')  # ty: ignore[missing-attribute]

MODELOS = ['Random Forest', 'LogReg (L2)']
N_FOLDS = 5
KS = [16, 32, 64, 128, 256, 512]  # numero de bits que se conserva en la curva de seleccion
FRECUENCIA_MIN = 0.01  # bits presentes en menos del 1 % de las moleculas: candidatos a descartar
CORR_MAX = 0.9  # correlacion por encima de la cual dos bits se consideran redundantes
C_L1 = 0.1  # regularizacion de la regresion logistica L1 que selecciona bits
N_VECINOS_SIM = 5  # vecinos mas similares que promedian las caracteristicas de similitud
SERIE5 = [*SERIE, '#e87ba4']  # quinta ranura categorica (magenta) para los cinco criterios


# ---------------------------------------------------------------------------
# Representaciones moleculares (RDKit) a partir del SMILES de cada documento
# ---------------------------------------------------------------------------
def representaciones(client, db_name):
    """Fingerprint binario de la BD y representaciones recalculadas con RDKit.

    Devuelve la matriz de bits almacenada en MongoDB (la misma que usa la fase 2), las cuentas
    de cada subestructura de Morgan, los descriptores fisico-quimicos y las claves MACCS.
    """
    df, _ = cargar_dataset(client, db_name)
    smiles = {d['_id']: d['smiles'] for d in moleculas(client[db_name]).find({}, {'smiles': 1})}
    ids = df.index.to_numpy()
    X = df.drop(columns='class').to_numpy()
    y = df['class'].to_numpy()

    # Morgan de radio 2 y 1024 bits: reproduce exactamente mfp.bits de la BD (se comprueba
    # mas abajo), de modo que la version con cuentas es la misma representacion sin binarizar.
    generador = fpgen.GetMorganGenerator(radius=2, fpSize=N_BITS)
    nombres_desc = [n for n, _ in Descriptors._descList]
    cuentas, descriptores, maccs = [], [], []
    for i in ids:
        mol = Chem.MolFromSmiles(smiles[i])
        cuentas.append(generador.GetCountFingerprintAsNumPy(mol))
        valores = Descriptors.CalcMolDescriptors(mol)
        descriptores.append([valores[n] for n in nombres_desc])
        claves = MACCSkeys.GenMACCSKeys(mol)  # ty: ignore[missing-attribute]
        maccs.append(np.array(claves, dtype=np.uint8))

    C = np.array(cuentas, dtype=np.int16)
    D = np.clip(np.nan_to_num(np.array(descriptores, dtype=np.float64), nan=np.nan), -1e12, 1e12)
    return {
        'ids': ids.astype(str),
        'nombres_bits': df.columns.drop('class').to_numpy().astype(str),
        'nombres_desc': np.array(nombres_desc),
        'X': X,
        'y': y,
        'C': C,
        'D': D,
        'M': np.array(maccs, dtype=np.uint8),
        'bits_reproducidos': bool(np.array_equal(X, (C > 0).astype(np.uint8))),
    }


# ---------------------------------------------------------------------------
# Ingenieria de caracteristicas: transformaciones ajustadas solo con el entrenamiento
# ---------------------------------------------------------------------------
def cuantiles(Da, Db):
    """Descriptores a rango uniforme [0, 1]; los ausentes toman la mediana del entrenamiento.

    Los descriptores tienen escalas y colas muy distintas (masas, logP, cuentas de anillos),
    asi que se transforman por cuantiles en lugar de estandarizarlos.
    """
    mediana = np.nanmedian(Da, axis=0)
    Da, Db = (np.where(np.isnan(Z), mediana, Z) for Z in (Da, Db))
    transformador = QuantileTransformer(n_quantiles=200, random_state=SEED).fit(Da)
    return transformador.transform(Da), transformador.transform(Db)


def tanimoto(A, B):
    """Matriz de coeficientes de Tanimoto entre las filas de A y las de B."""
    A, B = A.astype(np.float32), B.astype(np.float32)
    interseccion = A @ B.T
    union = A.sum(axis=1)[:, None] + B.sum(axis=1)[None, :] - interseccion
    return interseccion / np.maximum(union, 1.0)


def caracteristicas_similitud(Xa, ya, Xb, es_entrenamiento):
    """Cinco variables derivadas de la similitud con las moleculas de entrenamiento.

    Llevan el principio de similitud de Johnson y Maggiora a variables explicitas: para cada
    clase, la similitud maxima y la media de los N_VECINOS_SIM vecinos mas similares, mas la
    diferencia entre las dos similitudes maximas. Si las filas evaluadas son las propias de
    entrenamiento se descarta la diagonal, para que ninguna molecula se compare consigo misma.
    """
    S = tanimoto(Xb, Xa)
    if es_entrenamiento:
        np.fill_diagonal(S, -1.0)
    columnas = []
    for clase in (1, 0):
        vecinos = np.sort(S[:, ya == clase], axis=1)[:, ::-1]
        columnas += [vecinos[:, 0], vecinos[:, :N_VECINOS_SIM].mean(axis=1)]
    columnas.append(columnas[0] - columnas[2])
    return np.column_stack(columnas)


def nombres_similitud():
    return [
        'SimMaxActivas',
        'SimTop%dActivas' % N_VECINOS_SIM,
        'SimMaxInactivas',
        'SimTop%dInactivas' % N_VECINOS_SIM,
        'SimMaxActivas - SimMaxInactivas',
    ]


# ---------------------------------------------------------------------------
# Seleccion de caracteristicas: tres familias de criterios sobre los 1024 bits
# ---------------------------------------------------------------------------
def rankings(Xa, ya, modelo_rf, modelo_lr):
    """Ordenaciones de los bits de mejor a peor segun cada criterio.

    Filtrado univariante (chi cuadrado e informacion mutua), seleccion embebida (importancia
    por impureza del Random Forest y magnitud de los coeficientes de una logistica L1) y
    seleccion envolvente (eliminacion recursiva de caracteristicas con la logistica L2).
    """
    orden = {}
    orden['Chi cuadrado'] = np.argsort(-np.nan_to_num(chi2(Xa, ya)[0]), kind='stable')
    orden['Información mutua'] = np.argsort(
        -mutual_info_classif(Xa, ya, discrete_features=True, random_state=SEED), kind='stable'
    )
    orden['Importancia RF'] = np.argsort(-modelo_rf.fit(Xa, ya).feature_importances_, kind='stable')
    l1 = LogisticRegression(penalty='l1', solver='liblinear', C=C_L1, class_weight='balanced').fit(
        Xa, ya
    )
    orden['Logística L1'] = np.argsort(-np.abs(l1.coef_[0]), kind='stable')
    # RFE elimina un 10 % de los bits en cada paso y guarda el orden de eliminacion en
    # ranking_, de modo que ordenar por ranking_ da la lista completa de peor a mejor.
    rfe = RFE(modelo_lr, n_features_to_select=min(KS), step=0.1).fit(Xa, ya)
    orden['RFE (envolvente)'] = np.argsort(rfe.ranking_, kind='stable')
    return orden, int((l1.coef_ != 0).sum())


def filtro_frecuencia(Xa):
    """Bits presentes en al menos FRECUENCIA_MIN de las moleculas de entrenamiento."""
    frecuencia = Xa.mean(axis=0)
    return np.flatnonzero(frecuencia >= FRECUENCIA_MIN)


def filtro_redundancia(Xa, candidatos):
    """De cada grupo de bits con |r| > CORR_MAX se conserva solo el primero."""
    R = np.nan_to_num(np.corrcoef(Xa[:, candidatos].T.astype(np.float32)))
    conservados = []
    for i in range(len(candidatos)):
        if all(abs(R[i, j]) <= CORR_MAX for j in conservados):
            conservados.append(i)
    return candidatos[conservados]


# ---------------------------------------------------------------------------
# Espacios de caracteristicas que se comparan
# ---------------------------------------------------------------------------
BASE = 'Base: 1024 bits'


def construir_espacios(datos, a, v, ya):
    """Diccionario nombre -> (matriz de entrenamiento, matriz de validacion).

    Incluye los espacios de ingenieria (representaciones alternativas y uniones) y los dos
    filtros de seleccion que no dependen de un numero de bits prefijado.
    """
    X, C, D, M = datos['X'], datos['C'], datos['D'], datos['M']
    Xa, Xv = X[a], X[v]
    Ca, Cv = np.log1p(C[a]), np.log1p(C[v])  # las cuentas tienen cola larga
    Da, Dv = cuantiles(D[a], D[v])
    Ma, Mv = M[a], M[v]
    Sa = caracteristicas_similitud(Xa, ya, Xa, True)
    Sv = caracteristicas_similitud(Xa, ya, Xv, False)
    frecuentes = filtro_frecuencia(Xa)
    no_redundantes = filtro_redundancia(Xa, frecuentes)
    return {
        BASE: (Xa, Xv),
        'Cuentas de subestructuras': (Ca, Cv),
        'Descriptores fisicoquímicos': (Da, Dv),
        'Claves MACCS': (Ma, Mv),
        'Variables de similitud': (Sa, Sv),
        'Bits + descriptores': (np.hstack([Xa, Da]), np.hstack([Xv, Dv])),
        'Bits + MACCS': (np.hstack([Xa, Ma]), np.hstack([Xv, Mv])),
        'Bits + similitud': (np.hstack([Xa, Sa]), np.hstack([Xv, Sv])),
        'Descriptores + MACCS + similitud': (
            np.hstack([Da, Ma, Sa]),
            np.hstack([Dv, Mv, Sv]),
        ),
        'Cuentas + descriptores': (np.hstack([Ca, Da]), np.hstack([Cv, Dv])),
        'Todas las representaciones': (
            np.hstack([Xa, Ca, Da, Ma, Sa]),
            np.hstack([Xv, Cv, Dv, Mv, Sv]),
        ),
        'Filtro de frecuencia': (Xa[:, frecuentes], Xv[:, frecuentes]),
        'Frecuencia + redundancia': (Xa[:, no_redundantes], Xv[:, no_redundantes]),
    }


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figura_seleccion(curvas, base_cv, criterios):
    """Un panel por modelo: GMean en validacion cruzada frente al numero de bits."""
    fig = make_subplots(
        rows=1,
        cols=len(MODELOS),
        shared_yaxes=True,
        horizontal_spacing=0.035,
        subplot_titles=MODELOS,
    )
    for col, modelo in enumerate(MODELOS, start=1):
        for k, criterio in enumerate(criterios):
            medias = [curvas[(modelo, criterio, kk)][0] for kk in KS]
            fig.add_trace(
                go.Scatter(
                    x=KS,
                    y=medias,
                    mode='lines+markers',
                    line=dict(color=SERIE5[k], width=2),
                    marker=dict(color=SERIE5[k], size=8, line=dict(color='white', width=1)),
                    name=criterio,
                    legendgroup=criterio,
                    showlegend=(col == 1),
                    customdata=[curvas[(modelo, criterio, kk)][1] for kk in KS],
                    hovertemplate=criterio
                    + ' · '
                    + modelo
                    + '<br>%{x} bits<br>GMean = %{y:.3f} ± %{customdata:.3f}<extra></extra>',
                ),
                row=1,
                col=col,
            )
        fig.add_hline(
            y=base_cv[modelo][0],
            line=dict(color=TINTA_2, width=1, dash='dot'),
            row=1,
            col=col,
        )
        eje = '' if col == 1 else str(col)
        fig.add_annotation(
            text='Los 1024 bits: %.3f' % base_cv[modelo][0],
            xref='x%s domain' % eje,
            yref='y%s' % eje,
            x=0.03,
            y=base_cv[modelo][0],
            xanchor='left',
            yanchor='bottom',
            showarrow=False,
            font=dict(size=11, color=TINTA_2),
            bgcolor='rgba(255,255,255,0.8)',
        )
    estilo(fig)
    fig.update_xaxes(
        title_text='Bits conservados', type='log', tickvals=KS, ticktext=[str(k) for k in KS]
    )
    fig.update_yaxes(title_text='GMean (CV, 5 particiones)', row=1, col=1)
    fig.update_annotations(selector=dict(text=None), font=dict(size=13, color=TINTA))
    for t in MODELOS:
        fig.update_annotations(selector=dict(text=t), font=dict(size=13, color=TINTA))
    fig.update_layout(legend=dict(y=1.1), margin=dict(t=72))
    guardar(fig, 'fig_seleccion_cds29', 760, 330)


def figura_ingenieria(gm, nombres, dimensiones):
    """Puntos con barra de error: GMean de cada espacio de caracteristicas y modelo."""
    fig = go.Figure()
    for k, modelo in enumerate(MODELOS):
        medias = [gm[(modelo, n)][0] for n in nombres]
        desv = [gm[(modelo, n)][1] for n in nombres]
        fig.add_trace(
            go.Scatter(
                x=medias,
                y=[i + (-0.17 if k == 0 else 0.17) for i in range(len(nombres))],
                error_x=dict(type='data', array=desv, color=SERIE[k], thickness=1.5, width=0),
                mode='markers',
                marker=dict(color=SERIE[k], size=9, line=dict(color='white', width=1)),
                name=modelo,
                customdata=[[n, dimensiones[n], d] for n, d in zip(nombres, desv, strict=True)],
                hovertemplate='%{customdata[0]} · '
                + modelo
                + '<br>%{customdata[1]} variables'
                + '<br>GMean = %{x:.3f} ± %{customdata[2]:.3f}<extra></extra>',
            )
        )
    fig.update_yaxes(
        tickvals=list(range(len(nombres))),
        ticktext=['%s  (%d)' % (n, dimensiones[n]) for n in nombres],
        autorange='reversed',
        showgrid=False,
        tickfont=dict(color=TINTA),
    )
    fig.update_xaxes(title_text='GMean (CV, 5 particiones)', dtick=0.05)
    estilo(fig)
    fig.update_layout(legend=dict(y=1.04), margin=dict(l=10, t=58))
    guardar(fig, 'fig_ingenieria_cds29', 700, 420)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    with open('fase2_resultados.json') as fh:
        previos = json.load(fh)['CDS29_holdout']['resultados']
    params = {m: previos[m]['best_params'] for m in MODELOS}

    def modelo(nombre):
        return espacio(nombre, optuna.trial.FixedTrial(params[nombre]))

    client = pymongo.MongoClient(CONNECTION)
    t0 = time()
    datos = representaciones(client, 'CDS29')
    client.close()
    y = datos['y']
    i_tr, i_te, y_tr, y_te = particion_holdout(np.arange(len(y)), y)
    print(
        '### CDS29: seleccion e ingenieria de caracteristicas\n'
        'Representaciones calculadas en %.0f s (RDKit): %d bits de la BD, %d cuentas de '
        'subestructuras, %d descriptores fisico-quimicos, %d claves MACCS y %d variables de '
        'similitud.\nMorgan (radio 2, %d bits) recalculado con RDKit reproduce mfp.bits: %s'
        % (
            time() - t0,
            datos['X'].shape[1],
            datos['C'].shape[1],
            datos['D'].shape[1],
            datos['M'].shape[1],
            len(nombres_similitud()),
            N_BITS,
            'si' if datos['bits_reproducidos'] else 'no',
        )
    )
    print(
        'Hiperparametros de Optuna: %s\n'
        'Validacion cruzada estratificada de %d particiones sobre el entrenamiento '
        '(%d moleculas, %d activas); test reservado: %d moleculas.\n'
        % (params, N_FOLDS, len(y_tr), int(y_tr.sum()), len(y_te))
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    curvas, espacios_cv, dimensiones, l1_no_nulos, elegidos = {}, {}, {}, [], {}
    for k, (tr, va) in enumerate(cv.split(i_tr, y_tr)):
        t_fold = time()
        a, v = i_tr[tr], i_tr[va]
        ya, yv = y[a], y[v]
        Xa, Xv = datos['X'][a], datos['X'][v]

        orden, n_l1 = rankings(Xa, ya, modelo(MODELOS[0]), modelo(MODELOS[1]))
        l1_no_nulos.append(n_l1)
        for criterio, indices in orden.items():
            elegidos.setdefault(criterio, []).append(indices[: min(KS)].tolist())
            for n_bits in KS:
                sel = indices[:n_bits]
                for nombre in MODELOS:
                    m = modelo(nombre).fit(Xa[:, sel], ya)
                    g = gmean(yv, m.predict(Xv[:, sel]).astype(int))
                    curvas.setdefault((nombre, criterio, n_bits), []).append(g)

        for nombre_esp, (A, B) in construir_espacios(datos, a, v, ya).items():
            dimensiones[nombre_esp] = A.shape[1]
            for nombre in MODELOS:
                m = modelo(nombre).fit(A, ya)
                espacios_cv.setdefault((nombre, nombre_esp), []).append(
                    metricas(yv, m.predict(B).astype(int), m.predict_proba(B)[:, 1])
                )
        print('  particion %d/%d completada (%.0f s)' % (k + 1, N_FOLDS, time() - t_fold))

    criterios = list(orden)

    # --- Resultados de la seleccion -------------------------------------------------------
    curva = {c: (float(np.mean(g)), float(np.std(g, ddof=1))) for c, g in curvas.items()}
    gm, metricas_cv = {}, {}
    for clave, filas in espacios_cv.items():
        d = pd.DataFrame(filas)
        gm[clave] = (float(d.GMean.mean()), float(d.GMean.std()))
        metricas_cv[clave] = {'media': d.mean().to_dict(), 'std': d.std().to_dict()}
    base_cv = {m: gm[(m, BASE)] for m in MODELOS}

    print('\n## Seleccion de caracteristicas: GMean en validacion cruzada')
    print('%-20s %-18s %s' % ('Criterio', 'Modelo', '  '.join('%7d' % k for k in KS)))
    for criterio in criterios:
        for nombre in MODELOS:
            print(
                '%-20s %-18s %s'
                % (
                    criterio,
                    nombre,
                    '  '.join('%7.3f' % curva[(nombre, criterio, k)][0] for k in KS),
                )
            )
    for nombre in MODELOS:
        print('%-20s %-18s %7.3f (referencia)' % ('1024 bits', nombre, base_cv[nombre][0]))

    print('\n## Ingenieria de caracteristicas: GMean en validacion cruzada')
    print('%-40s %-8s %-15s %-15s' % ('Espacio', 'Vars.', MODELOS[0], MODELOS[1]))
    orden_fig = sorted(dimensiones, key=lambda n: -gm[(MODELOS[0], n)][0])
    for nombre_esp in orden_fig:
        print(
            '%-40s %-8d %s'
            % (
                nombre_esp,
                dimensiones[nombre_esp],
                '  '.join('%.3f ± %.3f' % gm[(m, nombre_esp)] for m in MODELOS),
            )
        )

    # --- Test: base, mejor seleccion y mejor espacio ---------------------------------------
    print('\n## Test reservado (%d moleculas)' % len(y_te))
    Xa_full, Xv_full = datos['X'][i_tr], datos['X'][i_te]
    orden_full, _ = rankings(Xa_full, y_tr, modelo(MODELOS[0]), modelo(MODELOS[1]))
    espacios_full = construir_espacios(datos, i_tr, i_te, y_tr)
    test = {}
    for nombre in MODELOS:
        # En caso de empate gana el espacio con menos variables, que es el objetivo de la
        # seleccion: el mismo GMean con menos informacion.
        pares = [(c, n_bits) for c in criterios for n_bits in KS]
        mejor_sel = max(pares, key=lambda par: (curva[(nombre, par[0], par[1])][0], -par[1]))
        mejor_esp = max(dimensiones, key=lambda n: (gm[(nombre, n)][0], -dimensiones[n]))
        sel = orden_full[mejor_sel[0]][: mejor_sel[1]]
        candidatos = {
            'base': (BASE, (Xa_full, Xv_full)),
            'seleccion': (
                '%s, %d bits' % mejor_sel,
                (Xa_full[:, sel], Xv_full[:, sel]),
            ),
            'ingenieria': (mejor_esp, espacios_full[mejor_esp]),
        }
        test[nombre] = {}
        for etiqueta, (descripcion, (A, B)) in candidatos.items():
            m = modelo(nombre).fit(A, y_tr)
            met = metricas(y_te, m.predict(B).astype(int), m.predict_proba(B)[:, 1])
            test[nombre][etiqueta] = {
                'espacio': descripcion,
                'n_variables': int(A.shape[1]),
                'test': met,
            }
            print(
                '  %-18s %-12s %-42s %4d vars.  GMean=%.3f Kappa=%.3f Acc=%.3f AUROC=%.3f'
                % (
                    nombre,
                    etiqueta,
                    descripcion,
                    A.shape[1],
                    met['GMean'],
                    met['Kappa'],
                    met['Accuracy'],
                    met['AUROC'],
                )
            )

    # --- Bits coincidentes entre criterios ------------------------------------------------
    estables = {}
    print('\n## Bits elegidos (primeros %d) por criterio' % min(KS))
    for criterio in criterios:
        conjuntos = [set(c) for c in elegidos[criterio]]
        comunes = sorted(set.intersection(*conjuntos))
        estables[criterio] = [str(datos['nombres_bits'][j]) for j in comunes]
        print(
            '  %-20s %2d bits en las %d particiones: %s'
            % (criterio, len(comunes), N_FOLDS, ', '.join(estables[criterio][:8]))
        )
    nucleo = sorted(set.intersection(*(set(estables[c]) for c in criterios)))
    print('  Comunes a los cinco criterios: %s' % (', '.join(nucleo) or '(ninguno)'))

    figura_seleccion(curva, base_cv, criterios)
    figura_ingenieria(gm, orden_fig, dimensiones)

    salida = {
        'modelos': MODELOS,
        'best_params': params,
        'folds': N_FOLDS,
        'n_train': int(len(y_tr)),
        'n_test': int(len(y_te)),
        'ks': KS,
        'frecuencia_min': FRECUENCIA_MIN,
        'corr_max': CORR_MAX,
        'c_l1': C_L1,
        'bits_no_nulos_l1_por_fold': l1_no_nulos,
        'morgan_reproduce_bd': datos['bits_reproducidos'],
        'nombres_similitud': nombres_similitud(),
        'seleccion': {
            '%s | %s | %d bits' % (m, c, k): {
                'gmean': curva[(m, c, k)][0],
                'gmean_std': curva[(m, c, k)][1],
            }
            for (m, c, k) in curva
        },
        'bits_estables': estables,
        'bits_nucleo': nucleo,
        'ingenieria': {
            '%s | %s' % (m, n): {
                'n_variables': dimensiones[n],
                'gmean': gm[(m, n)][0],
                'gmean_std': gm[(m, n)][1],
                'media': metricas_cv[(m, n)]['media'],
                'std': metricas_cv[(m, n)]['std'],
            }
            for (m, n) in gm
        },
        'test': test,
    }
    with open('seleccion_resultados.json', 'w') as fh:
        json.dump(salida, fh, indent=1, default=float, ensure_ascii=False)
    print(
        '\nResultados guardados en seleccion_resultados.json y figuras '
        'fig_seleccion_cds29 / fig_ingenieria_cds29 (.png y .html)'
    )


if __name__ == '__main__':
    main()
