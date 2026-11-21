#!/usr/bin/env python3
"""
ADP - Practicas (Apartado I)
Figura del cribado por similitud (Plotly) a partir de screening_resultados.json,
que genera screening_ampliado.py: proporcion de la coleccion que supera el filtrado
previo y aceleracion de la agregacion frente a la busqueda exhaustiva.

Autor: Pablo Caballero Maria
Uso: python3 figuras_screening.py
"""

import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

SERIE = ['#2a78d6', '#eb6834']  # CDS16, CDS29 (mismos colores que la fase 2)
TINTA, TINTA_2, REJILLA, EJE = '#0b0b0b', '#52514e', '#e6e5e1', '#b9b8b3'


def figura(res):
    """Figura de dos paneles a partir del resumen de screening_ampliado.py."""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=('Moléculas evaluadas (%)', 'Aceleración (t exhaustiva / t agregación)'),
    )
    for k, db in enumerate(['CDS16', 'CDS29']):
        filas = sorted((r for r in res if r['db'] == db), key=lambda r: r['threshold'])
        T = [r['threshold'] for r in filas]
        pct = [100 * r['candidatos'] / r['n_total'] for r in filas]
        acel = [r['t_fuerza_bruta_ms'] / r['t_agregacion_ms'] for r in filas]
        estilo_traza = dict(
            mode='lines+markers',
            line=dict(color=SERIE[k], width=2),
            marker=dict(color=SERIE[k], size=8, line=dict(color='white', width=1)),
            name=db,
            legendgroup=db,
        )
        fig.add_trace(
            go.Scatter(
                x=T,
                y=pct,
                showlegend=True,
                **estilo_traza,
                hovertemplate=db + ' · T = %{x}<br>%{y:.1f} % evaluado<extra></extra>',
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=T,
                y=acel,
                showlegend=False,
                **estilo_traza,
                hovertemplate=db + ' · T = %{x}<br>x%{y:.1f}<extra></extra>',
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=1, line=dict(color=EJE, width=1), row=1, col=2)
    fig.add_annotation(
        x=0.82,
        y=0,
        xref='x2',
        yref='y2',
        yshift=9,
        xanchor='right',
        showarrow=False,
        text='sin ventaja',
        font=dict(size=11, color=TINTA_2),
    )
    fig.update_xaxes(
        title_text='Umbral de Tanimoto T', tickvals=[0.3, 0.4, 0.6, 0.8], range=[0.27, 0.83]
    )
    fig.update_yaxes(range=[0, 100], dtick=20, row=1, col=1)
    fig.update_yaxes(
        type='log',
        tickvals=[0.25, 0.5, 1, 2, 4, 8],
        ticktext=['x0.25', 'x0.5', 'x1', 'x2', 'x4', 'x8'],
        range=[-0.7, 1.1],
        row=1,
        col=2,
    )
    fig.update_layout(
        template='simple_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial', size=13, color=TINTA_2),
        legend=dict(
            orientation='h', x=0, xanchor='left', y=1.13, yanchor='bottom', font=dict(color=TINTA)
        ),
        margin=dict(l=10, r=15, t=62, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=REJILLA, linecolor=EJE, ticks='', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=REJILLA, linecolor=EJE, ticks='', zeroline=False)
    fig.update_annotations(
        selector=dict(text='Moléculas evaluadas (%)'), font=dict(size=13, color=TINTA)
    )
    fig.update_annotations(
        selector=dict(text='Aceleración (t exhaustiva / t agregación)'),
        font=dict(size=13, color=TINTA),
    )
    return fig


def main(entrada='screening_resultados.json'):
    with open(entrada) as fh:
        fig = figura(json.load(fh))
    fig.write_html('fig_screening.html', include_plotlyjs='cdn')
    fig.write_image('fig_screening.png', width=640, height=290, scale=3)
    print('Figura guardada en fig_screening.png y fig_screening.html')


if __name__ == '__main__':
    main()
