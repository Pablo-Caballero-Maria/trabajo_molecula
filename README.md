# Prácticas ADP (Apartado I)

Máster en Inteligencia Computacional e Internet de las Cosas · Análisis, Diseño y
Procesamiento de Datos Aplicados a las Ciencias y a las Tecnologías (ADP).

Autor: Pablo Caballero María.

El trabajo tiene dos fases y una ampliación:

1. **Fase 1 · MongoDB.** Traducción del esquema entidad-interrelación DEPT/EMP (con la
   interrelación reflexiva de jefatura) a colecciones con validación `$jsonSchema`, carga de
   datos de prueba e informes Q1-Q11 con el framework de agregación (`$lookup`,
   `$graphLookup`, `updateMany` con pipeline).
2. **Fase 2 · Predicción de la actividad biológica.** Clasificación activo/inactivo a partir
   del fingerprint molecular de 1024 bits de las bases de datos CDS16 y CDS29: siete
   clasificadores (regresión logística, SVM, k-NN con Jaccard, Naive Bayes de Bernoulli,
   Random Forest, Extra Trees y una red neuronal MLP) evaluados con validación cruzada
   estratificada repetida, y ajuste de hiperparámetros con **Optuna** (TPE + poda por
   mediana) sobre una partición de entrenamiento de CDS29, con evaluación final en test.
3. **Ampliación · cribado virtual por similitud.** Búsqueda por coeficiente de Tanimoto con
   el pipeline de agregación de la asignatura, comparada con una búsqueda exhaustiva.

Todas las figuras se generan con **Plotly** (PNG para la memoria y HTML interactivo).

## Estructura

```
src/
  fase1_emp_dept.py                     Fase 1: colecciones, validación, carga e informes
  fase2_clasificacion.py                Fase 2: fingerprints, CV, Optuna, test y figuras
  screening_ampliado.py                 Ampliación: cribado con varios umbrales
  figuras_screening.py                  Figura del cribado (a partir de screening_resultados.json)
  screening_Mol-database-only_mongo_p.py  Ejemplo de la asignatura (sin cambios)
  *_salida.txt, *_resultados.json       Salidas de cada script
  fig_*.png / fig_*.html                Figuras (estáticas e interactivas)
  build_doc.py, memoria.html            Generadores de la primera versión de la memoria
                                        (desfasados; la memoria vigente está en entregables/)
instrucciones/                          Enunciado, teoría, Readme y copia de seguridad (dump.tar)
entregables/Pablo-Caballero-Maria.zip   Entrega: memoria (PDF y DOCX) + código y resultados
```

## Cómo reproducirlo

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Restaurar las bases de datos moleculares (CDS16 y CDS29) desde la copia de seguridad
tar xf instrucciones/dump.tar
mongorestore "mongodb://adp:adp@localhost:27017/?authSource=admin" dump

cd src
../.venv/bin/python fase1_emp_dept.py        > fase1_salida.txt
../.venv/bin/python fase2_clasificacion.py   > fase2_salida.txt
../.venv/bin/python screening_ampliado.py    > screening_salida.txt
../.venv/bin/python figuras_screening.py
```

Los scripts usan por defecto `mongodb://adp:adp@127.0.0.1:27017/?authSource=admin`.
`fase2_clasificacion.py` admite otra cadena de conexión en la variable de entorno
`MONGO_URI`. La exportación de figuras a PNG (Kaleido) necesita Chrome o Chromium instalado.
