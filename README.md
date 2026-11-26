<div align="center">

# 🧪 Prácticas ADP (Apartado I)

**MongoDB y predicción de la actividad biológica de compuestos a partir de fingerprints moleculares**

Máster en Inteligencia Computacional e Internet de las Cosas (Universidad de Córdoba) · Análisis,
Diseño y Procesamiento de Datos Aplicados a las Ciencias y a las Tecnologías (ADP)

[![CI](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/actions/workflows/ci.yml/badge.svg)](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/actions/workflows/codeql.yml/badge.svg)](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/actions/workflows/codeql.yml)
[![Cobertura](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FPablo-Caballero-Maria%2Ftrabajo_molecula%2Fbadges%2Fcoverage.json)](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/actions/workflows/ci.yml)
[![Versión](https://img.shields.io/github/v/release/Pablo-Caballero-Maria/trabajo_molecula?label=versi%C3%B3n)](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/releases)
[![Licencia: MIT](https://img.shields.io/github/license/Pablo-Caballero-Maria/trabajo_molecula?label=licencia)](LICENSE)

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-agregaci%C3%B3n-47A248?logo=mongodb&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-TPE-2F6EBA)
![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-FF0D57)
![RDKit](https://img.shields.io/badge/RDKit-2026.09-0B7285)
![Plotly](https://img.shields.io/badge/Plotly-figuras-3F4F75?logo=plotly&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-77_tests-0A9EDC?logo=pytest&logoColor=white)
![uv · Ruff · ty](https://img.shields.io/badge/uv_·_Ruff_·_ty-Astral-DE5FE9?logo=astral&logoColor=white)

[Memoria (PDF)](entregables/Pr%C3%A1cticas%20ADP%20%28Apartado%20I%29%20-%20Pablo%20Caballero%20Mar%C3%ADa.pdf) ·
[Wiki](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki) ·
[Registro de cambios](CHANGELOG.md) ·
[Resumen en mi portfolio](https://pablo-caballero.vercel.app/projects/molecular-activity)

<img src="src/fig_shap_cds29.png" alt="Gráfico de enjambre de SHAP: cuánto empuja cada bit del fingerprint a una molécula hacia activa o inactiva" width="640">

</div>

Autor: Pablo Caballero María.

| 🧬 Moléculas (CDS29) | 🤖 Clasificadores | 🏆 Mejor en test | ✂️ Bits necesarios | 🔎 Cribado a T = 0,8 |
| :---: | :---: | :---: | :---: | :---: |
| 7015 (19,6 % activas) | 7 | GMean 0,669 · AUROC 0,723 | 256 de 1024 | evalúa el 5,8 % de la colección |

## Índice

- [Qué contiene](#qué-contiene)
- [Ruido × outliers en la validación cruzada](#ruido--outliers-en-la-validación-cruzada)
- [Selección e ingeniería de características](#selección-e-ingeniería-de-características)
- [Explicabilidad con SHAP](#explicabilidad-con-shap)
- [Cribado virtual por similitud](#cribado-virtual-por-similitud)
- [Estructura](#estructura)
- [Cómo reproducirlo (en nativo)](#cómo-reproducirlo-en-nativo)
- [Tests](#tests)
- [Calidad del código e integración continua](#calidad-del-código-e-integración-continua)
- [Documentación](#documentación)
- [Contribuir](#contribuir)
- [Cómo citar](#cómo-citar)
- [Licencia](#licencia)

## Qué contiene

El trabajo tiene dos fases y una ampliación:

```mermaid
flowchart LR
    subgraph F1["Fase 1 · MongoDB"]
        ER["Esquema E-R DEPT/EMP<br/>con jefatura reflexiva"] --> COL[("dept · emp<br/>$jsonSchema estricto")]
        COL --> Q["Informes Q1-Q11<br/>$lookup · $graphLookup · updateMany"]
    end
    subgraph F2["Fase 2 · Actividad biológica"]
        DB[("CDS16 · CDS29<br/>SMILES + mfp.bits")] --> X["Matriz de bits<br/>1024 columnas"]
        X --> M["7 clasificadores<br/>CV estratificada repetida"]
        M --> O["Optuna (TPE + poda)<br/>test hold-out 80/20"]
        O --> R["Ruido × outliers"]
        O --> S["Selección e ingeniería<br/>(RDKit)"]
        O --> SH["SHAP"]
    end
    subgraph AM["Ampliación"]
        DB --> TS["Cribado por Tanimoto<br/>pipeline de agregación"]
    end
```

1. **Fase 1 · MongoDB.** Traducción del esquema entidad-interrelación DEPT/EMP (con la interrelación
   reflexiva de jefatura) a colecciones con validación `$jsonSchema`, carga de datos de prueba e
   informes Q1-Q11 con el framework de agregación (`$lookup`, `$graphLookup`, `updateMany` con
   pipeline).
2. **Fase 2 · Predicción de la actividad biológica.** Clasificación activo/inactivo a partir del
   fingerprint molecular de 1024 bits de las bases de datos CDS16 y CDS29:
   - siete clasificadores (regresión logística, SVM, k-NN con Jaccard, Naive Bayes de Bernoulli, Random
     Forest, Extra Trees y una red neuronal MLP) evaluados con validación cruzada estratificada
     repetida;
   - ajuste de hiperparámetros con **Optuna** (TPE + poda por mediana) sobre una partición de
     entrenamiento de CDS29, con evaluación final en test;
   - **ruido × outliers**: validación cruzada de 12 combinaciones de inyección de ruido y eliminación de
     outliers, con la frontera de decisión de cada una sobre dos componentes principales;
   - **selección e ingeniería de características**: cinco criterios de selección de bits (filtrado,
     embebidos y envolvente) frente a seis representaciones moleculares calculadas con RDKit (cuentas,
     descriptores fisicoquímicos, claves MACCS y variables de similitud);
   - **explicabilidad con SHAP**: contribución de cada bit del fingerprint a la probabilidad de ser
     activa (gráfico de enjambre o *beeswarm*).
3. **Ampliación · cribado virtual por similitud.** Búsqueda por coeficiente de Tanimoto con el pipeline
   de agregación de la asignatura, comparada con una búsqueda exhaustiva.

Todas las figuras se generan con **Plotly** (PNG para la memoria y HTML interactivo).

| Base de datos | Moléculas | Activas | Mejor modelo en CV (GMean) | GMean | AUROC |
| --- | ---: | ---: | --- | ---: | ---: |
| CDS16 | 82 | 50 % | Random Forest | 0,881 | 0,959 |
| CDS29 | 7015 | 19,6 % | SVM (RBF) | 0,643 | 0,720 |

![GMean y AUROC medios de los siete clasificadores en CDS16 y CDS29](src/fig_cv_modelos.png)

> [!NOTE]
> En CDS29 la exactitud engaña: el Random Forest tiene la mejor (0,802) pero solo encuentra el 14,7 % de
> las moléculas activas. Por eso se usan métricas que resisten el desbalance: GMean, kappa de Cohen y
> AUROC.

## Ruido × outliers en la validación cruzada

`fase2_ruido_outliers.py` toma el Random Forest con los hiperparámetros que encontró Optuna (leídos de
`fase2_resultados.json`, sin repetir la búsqueda) y lo evalúa con validación cruzada estratificada de 5
particiones sobre la partición de entrenamiento de CDS29 (5612 moléculas). En cada partición, y solo
sobre su parte de entrenamiento:

1. **Eliminación de outliers** (5 % de cada clase, detectados por separado en activas e inactivas para
   no vaciar la clase minoritaria): ninguna, *Isolation Forest* o *Local Outlier Factor* con distancia
   de Jaccard (1 − Tanimoto).
2. **Inyección de ruido** como aumento de datos (se añade una copia ruidosa del entrenamiento): ninguno,
   gaussiano (σ = 0,15), volteo de bits (cada bit se invierte con p = 0,005) o enmascarado (cada bit a 1
   se apaga con p = 0,10).

La figura dibuja una celda por combinación (filas: ruido; columnas: outliers). El fondo es P(activa) en
el plano de las dos primeras componentes principales (azul = inactiva, rojo = activa, línea gris =
frontera 0,5) y encima están las moléculas de validación del primer fold con su **clase real**: un punto
rojo sobre fondo rojo, o azul sobre azul, está bien clasificado. La PCA se ajusta con el entrenamiento
original del fold, así que todas las celdas comparten ejes. Como un modelo de 1024 dimensiones no tiene
una frontera dibujable en 2D, el fondo es el mismo Random Forest reentrenado con los mismos datos
proyectados en esas dos componentes: es su frontera exacta, no una aproximación. Cada celda indica la
GMean del modelo real (1024 bits, media ± desviación de las 5 particiones) y la de ese modelo 2D.

![Fronteras de decisión ruido × outliers](src/fig_ruido_outliers.png)

Resultados (detalle en `src/ruido_outliers_salida.txt`):

- Ninguna combinación cambia el resultado de forma clara: la GMean en validación cruzada va de 0,610 a
  0,630, con desviaciones de ±0,01-0,02. La mejor es **volteo de bits sin eliminar outliers** (0,630
  frente a 0,625 del caso base).
- Eliminar outliers no ayuda: con *Isolation Forest* y sin ruido la GMean baja a 0,610. En un conjunto
  químicamente diverso, las moléculas raras no son errores, sino parte del espacio que el modelo tiene
  que cubrir.
- En el test reservado, el caso base reproduce exactamente el Random Forest de la fase 2 (GMean 0,632,
  AUROC 0,717) y el volteo de bits lo mejora a GMean 0,660 y AUROC 0,727. La mejora es mayor que la
  observada en validación cruzada, así que parte puede deberse al azar de un único test.
- Las dos primeras componentes explican solo el 7,2 % de la varianza: en ese plano las clases se
  solapan y la GMean del modelo 2D se queda en torno a 0,52. La información que separa activas de
  inactivas está repartida entre muchos bits, como confirma SHAP.

## Selección e ingeniería de características

`fase2_seleccion.py` responde a dos preguntas con el mismo protocolo: validación cruzada estratificada
de 5 particiones sobre las 5612 moléculas de entrenamiento de CDS29, con el Random Forest y la regresión
logística que ajustó Optuna, y el test reservado solo para la medición final. Todo el preprocesado
—rankings, umbrales, transformación por cuantiles y variables de similitud— se ajusta **dentro** de cada
partición de entrenamiento, de modo que la validación nunca ve información derivada de las moléculas que
evalúa.

**Selección**: cinco criterios de tres familias, evaluados conservando 16, 32, 64, 128, 256 y 512 de los
1024 bits. Filtrado univariante (chi cuadrado e información mutua), selección embebida (importancia por
impureza del Random Forest y magnitud de los coeficientes de una logística L1) y selección envolvente
(eliminación recursiva de características, RFE).

![Selección de características](src/fig_seleccion_cds29.png)

**Ingeniería**: representaciones recalculadas con RDKit a partir del SMILES de cada molécula. Se
comprueba que el fingerprint de Morgan de radio 2 y 1024 bits reproduce exactamente `mfp.bits` de
MongoDB, así que las cuentas de subestructuras son la misma representación sin binarizar. A ellas se
suman 217 descriptores fisicoquímicos (transformados por cuantiles), las 167 claves MACCS y 5 variables
de similitud (máximo y media de los 5 vecinos más similares de cada clase, y su diferencia), solas y
combinadas con los bits.

![Espacios de representación](src/fig_ingenieria_cds29.png)

Resultados (detalle en `src/seleccion_salida.txt`):

- **El fingerprint se puede reducir a una cuarta parte sin coste medible**: con 256 bits el mejor
  criterio llega a 0,627 en el Random Forest (referencia con 1024 bits: 0,625) y a 0,632 en la logística
  (referencia: 0,632). Por debajo de 64 bits la GMean cae con claridad, hasta 0,535-0,590 con 16 bits.
- **Ningún criterio domina.** El RFE, con diferencia el más caro, es el mejor con 16 bits (0,584 y
  0,590) y comparte el primer puesto en los tamaños grandes, pero chi cuadrado y la información mutua se
  quedan a pocas milésimas —y a veces por delante— con una fracción del coste. La logística L1 es la
  peor con 16 bits (0,535) y de las mejores a partir de 128.
- **Los bits elegidos son estables y coinciden con SHAP**: FP111 está entre los 16 primeros de los cinco
  criterios en las cinco particiones; FP98 y FP379 lo están en cuatro criterios, y FP450, FP695 y FP831
  en tres. Son los mismos bits que encabezan el análisis SHAP.
- **La única representación que mejora a los bits es la similitud.** `Bits + similitud` gana en
  validación cruzada con los dos modelos (RF 0,635; logística 0,640) y en el test lleva a la logística
  hasta **GMean 0,669 y AUROC 0,723**, el mejor resultado de todo el trabajo (el SVM de la fase 2 se
  quedaba en 0,665). Las cinco variables codifican el principio de similitud de Johnson y Maggiora, lo
  que el k-NN hacía mal como clasificador y aquí funciona como información añadida.
- **Los descriptores fisicoquímicos no aportan y al Random Forest lo estorban**: solos dan 0,590 (RF) y
  0,604 (logística), y unidos a los bits hunden al Random Forest hasta 0,586, porque 217 variables
  continuas compiten con los bits en cada división del árbol. Las claves MACCS, con solo 167 bits,
  llegan a 0,613: casi lo mismo que los 1024 de Morgan.
- Para el Random Forest, la mejora de `Bits + similitud` no se confirma en el test (0,625 frente a 0,632
  del caso base), un recordatorio de que un único test de 1403 moléculas tiene su propio azar.

## Explicabilidad con SHAP

`fase2_shap.py` reentrena una vez el Random Forest de Optuna (misma partición y semilla, de modo que es
el mismo bosque evaluado en test) y calcula con `shap.TreeExplainer`, exacto para bosques, los valores
SHAP de las 1403 moléculas de test. Para clasificación binaria con muchas variables, el gráfico adecuado
es el de enjambre (*beeswarm*): ordena los bits por su impacto medio |SHAP| y muestra cada molécula como
un punto cuyo eje x es cuánto desplaza P(activa) y cuyo color es el valor del bit (presente o ausente).
La figura está en la cabecera de este README.

- El valor base es 0,500 (el bosque pondera las clases) y la suma de las contribuciones reproduce la
  probabilidad predicha con un error máximo de 1e-9.
- FP98, FP111, FP363 y FP185 empujan hacia *activa* cuando están presentes: entre las moléculas de
  entrenamiento con FP98 o FP111 activo hay un 45-48 % de activas, frente al 19,6 % global. FP379,
  FP695, FP722 y, sobre todo, FP831 empujan hacia *inactiva*.
- 7 de los 10 bits más importantes según SHAP coinciden con los de la importancia por impureza de la
  fase 2. SHAP añade lo que la impureza no da: el sentido del efecto y su tamaño en cada molécula.

## Cribado virtual por similitud

`screening_ampliado.py` busca las moléculas más parecidas a una consulta (la guanosina) con un pipeline
de agregación y lo compara con un recorrido exhaustivo. Antes de calcular el Tanimoto, dos filtros
**exactos** descartan candidatos: una cota del número de bits a 1 (entre T·Nₐ y Nₐ/T) y un conjunto con
los bits más raros de la consulta, de los que el candidato tiene que compartir al menos uno (para eso
está la colección `mfp_counts`).

```mermaid
flowchart LR
    Q["Consulta<br/>Nₐ bits a 1"] --> F1{"T·Nₐ ≤ mfp.count ≤ Nₐ/T"}
    F1 -- sí --> F2{"¿comparte alguno de los<br/>Nₐ − ⌊T·Nₐ⌋ + 1 bits más raros?"}
    F2 -- sí --> T["Tanimoto =<br/>común / (Nₐ + N_b − común)"]
    T --> R{"≥ T"}
    R -- sí --> OK(["Similar"])
    F1 -- no --> X(["Descartada sin calcular"])
    F2 -- no --> X
```

| Base | T | Candidatas evaluadas | Recuperadas | ¿Igual que la búsqueda exhaustiva? |
| --- | ---: | ---: | ---: | :---: |
| CDS16 | 0,8 | 9 de 82 | 1 | ✅ |
| CDS16 | 0,3 | 63 de 82 | 8 | ✅ |
| CDS29 | 0,8 | 407 de 7015 (5,8 %) | 0 | ✅ |
| CDS29 | 0,3 | 5029 de 7015 (72 %) | 2 | ✅ |

El pipeline devuelve exactamente las mismas moléculas que la búsqueda exhaustiva en los ocho casos (dos
bases × cuatro umbrales); con T = 0,8 en CDS29 evalúa solo el 5,8 % de la colección, y a T = 0,3 los
filtros dejan pasar el 72 % y la ventaja desaparece.

![Cribado: moléculas evaluadas y aceleración frente a la búsqueda exhaustiva](src/fig_screening.png)

## Estructura

```
src/
  documentos.py                           Esquema tipado: el TypedDict de cada colección y sus
                                          Collection[...] (dept, emp, molecules, mfp_counts)
  fase1_emp_dept.py                       Fase 1: colecciones, validación, carga e informes
  fase2_clasificacion.py                  Fase 2: fingerprints, CV, Optuna, test y figuras
  fase2_ruido_outliers.py                 Fase 2: CV de combinaciones ruido × outliers
  fase2_seleccion.py                      Fase 2: selección e ingeniería de características
  fase2_shap.py                           Fase 2: explicabilidad con SHAP
  screening_ampliado.py                   Ampliación: cribado con varios umbrales
  figuras_screening.py                    Figura del cribado (a partir de screening_resultados.json)
  screening_Mol-database-only_mongo_p.py  Ejemplo de la asignatura (sin cambios)
  *_salida.txt, *_resultados.json         Salidas de cada script
  fig_*.png / fig_*.html                  Figuras (estáticas e interactivas)
  build_doc.py, memoria.html              Generadores de la primera versión de la memoria
                                          (desfasados; la memoria vigente está en entregables/)
tests/                                    Tests con pytest (unitarios y de integración con MongoDB)
instrucciones/                            Enunciado, teoría, Readme y copia de seguridad (dump.tar)
entregables/                              Memoria (PDF y DOCX) y zip de la entrega
pyproject.toml, uv.lock, .python-version  Proyecto de uv: dependencias, versiones fijadas y Python
```

Las consultas están tipadas: cada colección se usa como un `Collection[...]` de PyMongo con el `TypedDict` de su
documento (`src/documentos.py`): `Dept` y `Emp` con los campos que exigen sus validadores `$jsonSchema`, y
`Molecula` y `FrecuenciaBit` con los del volcado. `ty` comprueba así los documentos que se cargan y los campos que
se leen de cada consulta; las consultas son las mismas que se entregaron.

## Cómo reproducirlo (en nativo)

Todo se ejecuta directamente en la máquina. Hace falta:

- [uv](https://docs.astral.sh/uv/), que instala Python 3.14 y las dependencias;
- MongoDB Community Server, `mongosh` y las MongoDB Database Tools (`mongorestore`);
- Chrome o Chromium, que Kaleido usa para exportar las figuras a PNG.

```bash
uv sync                                   # crea .venv con las versiones de uv.lock

# Servidor MongoDB local con el usuario de la asignatura
mkdir -p ~/mongodata
mongod --dbpath ~/mongodata --port 27017 --bind_ip 127.0.0.1 --fork --logpath ~/mongodata/mongod.log
mongosh --port 27017 --eval 'db.getSiblingDB("admin").createUser({user: "adp", pwd: "adp", roles: ["root"]})'

# Bases de datos moleculares (CDS16 y CDS29) desde la copia de seguridad
tar xf instrucciones/dump.tar
mongorestore "mongodb://adp:adp@localhost:27017/?authSource=admin" dump

cd src
uv run python fase1_emp_dept.py        > fase1_salida.txt
uv run python fase2_clasificacion.py   > fase2_salida.txt      # CV + Optuna (~30 min)
uv run python fase2_ruido_outliers.py  > ruido_outliers_salida.txt   # ~6 min
uv run python fase2_seleccion.py       > seleccion_salida.txt  # ~12 min
uv run python fase2_shap.py            > shap_salida.txt       # ~1 min
uv run python screening_ampliado.py    > screening_salida.txt
uv run python figuras_screening.py
```

`fase2_ruido_outliers.py`, `fase2_seleccion.py` y `fase2_shap.py` leen los hiperparámetros de
`fase2_resultados.json`, de modo que no repiten la búsqueda de Optuna.

> [!WARNING]
> Los scripts usan por defecto `mongodb://adp:adp@127.0.0.1:27017/?authSource=admin`, el usuario de
> prácticas de la asignatura, pensado solo para un servidor local. Para cualquier otro servidor pasa la
> cadena de conexión en la variable de entorno `MONGO_URI` (vale para todos los scripts).

## Tests

```bash
uv run pytest                 # 77 tests; los de integración se saltan si no hay MongoDB
uv run pytest -m "not mongo"  # solo los 51 unitarios (unos 3 s)
uv run pytest --cov           # con cobertura
```

| Fichero | Qué comprueba |
| --- | --- |
| `test_fase1.py` | datos y esquemas de la fase 1, rechazo de documentos inválidos, informes Q1-Q11 y que `main()` reproduce **byte a byte** `fase1_salida.txt` |
| `test_clasificacion.py` | GMean y métricas, los siete modelos, los espacios de Optuna, la partición, la CV y que el Random Forest de Optuna reproduce la GMean y el AUROC de test de `fase2_resultados.json` |
| `test_ruido_outliers.py` | los tres ruidos, la distancia de Jaccard y los detectores de outliers por clase |
| `test_seleccion.py` | Tanimoto, variables de similitud, cuantiles, filtros, rankings, espacios y que RDKit reproduce `mfp.bits` |
| `test_shap.py` | la disposición de los puntos del enjambre |
| `test_figuras.py` | las figuras de la fase 2, reconstruidas desde los JSON entregados |
| `test_screening.py` | que la agregación coincide con la búsqueda exhaustiva y que `main()` reproduce `screening_resultados.json` |

> [!NOTE]
> Los tests de integración (marcados `mongo`) **no necesitan `mongorestore`**: restauran CDS16 y CDS29
> desde `instrucciones/dump.tar` leyendo el BSON con PyMongo, en bases de datos temporales que se borran
> al terminar. Usan `mongodb://localhost:27017` o la cadena de `MOLECULA_TEST_URI`.

## Calidad del código e integración continua

El formato y el lint se comprueban con [Ruff](https://docs.astral.sh/ruff/) y los tipos con
[ty](https://docs.astral.sh/ty/), instalados por `uv sync` como dependencias de desarrollo y
configurados en `pyproject.toml`:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
```

Se excluyen `instrucciones/` y la copia de `screening_Mol-database-only_mongo_p.py`, que se conservan tal
como los facilitó la asignatura.

Cada *push* y *pull request* pasa por [GitHub Actions](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/actions):

```mermaid
flowchart LR
    P(["push / PR"]) --> L["Lint, formato y tipos<br/>Ruff · ty"]
    P --> T["Tests y cobertura<br/>pytest + MongoDB 8<br/>(CDS16 y CDS29 del dump)"]
    P --> C["CodeQL<br/>Python y workflows"]
    L & T --> I["Insignia de cobertura<br/>(solo en main)"]
```

Además: Dependabot (uv y GitHub Actions) y escaneo de secretos con protección al hacer *push*.

## Documentación

La [wiki](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki) amplía cada parte:

| Página | Contenido |
| --- | --- |
| [Fase 1: de E-R a MongoDB](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki/Fase-1-de-E-R-a-MongoDB) | referencias frente a incrustación, validación e informes |
| [Fase 2: clasificación](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki/Fase-2-Clasificaci%C3%B3n) | datos, modelos, métricas, Optuna y test |
| [Ruido y outliers](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki/Ruido-y-outliers) | las doce combinaciones y sus fronteras de decisión |
| [Selección e ingeniería de características](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki/Selecci%C3%B3n-e-ingenier%C3%ADa-de-caracter%C3%ADsticas) | cinco criterios de selección y seis representaciones con RDKit |
| [Explicabilidad con SHAP](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki/Explicabilidad-con-SHAP) | qué bits empujan hacia activa o inactiva |
| [Cribado por similitud](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki/Cribado-por-similitud) | los filtros exactos y los resultados |
| [Reproducción y pruebas](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki/Reproducci%C3%B3n-y-pruebas) | entorno, tiempos, tests, cobertura y CI |
| [Preguntas frecuentes](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki/Preguntas-frecuentes) | dudas habituales |

## Contribuir

- Abre un *issue* con la plantilla adecuada: [error](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/issues/new?template=bug_report.yml),
  [mejora](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/issues/new?template=feature_request.yml) o
  [resultado que no se reproduce](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/issues/new?template=resultado_distinto.yml).
- Los *pull requests* siguen la [plantilla](.github/pull_request_template.md) y tienen que pasar la CI.
- Toda la participación se rige por el [código de conducta](CODE_OF_CONDUCT.md); los problemas de
  seguridad se informan en privado ([SECURITY.md](SECURITY.md)).

## Cómo citar

Usa los metadatos de [CITATION.cff](CITATION.cff) (*Cite this repository* en GitHub).

<details>
<summary>BibTeX</summary>

```bibtex
@software{caballero_maria_adp_2026,
  author  = {Caballero María, Pablo},
  title   = {Prácticas ADP (Apartado I): MongoDB y predicción de la actividad biológica},
  year    = {2026},
  version = {1.1.0},
  url     = {https://github.com/Pablo-Caballero-Maria/trabajo_molecula},
  license = {MIT}
}
```

</details>

## Licencia

El código del autor está bajo la [licencia MIT](LICENSE). El contenido de `instrucciones/` (enunciado,
teoría, la copia de seguridad `dump.tar` y el ejemplo `screening_Mol-database-only_mongo_p.py`, del que
hay una copia sin cambios en `src/`) es material de la asignatura y pertenece a sus autores; no está cubierto por esa licencia.
