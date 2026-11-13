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
   del fingerprint molecular de 1024 bits de las bases de datos CDS16 y CDS29:
   - siete clasificadores (regresión logística, SVM, k-NN con Jaccard, Naive Bayes de Bernoulli,
     Random Forest, Extra Trees y una red neuronal MLP) evaluados con validación cruzada
     estratificada repetida;
   - ajuste de hiperparámetros con **Optuna** (TPE + poda por mediana) sobre una partición de
     entrenamiento de CDS29, con evaluación final en test;
   - **ruido × outliers**: validación cruzada de 12 combinaciones de inyección de ruido y
     eliminación de outliers, con la frontera de decisión de cada una sobre dos componentes
     principales;
   - **selección e ingeniería de características**: cinco criterios de selección de bits
     (filtrado, embebidos y envolvente) frente a seis representaciones moleculares calculadas
     con RDKit (cuentas, descriptores fisicoquímicos, claves MACCS y variables de similitud);
   - **explicabilidad con SHAP**: contribución de cada bit del fingerprint a la probabilidad de
     ser activa (gráfico de enjambre o *beeswarm*).
3. **Ampliación · cribado virtual por similitud.** Búsqueda por coeficiente de Tanimoto con
   el pipeline de agregación de la asignatura, comparada con una búsqueda exhaustiva.

Todas las figuras se generan con **Plotly** (PNG para la memoria y HTML interactivo).

## Ruido × outliers en la validación cruzada

`fase2_ruido_outliers.py` toma el Random Forest con los hiperparámetros que encontró Optuna
(leídos de `fase2_resultados.json`, sin repetir la búsqueda) y lo evalúa con validación
cruzada estratificada de 5 particiones sobre la partición de entrenamiento de CDS29 (5612
moléculas). En cada partición, y solo sobre su parte de entrenamiento:

1. **Eliminación de outliers** (5 % de cada clase, detectados por separado en activas e
   inactivas para no vaciar la clase minoritaria): ninguna, *Isolation Forest* o *Local
   Outlier Factor* con distancia de Jaccard (1 − Tanimoto).
2. **Inyección de ruido** como aumento de datos (se añade una copia ruidosa del entrenamiento):
   ninguno, gaussiano (σ = 0,15), volteo de bits (cada bit se invierte con p = 0,005) o
   enmascarado (cada bit a 1 se apaga con p = 0,10).

La figura dibuja una celda por combinación (filas: ruido; columnas: outliers). El fondo es
P(activa) en el plano de las dos primeras componentes principales (azul = inactiva, rojo =
activa, línea gris = frontera 0,5) y encima están las moléculas de validación del primer fold
con su **clase real**: un punto rojo sobre fondo rojo, o azul sobre azul, está bien
clasificado. La PCA se ajusta con el entrenamiento original del fold, así que todas las celdas
comparten ejes. Como un modelo de 1024 dimensiones no tiene una frontera dibujable en 2D, el
fondo es el mismo Random Forest reentrenado con los mismos datos proyectados en esas dos
componentes: es su frontera exacta, no una aproximación. Cada celda indica la GMean del modelo
real (1024 bits, media ± desviación de las 5 particiones) y la de ese modelo 2D.

![Fronteras de decisión ruido × outliers](src/fig_ruido_outliers.png)

Resultados (detalle en `src/ruido_outliers_salida.txt`):

- Ninguna combinación cambia el resultado de forma clara: la GMean en validación cruzada va de
  0,610 a 0,630, con desviaciones de ±0,01-0,02. La mejor es **volteo de bits sin eliminar
  outliers** (0,630 frente a 0,625 del caso base).
- Eliminar outliers no ayuda: con *Isolation Forest* y sin ruido la GMean baja a 0,610. En un
  conjunto químicamente diverso, las moléculas raras no son errores, sino parte del espacio
  que el modelo tiene que cubrir.
- En el test reservado, el caso base reproduce exactamente el Random Forest de la fase 2
  (GMean 0,632, AUROC 0,717) y el volteo de bits lo mejora a GMean 0,660 y AUROC 0,727. La
  mejora es mayor que la observada en validación cruzada, así que parte puede deberse al azar
  de un único test.
- Las dos primeras componentes explican solo el 7,2 % de la varianza: en ese plano las clases
  se solapan y la GMean del modelo 2D se queda en torno a 0,52. La información que separa
  activas de inactivas está repartida entre muchos bits, como confirma SHAP.

## Selección e ingeniería de características

`fase2_seleccion.py` responde a dos preguntas con el mismo protocolo: validación cruzada
estratificada de 5 particiones sobre las 5612 moléculas de entrenamiento de CDS29, con el
Random Forest y la regresión logística que ajustó Optuna, y el test reservado solo para la
medición final. Todo el preprocesado —rankings, umbrales, transformación por cuantiles y
variables de similitud— se ajusta **dentro** de cada partición de entrenamiento, de modo que
la validación nunca ve información derivada de las moléculas que evalúa.

**Selección**: cinco criterios de tres familias, evaluados conservando 16, 32, 64, 128, 256 y
512 de los 1024 bits. Filtrado univariante (chi cuadrado e información mutua), selección
embebida (importancia por impureza del Random Forest y magnitud de los coeficientes de una
logística L1) y selección envolvente (eliminación recursiva de características, RFE).

![Selección de características](src/fig_seleccion_cds29.png)

**Ingeniería**: representaciones recalculadas con RDKit a partir del SMILES de cada molécula.
Se comprueba que el fingerprint de Morgan de radio 2 y 1024 bits reproduce exactamente
`mfp.bits` de MongoDB, así que las cuentas de subestructuras son la misma representación sin
binarizar. A ellas se suman 217 descriptores fisicoquímicos (transformados por cuantiles),
las 167 claves MACCS y 5 variables de similitud (máximo y media de los 5 vecinos más
similares de cada clase, y su diferencia), solas y combinadas con los bits.

![Espacios de representación](src/fig_ingenieria_cds29.png)

Resultados (detalle en `src/seleccion_salida.txt`):

- **El fingerprint se puede reducir a una cuarta parte sin coste medible**: con 256 bits el
  mejor criterio llega a 0,627 en el Random Forest (referencia con 1024 bits: 0,625) y a
  0,632 en la logística (referencia: 0,632). Por debajo de 64 bits la GMean cae con claridad,
  hasta 0,535-0,590 con 16 bits.
- **Ningún criterio domina.** El RFE, con diferencia el más caro, es el mejor con 16 bits
  (0,584 y 0,590) y comparte el primer puesto en los tamaños grandes, pero chi cuadrado y la
  información mutua se quedan a pocas milésimas —y a veces por delante— con una fracción del
  coste. La logística L1 es la peor con 16 bits (0,535) y de las mejores a partir de 128.
- **Los bits elegidos son estables y coinciden con SHAP**: FP111 está entre los 16 primeros
  de los cinco criterios en las cinco particiones; FP98 y FP379 lo están en cuatro criterios,
  y FP450, FP695 y FP831 en tres. Son los mismos bits que encabezan el análisis SHAP.
- **La única representación que mejora a los bits es la similitud.** `Bits + similitud` gana
  en validación cruzada con los dos modelos (RF 0,635; logística 0,640) y en el test lleva a
  la logística hasta **GMean 0,669 y AUROC 0,723**, el mejor resultado de todo el trabajo (el
  SVM de la fase 2 se quedaba en 0,665). Las cinco variables codifican el principio de
  similitud de Johnson y Maggiora, lo que el k-NN hacía mal como clasificador y aquí funciona
  como información añadida.
- **Los descriptores fisicoquímicos no aportan y al Random Forest lo estorban**: solos dan
  0,590 (RF) y 0,604 (logística), y unidos a los bits hunden al Random Forest hasta 0,586,
  porque 217 variables continuas compiten con los bits en cada división del árbol. Las claves
  MACCS, con solo 167 bits, llegan a 0,613: casi lo mismo que los 1024 de Morgan.
- Para el Random Forest, la mejora de `Bits + similitud` no se confirma en el test (0,625
  frente a 0,632 del caso base), un recordatorio de que un único test de 1403 moléculas tiene
  su propio azar.

## Explicabilidad con SHAP

`fase2_shap.py` reentrena una vez el Random Forest de Optuna (misma partición y semilla, de
modo que es el mismo bosque evaluado en test) y calcula con `shap.TreeExplainer`, exacto para
bosques, los valores SHAP de las 1403 moléculas de test. Para clasificación binaria con muchas
variables, el gráfico adecuado es el de enjambre (*beeswarm*): ordena los bits por su impacto
medio |SHAP| y muestra cada molécula como un punto cuyo eje x es cuánto desplaza P(activa) y
cuyo color es el valor del bit (presente o ausente).

![SHAP beeswarm](src/fig_shap_cds29.png)

- El valor base es 0,500 (el bosque pondera las clases) y la suma de las contribuciones
  reproduce la probabilidad predicha con un error máximo de 1e-9.
- FP98, FP111, FP363 y FP185 empujan hacia *activa* cuando están presentes: entre las
  moléculas de entrenamiento con FP98 o FP111 activo hay un 45-48 % de activas, frente al
  19,6 % global. FP379, FP695, FP722 y, sobre todo, FP831 empujan hacia *inactiva*.
- 7 de los 10 bits más importantes según SHAP coinciden con los de la importancia por impureza
  de la fase 2. SHAP añade lo que la impureza no da: el sentido del efecto y su tamaño en cada
  molécula.

## Estructura

```
src/
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
instrucciones/                            Enunciado, teoría, Readme y copia de seguridad (dump.tar)
entregables/                              Memoria (PDF y DOCX) y zip de la entrega
pyproject.toml, uv.lock, .python-version  Proyecto de uv: dependencias, versiones fijadas y Python
```

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
`fase2_resultados.json`, de modo que no repiten la búsqueda de Optuna. Los scripts usan por
defecto `mongodb://adp:adp@127.0.0.1:27017/?authSource=admin`; los de la fase 2 admiten otra
cadena de conexión en la variable de entorno `MONGO_URI`.

## Calidad del código

El formato y el lint se comprueban con [ruff](https://docs.astral.sh/ruff/) y los tipos con
[ty](https://docs.astral.sh/ty/), ambos instalados por `uv sync` como dependencias de
desarrollo y configurados en `pyproject.toml`:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
```

Se excluyen `instrucciones/` y la copia de `screening_Mol-database-only_mongo_p.py`, que se
conservan tal como los facilitó la asignatura.
