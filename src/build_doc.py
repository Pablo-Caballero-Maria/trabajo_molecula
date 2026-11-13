#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera memoria.html (que Drive convierte a Google Doc) a partir de los
resultados de fase1/fase2/screening. Autor: Pablo Caballero Maria."""
import json, html

R2 = json.load(open('fase2_resultados.json'))
SCR = json.load(open('screening_resultados.json'))

P = []
def w(s): P.append(s)
def h(t, n=2): w('<h%d>%s</h%d>' % (n, t, n))
def p(t): w('<p>%s</p>' % t)
def ul(items): w('<ul>' + ''.join('<li>%s</li>' % i for i in items) + '</ul>')
def ol(items): w('<ol>' + ''.join('<li>%s</li>' % i for i in items) + '</ol>')
def pre(t): w('<p><span style="font-family:\'Courier New\'">%s</span></p>' %
              html.escape(t).replace('\n', '<br>').replace(' ', '&nbsp;'))
def code(t): return '<span style="font-family:\'Courier New\'">%s</span>' % html.escape(t)

def table(headers, rows, caption=None):
    w('<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse">')
    w('<tr>' + ''.join('<td><b>%s</b></td>' % c for c in headers) + '</tr>')
    for r in rows:
        w('<tr>' + ''.join('<td>%s</td>' % c for c in r) + '</tr>')
    w('</table>')
    if caption:
        w('<p><i>%s</i></p>' % caption)

MET = ['Accuracy', 'BalAcc', 'Sensitivity', 'Specificity', 'GMean', 'Kappa', 'MCC', 'F1', 'AUROC']

def tabla_cv(db):
    d = R2[db]['resultados']
    rows = []
    for nombre, r in d.items():
        m, s = r['media'], r['std']
        rows.append([nombre] + ['%.3f ± %.3f' % (m[k], s[k]) for k in MET] +
                    ['%.1f' % r['tiempo_s']])
    return ['Modelo'] + MET + ['t (s)'], rows

def mejor(db, metrica='GMean'):
    d = R2[db]['resultados']
    return max(d.items(), key=lambda kv: kv[1]['media'][metrica])


# ===========================================================================
# PORTADA
# ===========================================================================
w('<p style="text-align:center"><b>Máster universitario en Inteligencia Computacional '
  'e Internet de las Cosas</b></p>')
w('<p style="text-align:center">Análisis, Diseño y Procesamiento de Datos Aplicados a las '
  'Ciencias y a las Tecnologías (ADP)</p>')
w('<p>&nbsp;</p><p>&nbsp;</p>')
w('<h1 style="text-align:center">Prácticas (Apartado I)</h1>')
w('<p style="text-align:center"><b>Fase 1: solución NoSQL MongoDB<br>'
  'Fase 2: predicción de la actividad biológica de compuestos</b></p>')
w('<p>&nbsp;</p><p>&nbsp;</p><p>&nbsp;</p>')
w('<p style="text-align:center">Pablo Caballero María</p>')
w('<p style="text-align:center">10 de noviembre de 2026</p>')
w('<p>&nbsp;</p>')
w('<div style="page-break-after:always"></div>')

# ===========================================================================
h('1. Introducción y objetivos', 1)
p('Esta memoria recoge las dos fases del primer apartado de prácticas de la asignatura. '
  'La primera traslada un esquema entidad-interrelación clásico (departamentos y empleados, '
  'con una interrelación reflexiva de jefatura) a un modelo de documentos de MongoDB, lo puebla '
  'con datos de prueba y lo interroga mediante el framework de agregación. La segunda parte del '
  'ejemplo práctico de quimioinformática desarrollado en teoría: a partir de los fingerprints '
  'moleculares almacenados en las bases de datos CDS16 y CDS29 se construyen y evalúan modelos '
  'de clasificación que predicen si un compuesto es activo o inactivo.')
p('Como ampliación se reproduce y extiende el ejemplo de cribado virtual por similitud '
  '(%s), que es el que motiva el diseño de las colecciones moleculares, '
  'y se mide cuánto aporta realmente el filtrado previo basado en cotas de bits.'
  % code('screening_Mol-database-only_mongo_p.py'))

h('2. Entorno de trabajo', 1)
p('Todo el trabajo se ha realizado sobre un servidor MongoDB local, equivalente al contenedor '
  'de Portainer descrito en el fichero %s de la asignatura, evitando así la dependencia de un '
  'clúster de MongoDB Atlas.' % code('Readme.txt'))
ul(['Servidor: MongoDB 8.3.7 (standalone, autenticación activada, usuario %s).'
    % code('adp/adp'),
    'Cliente y utilidades: %s y %s (MongoDB Database Tools 100.13.0).'
    % (code('mongosh/pymongo 4.x'), code('mongorestore')),
    'Python 3.13 con %s, %s, %s y %s.'
    % (code('pymongo'), code('numpy/pandas'), code('scikit-learn'), code('matplotlib')),
    'Cadena de conexión empleada en todos los scripts: %s.'
    % code('mongodb://adp:adp@127.0.0.1:27017/?authSource=admin')])
p('La restauración de la copia de seguridad molecular sigue las indicaciones del fichero '
  '%s y de la última diapositiva de la presentación de la asignatura: la sentencia se ejecuta '
  'en el directorio de nivel superior a %s, de forma que se recuperan todas las bases de datos '
  'contenidas en él.' % (code('Readme.txt'), code('./dump')))
pre('# Descarga de la copia de seguridad\n'
    'mkdir /home/mongodata && cd /home/mongodata\n'
    'wget --no-check-certificate \'https://docs.google.com/uc?export=download&id=...\' -O dump.tar\n'
    'tar -xf dump.tar\n\n'
    '# Restauración sobre el servidor local\n'
    'mongorestore "mongodb://adp:adp@localhost:27017/?authSource=admin&readPreference=primary'
    '&ssl=false&directConnection=true" dump')
p('La restauración recupera las dos bases de datos moleculares, con las colecciones '
  '%s y %s en cada una, además de los índices sobre %s y %s que el propio dump incluye.'
  % (code('molecules'), code('mfp_counts'), code('mfp.bits'), code('mfp.count')))

# ===========================================================================
h('3. Fase 1: solución NoSQL MongoDB', 1)

h('3.1. Del esquema entidad-interrelación al modelo de documentos')
p('El esquema de partida tiene dos entidades, DEPT y EMP, y dos interrelaciones. La primera '
  'asocia cada departamento con sus empleados con cardinalidades (0,1) por el lado de DEPT y '
  '(0,N) por el lado de EMP: un empleado pertenece como mucho a un departamento y un '
  'departamento puede no tener ninguno. La segunda es reflexiva sobre EMP, «Es jefe / '
  'Subordinado de», también (0,1):(0,N): cada empleado tiene a lo sumo un jefe y un jefe puede '
  'tener varios subordinados.')
p('MongoDB admite dos traducciones para una relación 1:N. La primera es <b>embeber</b> los '
  'empleados dentro del documento de su departamento; la segunda es <b>referenciar</b>, con dos '
  'colecciones y un campo que guarda la clave del documento relacionado. Aquí se ha optado por '
  'referenciar, por tres motivos:')
ol(['Los empleados son entidades con identidad propia (EMPNO es clave, los informes los '
    'consultan directamente y la mayoría de las consultas del enunciado parten de ellos), no '
    'detalles subordinados que sólo tengan sentido dentro de un departamento.',
    'La interrelación reflexiva de jefatura cruza los límites del departamento: BLAKE dirige a '
    'empleados del departamento 30, pero su propio jefe, KING, está en el 10. Con empleados '
    'embebidos esa relación no se podría resolver dentro de un único documento.',
    'Las cardinalidades mínimas 0 exigen representar empleados sin departamento y departamentos '
    'sin empleados; con referencias basta con omitir el campo %s, mientras que el modelo '
    'embebido obligaría a duplicar o a inventar un departamento ficticio.' % code('deptno')])
p('El modelo resultante es, por tanto, cercano al relacional, pero aprovechando dos rasgos '
  'propios del documento: el esquema flexible (un atributo opcional como %s simplemente no '
  'aparece en el documento, en lugar de almacenarse como NULL) y la validación declarativa '
  'mediante %s.' % (code('comm'), code('$jsonSchema')))

h('3.2. Estructura de las colecciones')
p('Se crean dos colecciones en la base de datos %s. El identificador propio de cada entidad '
  '(DEPTNO y EMPNO) se usa como %s, de modo que MongoDB garantiza su unicidad con el índice '
  'que crea por defecto y no se desperdicia espacio en un ObjectId adicional.'
  % (code('empresa'), code('_id')))
table(['Colección', 'Campo', 'Tipo', 'Obligatorio', 'Descripción'],
      [['dept', code('_id'), 'int', 'Sí', 'DEPTNO, clave primaria'],
       ['dept', code('dname'), 'string (≤14)', 'Sí', 'Nombre del departamento'],
       ['dept', code('loc'), 'string (≤13)', 'Sí', 'Localidad'],
       ['emp', code('_id'), 'int', 'Sí', 'EMPNO, clave primaria'],
       ['emp', code('ename'), 'string (≤10)', 'Sí', 'Nombre del empleado'],
       ['emp', code('job'), 'string (≤9)', 'Sí', 'Puesto'],
       ['emp', code('hiredate'), 'date', 'Sí', 'Fecha de contratación'],
       ['emp', code('sal'), 'double ≥ 0', 'Sí', 'Salario'],
       ['emp', code('comm'), 'double ≥ 0', 'No', 'Comisión; ausente si no la percibe'],
       ['emp', code('deptno'), 'int', 'No', 'Referencia a %s; ausente si (0,1) se cumple con 0'
        % code('dept._id')],
       ['emp', code('mgr'), 'int', 'No', 'Referencia al %s del jefe (interrelación reflexiva)'
        % code('emp._id')]],
      'Tabla 1. Estructura de los documentos de las colecciones dept y emp.')
p('La validación se declara en la creación de la colección con %s y nivel %s, de manera que '
  'MongoDB rechaza cualquier documento que incumpla tipos, obligatoriedad o rangos. El '
  'parámetro %s impide además que se cuelen campos no previstos por una errata.'
  % (code('$jsonSchema'), code('strict'), code('additionalProperties: false')))
pre("""db.create_collection('emp', validator={'$jsonSchema': {
    'bsonType': 'object',
    'required': ['_id', 'ename', 'job', 'hiredate', 'sal'],
    'additionalProperties': False,
    'properties': {
        '_id':      {'bsonType': 'int'},
        'ename':    {'bsonType': 'string', 'maxLength': 10},
        'job':      {'bsonType': 'string', 'maxLength': 9},
        'hiredate': {'bsonType': 'date'},
        'sal':      {'bsonType': ['double', 'int'], 'minimum': 0},
        'comm':     {'bsonType': ['double', 'int'], 'minimum': 0},
        'deptno':   {'bsonType': 'int'},
        'mgr':      {'bsonType': 'int'}}}},
    validationLevel='strict')""")
p('Sobre %s se crean además índices secundarios en %s, %s y %s: los dos primeros son los campos '
  'por los que se resuelven las reuniones (%s y %s) y el tercero aparece en los filtros de '
  'varios informes. La comprobación del plan de ejecución al final del script confirma que la '
  'búsqueda por departamento usa el índice (%s sobre %s) en lugar de recorrer la colección.'
  % (code('emp'), code('deptno'), code('mgr'), code('job'), code('$lookup'),
     code('$graphLookup'), code('IXSCAN'), code('deptno_1')))

# --- utilidades para incrustar las salidas reales de los scripts -----------
F1 = open('fase1_salida.txt').read()

def bloque_q(clave):
    """Devuelve la salida impresa por el script fase1 para la consulta indicada."""
    i = F1.index(clave)
    j = F1.index('=' * 78, i)          # fin de la cabecera de la consulta
    k = F1.find('=' * 78, j + 80)      # inicio de la cabecera siguiente
    txt = F1[j + 79: k if k != -1 else len(F1)]
    return txt.strip('\n')

h('3.3. Carga de información')
p('El script %s crea la base de datos desde cero, carga los datos y ejecuta los informes. Los '
  'datos de prueba amplían el juego clásico de catorce empleados y cuatro departamentos con '
  'cuatro empleados y un departamento adicionales, elegidos para que el conjunto cubra todos '
  'los casos que el esquema permite:' % code('fase1_emp_dept.py'))
ul(['18 empleados y 5 departamentos, con una jerarquía de cuatro niveles (PRESIDENT → MANAGER '
    '→ ANALYST → CLERK).',
    'Un departamento sin ningún empleado (OPERATIONS) y un empleado sin departamento (LOPEZ): '
    'las cardinalidades mínimas 0 de la primera interrelación.',
    'Un empleado sin jefe (KING, el presidente): la cardinalidad mínima 0 de la interrelación '
    'reflexiva.',
    'Empleados con y sin comisión, incluida una comisión de valor 0 (TURNER), que es distinta '
    'de la ausencia del campo.',
    'Contrataciones repartidas entre 1980 y 2025, para que las agrupaciones por fecha tengan '
    'contenido.'])
p('Tras la carga, el script comprueba la validación insertando a propósito un empleado con '
  'salario negativo; MongoDB lo rechaza con un error de validación que identifica la regla '
  'incumplida. A continuación verifica de forma explícita la integridad referencial, porque '
  'MongoDB no impone claves ajenas: ningún %s apunta a un departamento inexistente y ningún '
  '%s apunta a un empleado inexistente.' % (code('deptno'), code('mgr')))
pre(F1[:F1.index('=' * 78)].strip())

h('3.4. Consultas e informes')
p('Los informes se resuelven con el framework de agregación. Las reuniones entre colecciones '
  'se hacen con %s, y la jerarquía de mando, que es recursiva y de profundidad desconocida, con '
  '%s, el operador específico de MongoDB para recorrer grafos.'
  % (code('$lookup'), code('$graphLookup')))

h('Q1. Empleados con su departamento', 3)
p('Reunión de %s con %s por %s. El %s con %s conserva en el informe a los empleados sin '
  'departamento, que de otro modo desaparecerían.'
  % (code('emp'), code('dept'), code('deptno'), code('$unwind'),
     code('preserveNullAndEmptyArrays: true')))
pre(bloque_q('Q1. Empleados con su departamento'))

h('Q2. Resumen salarial por departamento', 3)
p('La reunión se plantea ahora en sentido inverso, de %s a %s, para que los departamentos sin '
  'empleados aparezcan con cero empleados en lugar de omitirse. Es el equivalente documental de '
  'un LEFT JOIN con agregación.' % (code('dept'), code('emp')))
pre(bloque_q('Q2. Resumen salarial por departamento'))

h('Q3 y Q4. La interrelación reflexiva', 3)
p('La jefatura se recorre uniendo la colección %s consigo misma: en Q3 desde el subordinado '
  'hacia su jefe y en Q4 agrupando por %s para contar los subordinados directos de cada uno.'
  % (code('emp'), code('mgr')))
pre(bloque_q('Q3. Empleado y jefe directo'))
pre(bloque_q('Q4. Jefes y número de subordinados'))

h('Q5 y Q6. Jerarquía completa con $graphLookup', 3)
p('Un %s sólo salva un nivel. Para obtener todo el equipo que depende de JONES, a cualquier '
  'profundidad, se usa %s: parte del %s de JONES, enlaza %s con %s de forma repetida y devuelve '
  'en %s el número de saltos, que es el nivel jerárquico.'
  % (code('$lookup'), code('$graphLookup'), code('_id'), code('connectFromField: _id'),
     code('connectToField: mgr'), code('depthField')))
pre(bloque_q('Q5. Organigrama recursivo'))
p('Invirtiendo los campos de enlace, el mismo operador recorre la cadena de mando en sentido '
  'ascendente, desde un empleado hasta la cima de la organización.')
pre(bloque_q('Q6. Cadena de mando de ADAMS'))

h('Q7 a Q9. Informes de negocio', 3)
p('Q7 compara cada empleado con su jefe dentro de la propia tubería, usando %s en la etapa '
  '%s para referirse a dos campos del mismo documento ya reunido. Q8 calcula la retribución '
  'total de los vendedores sumando salario y comisión, y Q9 agrupa las contrataciones por año '
  'con %s.' % (code('$expr'), code('$match'), code('$year')))
pre(bloque_q('Q7. Empleados cuyo salario'))
pre(bloque_q('Q8. Retribución total'))
pre(bloque_q('Q9. Contrataciones por año'))

h('Q10 y Q11. Casos límite y actualización', 3)
p('Q10 recupera explícitamente los tres casos que justifican las cardinalidades mínimas 0 del '
  'esquema, y muestra la diferencia entre un campo ausente y un campo con valor: TURNER sí '
  'tiene comisión, aunque valga 0, mientras que LOPEZ no tiene departamento en absoluto. Q11 '
  'ejecuta una actualización masiva con una tubería de agregación como argumento de %s, de modo '
  'que el nuevo salario se calcula a partir del anterior en el propio servidor.'
  % code('updateMany'))
pre(bloque_q('Q10. Casos límite'))
pre(bloque_q('Q11. updateMany'))

h('3.5. Valoración de la fase 1')
p('El modelo documental resuelve sin dificultad las consultas que en el modelo relacional se '
  'harían con reuniones, y el caso recursivo (el organigrama completo) resulta incluso más '
  'directo con %s que con una consulta SQL recursiva. A cambio, MongoDB no impone la '
  'integridad referencial: nada impide escribir un %s que no exista, de modo que esa '
  'comprobación se ha tenido que programar en la aplicación. La validación por esquema cubre '
  'los tipos y los rangos, pero no las referencias.'
  % (code('$graphLookup'), code('deptno')))

# ===========================================================================
h('4. Fase 2: predicción de la actividad biológica', 1)

h('4.1. Los datos')
i16, i29 = R2['CDS16']['info'], R2['CDS29']['info']
p('Las dos bases de datos restauradas comparten estructura. Cada documento de %s guarda el '
  'identificador del compuesto, su cadena SMILES, la molécula serializada de RDKit, la clase '
  '(%s o %s) y el fingerprint molecular, almacenado como el número de bits a 1 (%s) y la lista '
  'de sus posiciones (%s). La colección auxiliar %s guarda, para cada posición del fingerprint, '
  'en cuántas moléculas aparece; es la que permite el filtrado previo del cribado por '
  'similitud.' % (code('molecules'), code('Active'), code('Inactive'), code('mfp.count'),
                  code('mfp.bits'), code('mfp_counts')))
table(['', 'CDS16', 'CDS29'],
      [['Moléculas', i16['n_molecules'], i29['n_molecules']],
       ['Activas', '%d (%.1f %%)' % (i16['n_active'], 100 * i16['n_active'] / i16['n_molecules']),
        '%d (%.1f %%)' % (i29['n_active'], 100 * i29['n_active'] / i29['n_molecules'])],
       ['Inactivas', i16['n_inactive'], i29['n_inactive']],
       ['Bits del fingerprint', i16['bits_total'], i29['bits_total']],
       ['Bits siempre a 0 (eliminados)', i16['bits_eliminados'], i29['bits_eliminados']],
       ['Variables predictoras finales', i16['bits_utiles'], i29['bits_utiles']],
       ['Bits a 1 por molécula (media)', '%.1f' % i16['bits_por_mol_media'],
        '%.1f' % i29['bits_por_mol_media']],
       ['Bits a 1 por molécula (mín-máx)', '%d - %d' % (i16['bits_por_mol_min'], i16['bits_por_mol_max']),
        '%d - %d' % (i29['bits_por_mol_min'], i29['bits_por_mol_max'])],
       ['Densidad de la matriz', '%.2f %%' % (100 * i16['densidad']),
        '%.2f %%' % (100 * i29['densidad'])]],
      'Tabla 2. Características de los dos conjuntos de datos.')
p('La diferencia entre ambos no es sólo de tamaño. CDS16 está perfectamente balanceado, '
  'mientras que CDS29 tiene una razón de desequilibrio cercana a %.1f:1 a favor de los '
  'compuestos inactivos, que es la situación habitual en cribado virtual y la que obliga a '
  'elegir con cuidado tanto las métricas como el tratamiento de las clases.'
  % (i29['n_inactive'] / i29['n_active']))

h('4.2. Construcción de la matriz de variables predictoras')
p('En la base de datos sólo están almacenadas las posiciones de los bits con valor 1, de modo '
  'que la matriz de variables predictoras hay que reconstruirla. Se recorre la colección '
  'proyectando únicamente los campos necesarios y se marca un 1 en cada posición indicada:')
pre("""cur = client[db_name].molecules.find({}, {'_id': 1, 'class': 1, 'mfp.bits': 1})
X = np.zeros((n_moleculas, 1024), dtype=np.uint8)
for i, doc in enumerate(cur):
    X[i, doc['mfp']['bits']] = 1          # el resto de posiciones queda a 0
y = [1 if doc['class'] == 'Active' else 0 for doc in ...]""")
p('Siguiendo la cuarta recomendación del enunciado se eliminan las columnas cuyos valores son '
  'todos 0, que no aportan información a ningún clasificador: %d columnas en CDS16 y %d en '
  'CDS29. La matriz resultante es muy dispersa (sólo alrededor del %.0f %% y el %.0f %% de las '
  'posiciones valen 1), lo que condiciona la elección de los modelos: los métodos basados en '
  'árboles y los lineales regularizados funcionan bien en este régimen, mientras que las '
  'distancias euclídeas pierden sentido y conviene sustituirlas por una medida de similitud '
  'binaria.' % (i16['bits_eliminados'], i29['bits_eliminados'],
                100 * i16['densidad'], 100 * i29['densidad']))

h('4.3. Modelos y metodología de evaluación')
p('Se han comparado seis clasificadores de scikit-learn, elegidos para cubrir familias '
  'distintas:')
ul(['<b>Regresión logística</b> con regularización L2, como referencia lineal.',
    '<b>SVM</b> con núcleo RBF.',
    '<b>k-NN</b> con k = 5 y distancia de Jaccard, que es el complemento del coeficiente de '
    'Tanimoto; es el clasificador que traslada directamente el principio de similitud de '
    'Johnson y Maggiora: los compuestos similares tienen propiedades similares.',
    '<b>Naive Bayes de Bernoulli</b>, el modelo probabilístico natural para variables binarias.',
    '<b>Random Forest</b> y <b>Extra Trees</b> con 500 árboles, como representantes de los '
    'ensembles.'])
p('Salvo Naive Bayes y k-NN, todos los modelos se han entrenado con %s, que pondera cada clase '
  'de forma inversamente proporcional a su frecuencia. Sin esa corrección, en CDS29 un '
  'clasificador que prediga siempre «inactivo» acertaría el %.1f %% de las veces sin haber '
  'aprendido nada, y es justo lo que tienden a hacer los modelos entrenados sobre datos '
  'desequilibrados.' % (code("class_weight='balanced'"),
                        100 * i29['n_inactive'] / i29['n_molecules']))
p('La evaluación se hace con validación cruzada estratificada repetida: %d particiones y %d '
  'repeticiones en CDS16 y %d particiones y %d repeticiones en CDS29. La estratificación '
  'mantiene la proporción de activos en cada partición y la repetición reduce la dependencia '
  'del azar de un único reparto. Se informa de la media y la desviación típica de cada métrica '
  'sobre el total de particiones.'
  % (R2['CDS16']['cv']['folds'], R2['CDS16']['cv']['repeats'],
     R2['CDS29']['cv']['folds'], R2['CDS29']['cv']['repeats']))

h('4.4. Métricas')
p('Con clases desequilibradas la exactitud es engañosa, de modo que se calculan las métricas '
  'clásicas que recomienda el enunciado junto a las que miden el comportamiento en ambas clases:')
ul(['<b>Accuracy</b>: proporción de aciertos; se incluye por referencia, pero no decide.',
    '<b>Sensibilidad</b> y <b>especificidad</b>: aciertos dentro de los activos y dentro de los '
    'inactivos.',
    '<b>GMean</b>: media geométrica de ambas. Penaliza con dureza que una clase se sacrifique '
    'por la otra, y es la métrica usada para elegir el mejor modelo.',
    '<b>Kappa de Cohen</b>: concordancia corregida por el acuerdo esperado al azar.',
    '<b>MCC</b> y <b>F1</b>: coeficiente de correlación de Matthews y media armónica de '
    'precisión y sensibilidad.',
    '<b>AUROC</b>: área bajo la curva ROC, que evalúa la ordenación de los compuestos por '
    'probabilidad de ser activos, con independencia del umbral de decisión. Es la métrica más '
    'cercana al uso real en cribado virtual, donde lo que se hace es ordenar una quimioteca y '
    'ensayar sólo los primeros compuestos.'])

h('4.5. Resultados de la validación cruzada')
n16, r16 = mejor('CDS16')
hdr, rows = tabla_cv('CDS16')
table(hdr, rows, 'Tabla 3. CDS16: media ± desviación típica sobre %d x %d particiones.'
      % (R2['CDS16']['cv']['folds'], R2['CDS16']['cv']['repeats']))
p('En CDS16 todos los modelos superan con holgura el azar. El mejor según GMean es <b>%s</b> '
  '(GMean = %.3f, Kappa = %.3f, AUROC = %.3f). Las desviaciones típicas son grandes, de entre '
  '%.2f y %.2f en GMean, lo que era previsible: con %d moléculas, cada partición de prueba '
  'tiene sólo unas %d, de modo que un acierto de más o de menos mueve la métrica varios puntos. '
  'Las diferencias entre los primeros clasificadores quedan dentro de ese margen y no conviene '
  'leerlas como una ordenación firme.'
  % (n16, r16['media']['GMean'], r16['media']['Kappa'], r16['media']['AUROC'],
     min(v['std']['GMean'] for v in R2['CDS16']['resultados'].values()),
     max(v['std']['GMean'] for v in R2['CDS16']['resultados'].values()),
     i16['n_molecules'], round(i16['n_molecules'] / R2['CDS16']['cv']['folds'])))

n29, r29 = mejor('CDS29')
hdr, rows = tabla_cv('CDS29')
table(hdr, rows, 'Tabla 4. CDS29: media ± desviación típica sobre %d x %d particiones.'
      % (R2['CDS29']['cv']['folds'], R2['CDS29']['cv']['repeats']))
p('En CDS29 el panorama cambia por completo. El mejor modelo, <b>%s</b>, alcanza GMean = %.3f '
  'y AUROC = %.3f, muy lejos de los valores de CDS16, y los índices de Kappa se mueven en torno '
  'a %.2f, es decir, una concordancia débil. La desviación típica, en cambio, es mucho menor '
  '(alrededor de %.3f en GMean), porque cada partición de prueba contiene unas %d moléculas: '
  'el resultado es estable, y lo que mide es que el problema es genuinamente más difícil.'
  % (n29, r29['media']['GMean'], r29['media']['AUROC'],
     sum(v['media']['Kappa'] for v in R2['CDS29']['resultados'].values()) / 6,
     sum(v['std']['GMean'] for v in R2['CDS29']['resultados'].values()) / 6,
     round(i29['n_molecules'] / R2['CDS29']['cv']['folds'])))
r29r = R2['CDS29']['resultados']
p('Conviene mirar la sensibilidad y la especificidad por separado, porque es ahí donde se ve '
  'lo que cada modelo está haciendo en realidad. Random Forest obtiene la mayor exactitud de '
  'la tabla (%.3f) y, sin embargo, el peor GMean (%.3f): clasifica correctamente el %.1f %% de '
  'los inactivos pero sólo recupera el %.1f %% de los activos. Es el fallo típico de las clases '
  'desequilibradas, y el motivo por el que el enunciado pide métricas como GMean o Kappa en '
  'lugar de la exactitud. La ponderación %s atenúa el efecto pero no lo elimina en los '
  'ensembles de árboles, que siguen inclinándose hacia la clase mayoritaria; los modelos '
  'lineales y el SVM reparten el error de forma mucho más equilibrada (sensibilidad %.3f y '
  'especificidad %.3f en el SVM), y por eso encabezan la clasificación por GMean pese a tener '
  'menos exactitud.'
  % (r29r['Random Forest']['media']['Accuracy'], r29r['Random Forest']['media']['GMean'],
     100 * r29r['Random Forest']['media']['Specificity'],
     100 * r29r['Random Forest']['media']['Sensitivity'],
     code("class_weight='balanced'"),
     r29r['SVM (RBF)']['media']['Sensitivity'], r29r['SVM (RBF)']['media']['Specificity']))
p('La diferencia importa en cribado virtual, donde un falso negativo es un compuesto '
  'prometedor que nunca llega a ensayarse. Nótese además que, medidos por AUROC, los seis '
  'modelos quedan mucho más juntos (entre %.3f y %.3f): todos ordenan los compuestos de forma '
  'parecida, y lo que los separa es dónde sitúan el umbral de decisión.'
  % (min(v['media']['AUROC'] for v in r29r.values()),
     max(v['media']['AUROC'] for v in r29r.values())))

h('4.6. Ajuste de hiperparámetros y evaluación en test independiente')
ho = R2['CDS29_holdout']
p('Para la base de datos grande se ha añadido un paso más riguroso. CDS29 se divide de forma '
  'estratificada en %d moléculas de entrenamiento y %d de test (80/20). Sobre el conjunto de '
  'entrenamiento se buscan los hiperparámetros de tres modelos con %s, optimizando GMean con '
  'validación cruzada interna de 3 particiones; el conjunto de test no interviene en ninguna '
  'decisión y sólo se usa para la medición final.'
  % (ho['n_train'], ho['n_test'], code('GridSearchCV')))
rows = []
for nombre, d in ho['resultados'].items():
    m = d['test']
    tp, fn = d['confusion'][0]
    fp, tn = d['confusion'][1]
    rows.append([nombre, html.escape(str(d['best_params'])), '%.3f' % d['gmean_cv_interno'],
                 '%.3f' % m['GMean'], '%.3f' % m['Sensitivity'], '%.3f' % m['Specificity'],
                 '%.3f' % m['Kappa'], '%.3f' % m['AUROC'],
                 '%d / %d / %d / %d' % (tp, fn, fp, tn)])
table(['Modelo', 'Mejores parámetros', 'GMean (CV int.)', 'GMean', 'Sens.', 'Espec.',
       'Kappa', 'AUROC', 'TP/FN/FP/TN'], rows,
      'Tabla 5. CDS29: resultados sobre el conjunto de test (%d moléculas, %d activas).'
      % (ho['n_test'], ho['test_active']))
mejor_test = max(ho['resultados'].items(), key=lambda kv: kv[1]['test']['GMean'])
p('El mejor resultado en test lo obtiene <b>%s</b>, con GMean = %.3f y AUROC = %.3f. La matriz '
  'de confusión muestra el compromiso al que llega: recupera %d de los %d activos del test '
  'pero marca también %d inactivos como activos. En un cribado real esa es una decisión de '
  'coste: cada falso positivo es un ensayo que no dará resultado, y cada falso negativo, un '
  'compuesto activo descartado. El valor de GMean obtenido en el test es coherente con el de '
  'la validación cruzada interna (%.3f), lo que indica que la búsqueda de hiperparámetros no ha '
  'sobreajustado.'
  % (mejor_test[0], mejor_test[1]['test']['GMean'], mejor_test[1]['test']['AUROC'],
     mejor_test[1]['confusion'][0][0], ho['test_active'], mejor_test[1]['confusion'][1][0],
     mejor_test[1]['gmean_cv_interno']))
p('La figura 1 (fichero %s, entregado junto a esta memoria) recoge las curvas ROC de los tres '
  'modelos sobre el conjunto de test. Las áreas bajo la curva confirman la ordenación de la '
  'tabla 5.' % code('fig_roc_cds29.png'))
top = list(R2['CDS29_top_bits'].items())
p('Por último, las posiciones del fingerprint con mayor importancia según el Random Forest '
  'ajustado son %s. Son los fragmentos estructurales que más discriminan entre activos e '
  'inactivos en este conjunto y, en un estudio QSAR, el punto de partida natural para '
  'interpretar el modelo o para aplicar selección de características.'
  % ', '.join(code(k) for k, _ in top[:8]))

h('4.7. Discusión')
p('La comparación entre las dos bases de datos es el resultado más instructivo de esta fase. '
  'Con el mismo código, las mismas variables predictoras y la misma metodología, CDS16 alcanza '
  'un AUROC de %.3f y CDS29 se queda en %.3f. Las causas son varias y se refuerzan entre sí:'
  % (r16['media']['AUROC'], r29['media']['AUROC']))
ol(['<b>Desequilibrio de clases.</b> En CDS29 sólo el %.1f %% de los compuestos son activos. '
    'Las métricas que corrigen el azar (Kappa, MCC, GMean) caen mucho más que la exactitud, que '
    'se mantiene artificialmente alta.' % (100 * i29['n_active'] / i29['n_molecules']),
    '<b>Diversidad química.</b> CDS16 tiene %d moléculas; es fácil que las de una partición se '
    'parezcan a las de otra, de modo que el clasificador casi está haciendo una búsqueda por '
    'similitud. CDS29, con %d moléculas, cubre un espacio químico mucho más amplio, y la '
    'actividad depende de relaciones que un fingerprint de 1024 bits no captura siempre.'
    % (i16['n_molecules'], i29['n_molecules']),
    '<b>Optimismo del conjunto pequeño.</b> Con %d moléculas, las estimaciones tienen una '
    'desviación típica muy alta y tienden a ser optimistas. El valor de CDS16 debe leerse como '
    'una cota superior poco fiable, no como el rendimiento esperable sobre compuestos nuevos.'
    % i16['n_molecules']])
p('La ordenación de los modelos tampoco se conserva entre las dos bases de datos. En CDS16 '
  'ganan los ensembles de árboles; en CDS29, medidos por GMean, son los peores, y pasan al '
  'frente el SVM y la regresión logística. Esto confirma que ningún algoritmo es mejor en '
  'abstracto: lo que cambia entre los dos casos no es el modelo sino el equilibrio de clases, '
  'y la elección debe hacerse sobre los datos concretos y con la métrica que refleje el coste '
  'real de cada tipo de error.')
p('El k-NN con distancia de Jaccard merece un comentario aparte, porque es la traducción '
  'directa del principio de similitud de Johnson y Maggiora: predice la actividad de un '
  'compuesto a partir de la de sus vecinos más parecidos. En CDS16 funciona razonablemente '
  '(GMean = %.3f) y en CDS29 es el segundo peor (GMean = %.3f). La similitud global entre '
  'fingerprints no basta, por tanto, para decidir la actividad, que es justamente la paradoja '
  'del bioisosterismo citada en la teoría: moléculas distintas pueden producir respuestas '
  'biológicas similares, y moléculas parecidas, respuestas distintas.'
  % (R2['CDS16']['resultados']['k-NN (Jaccard)']['media']['GMean'],
     r29r['k-NN (Jaccard)']['media']['GMean']))
p('Las líneas de mejora naturales, apuntadas también en la parte teórica, serían la selección '
  'de características para quedarse con los bits realmente informativos, el uso de técnicas '
  'específicas para datos desequilibrados (remuestreo tipo SMOTE o ensembles con '
  '<i>boosting</i> sensible al coste) y el empleo de descriptores adicionales a los '
  'fingerprints de 1024 bits.')

# ===========================================================================
h('5. Ampliación: cribado virtual por similitud', 1)
p('El ejemplo %s publicado por la asignatura se ejecuta sin cambios sobre la base de datos '
  'restaurada y devuelve, para la guanosina como molécula de consulta y un umbral de Tanimoto '
  'de 0,4, las seis moléculas de CDS16 que superan ese umbral, empezando por ella misma con '
  'similitud 1,0.' % code('screening_Mol-database-only_mongo_p.py'))
p('Para entender qué aporta cada pieza de la tubería se ha escrito %s, que repite la búsqueda '
  'con cuatro umbrales sobre ambas bases de datos y la compara con una búsqueda exhaustiva que '
  'recorre toda la colección calculando el coeficiente de Tanimoto molécula a molécula. El '
  'objetivo es doble: comprobar que el filtrado previo no pierde ningún resultado y medir '
  'cuántas moléculas llega a descartar.' % code('screening_ampliado.py'))
p('El pipeline aplica dos filtros antes de calcular la similitud exacta. El primero son las '
  '<b>cotas de recuento</b>: para que dos fingerprints alcancen un Tanimoto T, el número de '
  'bits a 1 del candidato debe estar entre T·N<sub>a</sub> y N<sub>a</sub>/T. El segundo son '
  'los <b>bits requeridos</b>: basta con exigir que el candidato comparta al menos uno de los '
  'N<sub>a</sub> − T·N<sub>a</sub> + 1 bits menos frecuentes de la consulta, que es para lo que '
  'sirve la colección %s. Sólo los documentos que pasan ambos filtros llegan a la etapa %s '
  'que calcula el Tanimoto exacto.' % (code('mfp_counts'), code('$project')))
rows = [[r['db'], r['n_total'], '%.1f' % r['threshold'], '[%d, %d]' % (r['qmin'], r['qmax']),
         r['req_bits'], '%d (%.1f %%)' % (r['candidatos'], 100 * r['candidatos'] / r['n_total']),
         r['recuperadas'], '%.1f' % r['t_agregacion_ms'], '%.1f' % r['t_fuerza_bruta_ms'],
         'Sí' if r['mismo_resultado'] else 'No'] for r in SCR]
table(['BD', 'Moléculas', 'Umbral T', 'Cota de bits', 'Bits req.', 'Candidatos evaluados',
       'Recuperadas', 'Agregación (ms)', 'Exhaustiva (ms)', '¿Mismo resultado?'], rows,
      'Tabla 6. Efecto del filtrado previo en la búsqueda por similitud. La búsqueda exhaustiva '
      'se ejecuta en el cliente, en Python, sobre la colección completa.')
s8 = [r for r in SCR if r['db'] == 'CDS29' and r['threshold'] == 0.8][0]
s3 = [r for r in SCR if r['db'] == 'CDS29' and r['threshold'] == 0.3][0]
p('El resultado coincide en todos los casos con la búsqueda exhaustiva, de modo que los filtros '
  'son exactos y no meramente heurísticos: descartan únicamente moléculas que no podrían superar '
  'el umbral. Su eficacia, en cambio, depende por completo de lo exigente que sea la búsqueda. '
  'Sobre CDS29 con T = 0,8 sólo llega a evaluarse el %.1f %% de la colección y la consulta se '
  'resuelve en %.1f ms frente a los %.1f ms del recorrido completo; con T = 0,3 pasa el filtro '
  'el %.1f %% de las moléculas y la ventaja desaparece (%.1f ms frente a %.1f ms), porque '
  'aplicar los filtros cuesta más que lo que ahorran.'
  % (100 * s8['candidatos'] / s8['n_total'], s8['t_agregacion_ms'], s8['t_fuerza_bruta_ms'],
     100 * s3['candidatos'] / s3['n_total'], s3['t_agregacion_ms'], s3['t_fuerza_bruta_ms']))
p('En CDS16 el filtrado nunca compensa, sencillamente porque recorrer %d moléculas es más '
  'barato que preparar la consulta. El esquema está pensado para el caso contrario: cuanto '
  'mayor es la colección y más alto el umbral, mayor es la poda, que es lo que hace viable este '
  'enfoque sobre los cientos de millones de compuestos del caso de AstraZeneca citado en la '
  'teoría de la asignatura. Conviene añadir que la comparación no es del todo equitativa: la '
  'agregación se resuelve dentro del servidor sobre documentos BSON, mientras que la búsqueda '
  'exhaustiva se ha programado en Python en el cliente; lo que la tabla mide con fiabilidad no '
  'son los milisegundos absolutos sino la proporción de la colección que cada umbral permite '
  'descartar.' % i16['n_molecules'])

# ===========================================================================
h('6. Conclusiones', 1)
p('La primera fase muestra que un esquema entidad-interrelación clásico se traslada a MongoDB '
  'sin perder capacidad de consulta, siempre que la decisión entre embeber y referenciar se '
  'tome a partir de las consultas previstas y no por analogía con el modelo relacional. Aquí '
  'referenciar era la opción correcta porque la jefatura cruza los departamentos y porque las '
  'cardinalidades mínimas 0 se representan de forma natural con campos ausentes. El precio es '
  'que la integridad referencial queda en manos de la aplicación.')
p('La segunda fase deja una conclusión metodológica más general: la calidad de un modelo QSAR '
  'depende mucho más de los datos que del algoritmo. Seis clasificadores distintos sobre CDS16 '
  'caben en un margen de pocas centésimas de GMean, mientras que el cambio de CDS16 a CDS29 '
  'hunde el rendimiento de todos ellos. Evaluar con la métrica adecuada, estratificar, repetir '
  'la validación y reservar un conjunto de test independiente no son formalidades: son lo que '
  'distingue una estimación creíble de un número optimista.')
p('La ampliación sobre el cribado por similitud cierra el recorrido: las mismas estructuras '
  'que hacen eficiente la búsqueda de análogos (el fingerprint almacenado como lista de bits y '
  'la colección de frecuencias) son las que alimentan después los modelos de predicción, de '
  'modo que el diseño de la base de datos y el del modelo predictivo son dos caras del mismo '
  'problema.')

h('7. Ficheros entregados', 1)
table(['Fichero', 'Contenido'],
      [[code('fase1_emp_dept.py'), 'Fase 1: creación de colecciones con validación, carga de '
        'datos e informes Q1-Q11.'],
       [code('fase1_salida.txt'), 'Salida completa de la ejecución de la fase 1.'],
       [code('fase2_clasificacion.py'), 'Fase 2: construcción de la matriz de fingerprints, '
        'validación cruzada de seis clasificadores, ajuste de hiperparámetros y evaluación en '
        'test.'],
       [code('fase2_salida.txt'), 'Salida completa de la ejecución de la fase 2.'],
       [code('fase2_resultados.json'), 'Todas las métricas en formato estructurado.'],
       [code('fig_roc_cds29.png'), 'Figura 1: curvas ROC sobre el conjunto de test de CDS29.'],
       [code('screening_ampliado.py'), 'Ampliación del ejemplo de cribado por similitud y '
        'comparación con la búsqueda exhaustiva.'],
       [code('screening_salida.txt') + ', ' + code('screening_resultados.json'),
        'Salida y resumen del cribado.'],
       [code('build_doc.py'), 'Generador de esta memoria a partir de los resultados anteriores.']],
      'Tabla 7. Ficheros fuente que acompañan a la memoria.')

h('8. Referencias', 1)
ul(['G. Cerruela García. <i>Aplicaciones Científicas y Empresariales (I): '
    'Bio/Quimio-Informática</i>. Material de teoría de la asignatura ADP, Universidad de '
    'Córdoba.',
    'A. M. Johnson y G. M. Maggiora. <i>Concepts and Applications of Molecular Similarity</i>. '
    'John Wiley &amp; Sons, 1990.',
    'S. J. Swamidass y P. Baldi. «Bounds and algorithms for fast exact searches of chemical '
    'fingerprints in linear and sublinear time». <i>Journal of Chemical Information and '
    'Modeling</i>, vol. 47, n.º 2, pp. 302-317, 2007.',
    'D. Suvee. «Fingerprint similarity searching with MongoDB», 2011.',
    'MongoDB Inc. <i>MongoDB Manual: Aggregation Pipeline, $lookup, $graphLookup y Schema '
    'Validation</i>.',
    'F. Pedregosa et al. «Scikit-learn: Machine Learning in Python». <i>Journal of Machine '
    'Learning Research</i>, vol. 12, pp. 2825-2830, 2011.'])

open('memoria.html', 'w').write(
    '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Prácticas ADP</title></head>'
    '<body>' + '\n'.join(P) + '</body></html>')
print('memoria.html generado: %d bloques, %d caracteres'
      % (len(P), sum(len(x) for x in P)))
