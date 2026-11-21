# Registro de cambios

Todos los cambios relevantes del trabajo se anotan aquí. El formato sigue
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto usa
[versionado semántico](https://semver.org/lang/es/). Cada versión tiene su etiqueta (`vX.Y.Z`) en el
repositorio.

## [Sin publicar]

## [1.1.0] - 2026-11-21

### Añadido

- **Tests con pytest** (`tests/`): 48 unitarios (métricas, modelos, Optuna, ruido y outliers,
  similitud de Tanimoto, selección e ingeniería de características, enjambre de SHAP y figuras) y 26 de
  integración contra MongoDB (marcados `mongo`), que restauran CDS16 y CDS29 desde
  `instrucciones/dump.tar` y comprueban que se reproducen las salidas entregadas: el informe completo de
  la fase 1 (`fase1_salida.txt`), el cribado (`screening_resultados.json`) y la GMean y el AUROC en test
  del Random Forest de Optuna (`fase2_resultados.json`). Cobertura con `pytest-cov`.
- Integración continua con GitHub Actions en cada *push* y *pull request*: Ruff (lint y formato), ty y
  los tests con cobertura contra un MongoDB 8 real (resumen en cada ejecución e insignia en el README).
- Seguridad: análisis con CodeQL, Dependabot (uv y GitHub Actions), escaneo de secretos con protección
  al hacer *push* y [política de seguridad](SECURITY.md) con avisos privados.
- Licencia [MIT](LICENSE) para el código, [código de conducta](CODE_OF_CONDUCT.md),
  [CITATION.cff](CITATION.cff), plantillas de *issues* (error, mejora y resultado que no se reproduce) y
  de *pull request*.
- [Wiki](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/wiki) con cada fase, la ampliación,
  la reproducción y los tests.
- La variable de entorno `MONGO_URI` también vale para la fase 1 y el cribado.

### Cambiado

- `fase1_emp_dept.py`, `screening_ampliado.py` y `figuras_screening.py` se organizan en funciones con un
  `main()`, para poder importarlos desde los tests. Lo que imprimen y generan no cambia: la fase 1
  reproduce `fase1_salida.txt` byte a byte y la figura del cribado sale idéntica.
- README reorganizado, con insignias, índice, diagramas y figuras.

## [1.0.0] - 2026-11-19

Versión entregada.

### Añadido

- Fase 2: validación cruzada de 12 combinaciones de ruido × eliminación de outliers, con la frontera de
  decisión de cada una sobre dos componentes principales.
- Fase 2: selección (cinco criterios de filtrado, embebidos y envolvente) e ingeniería de
  características con RDKit (cuentas de subestructuras, descriptores fisicoquímicos, claves MACCS y
  variables de similitud).
- Fase 2: explicabilidad con SHAP (gráfico de enjambre del Random Forest ajustado con Optuna).
- Proyecto nativo con uv (`pyproject.toml`, `uv.lock`), Ruff y ty.
- La memoria final en PDF y DOCX y el zip de la entrega (`entregables/`).

## [0.1.0] - 2026-11-13

### Añadido

- Fase 1: esquema DEPT/EMP en MongoDB con validación `$jsonSchema`, datos de prueba e informes Q1-Q11.
- Fase 2: siete clasificadores con validación cruzada estratificada repetida sobre CDS16 y CDS29, y
  ajuste con Optuna y evaluación en test sobre CDS29.
- Ampliación: cribado virtual por similitud de Tanimoto con el framework de agregación, comparado con
  una búsqueda exhaustiva.

[Sin publicar]: https://github.com/Pablo-Caballero-Maria/trabajo_molecula/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Pablo-Caballero-Maria/trabajo_molecula/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Pablo-Caballero-Maria/trabajo_molecula/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/Pablo-Caballero-Maria/trabajo_molecula/releases/tag/v0.1.0
