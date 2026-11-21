# Política de seguridad

## Versiones con soporte

| Versión | Soporte |
| --- | :---: |
| 1.1.x (la última, `main`) | ✅ |
| < 1.1.0 | ❌ |

Solo la última versión de `main` recibe correcciones.

## Cómo informar de una vulnerabilidad

> [!IMPORTANT]
> **No abras un *issue* público** para un problema de seguridad.

Infórmame en privado a través de GitHub: **[Report a vulnerability](https://github.com/Pablo-Caballero-Maria/trabajo_molecula/security/advisories/new)** (pestaña *Security* →
*Advisories* → *Report a vulnerability*). Incluye:

1. qué está afectado (fichero, ruta de la API, función…) y la versión o el *commit*;
2. cómo reproducirlo, paso a paso;
3. el impacto que esperas (qué podría leer, cambiar o romper alguien);
4. una corrección o mitigación, si se te ocurre.

Confirmaré que lo he recibido en un plazo de **7 días**, te mantendré al tanto mientras lo corrijo y te
mencionaré en el aviso y en el [registro de cambios](CHANGELOG.md), salvo que prefieras el anonimato.

## Alcance

- Credenciales: los scripts usan por defecto el usuario de prácticas `adp:adp` de la asignatura,
  pensado solo para un servidor local en `127.0.0.1`. Cualquier credencial real debe pasarse con
  la variable de entorno `MONGO_URI`, nunca escribirse en el código.
- Dependencias con vulnerabilidades conocidas.

Fuera de alcance: Diferencias en los resultados (usa la plantilla «Un resultado no se reproduce»).

## Qué hace ya el repositorio

- **Escaneo de secretos** con protección al hacer *push*: se bloquea un *commit* que contenga un token
  conocido.
- **Dependabot**: alertas y actualizaciones de seguridad de las dependencias y de las GitHub Actions,
  y actualizaciones semanales de versión.
- **CodeQL**: análisis del código y de los *workflows* en cada *push*, *pull request* y cada semana.
