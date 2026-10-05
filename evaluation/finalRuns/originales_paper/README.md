# originales_paper — réplica del paper original de StereoLoopDetector

Corrida que replica los resultados del paper original de StereoLoopDetector en FieldSAFE. El algoritmo es el
original, sin cambios: la ventana de exclusión es temporal (`dislocal` = 20 s, o sea 200 imágenes a 10 Hz) y
todos los parámetros del detector son los del demo original.

## Código

Commit **d3fff0e** del repo. Los comandos exactos están en [comandos.txt](comandos.txt).

Con el código actual de `main` estos comandos no corren: `--min-distance` pasó a ser obligatorio y reemplaza
la ventana de 20 s por una distancia mínima recorrida. Para reproducir esta corrida hay que compilar desde
d3fff0e.

## Cambios respecto del repo original

Ninguno en el algoritmo. Solo cambios de infraestructura:

- `--frequency` como parámetro de `demo_stereo` (antes estaba fijo en 10 Hz). Se usa `--frequency 10`, que es
  el valor que tenía hardcodeado.
- Los loops detectados se guardan en `<fecha>_results.yml` (ids de query y match, y la traslación de PnP).
- Vocabulario ORB: `resources/ORBvoc.yml.gz`, el de ORB-SLAM3 convertido al formato YAML de DBoW2 con
  `resources/convert_voc.py`.

## Sesiones

| Archivo | Sesión de FieldSAFE | Imágenes | Loops |
|---|---|---|---|
| `fs_dynamic1_results.yml` | Dinámica #1 (2016-10-25-11-41-21) | 11363 | 197 |
| `fs_static1_results.yml` | Estática #1 (2016-10-25-11-09-42) | 9475 | 198 |
| `fs_dynamic2_results.yml` | Dinámica #2 (2016-10-25-12-07-22) | 12182 | 122 |

## Evaluación

[translation_error.ipynb](translation_error.ipynb) compara la magnitud de la traslación de PnP contra la
distancia GPS entre query y match. El gráfico es `translation_error_boxplot.png`.

## Observaciones

- La mayoría de los loops son con el robot detenido (136/197, 187/198 y 122/122): el robot se ve a sí mismo
  después de más de 20 s quieto. La mediana de ~0,15 m del error sale casi toda de esos loops.
- Con el robot quieto, PnP devuelve ‖t‖ ≈ 0,16 m cuando la verdad es ≈ 0. Es un sesgo del código original:
  PnP usa la K cruda con distorsión en vez de la matriz rectificada P (4 % de diferencia de focal en
  FieldSAFE). Está diagnosticado en la rama `claude` (`evaluation/BITACORA.md`, Fase 1, config `p1_rectK`) y
  arreglado en [pnp_rectificado](../pnp_rectificado/).
