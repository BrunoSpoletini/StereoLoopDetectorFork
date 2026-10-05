# 03_distancia_minima_20m — exclusión por distancia recorrida

Misma corrida que [02_pnp_rectificado](../02_pnp_rectificado/) (PnP con los intrínsecos rectificados), con la exclusión
cambiada: en vez de no matchear contra las imágenes de los últimos 20 s (`dislocal`), solo se matchea contra imágenes
desde las que el robot recorrió al menos 20 m (`--min-distance 20`, medido sobre el plano xy con el archivo de poses).

## Código

Commit **5dc376c** del repo. Los comandos exactos están en [comandos.txt](comandos.txt).

## Sesiones

| Archivo | Sesión de FieldSAFE | Imágenes | Loops |
|---|---|---|---|
| `fs_dynamic1_results.yml` | Dinámica #1 (2016-10-25-11-41-21) | 11363 | 55 |
| `fs_static1_results.yml` | Estática #1 (2016-10-25-11-09-42) | 9475 | 0 |
| `fs_dynamic2_results.yml` | Dinámica #2 (2016-10-25-12-07-22) | 12182 | 0 |

## Evaluación

[translation_error.ipynb](translation_error.ipynb) compara contra 02_pnp_rectificado: cantidad de loops, dónde caen en la
trayectoria (`loops_map.png`) y error de traslación de los loops en movimiento (`translation_error_boxplot.png`).

## Observaciones

- Desaparecen todos los loops con el robot detenido (445 de 517 con la ventana de 20 s).
- Estática #1 y Dinámica #2 quedan sin loops. Los 11 loops "en movimiento" de Estática #1 en 02_pnp_rectificado tenían solo
  1–2 m de recorrido entre match y query: el robot avanzaba muy despacio, no eran revisitas.
- Dinámica #1 queda con 55 loops, todos con más de 229 m recorridos entre match y query. 52 son los mismos pares que en
  02_pnp_rectificado. La mediana del error de traslación es 0,049 m (0,051 m con la ventana de 20 s).
- Las tres sesiones tienen miles de frames que revisitan un lugar, así que la cobertura del detector original en
  FieldSAFE es muy baja.
