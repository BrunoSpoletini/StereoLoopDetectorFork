# 02_pnp_rectificado — PnP con los intrínsecos rectificados

Misma corrida que [01_originales_paper](../01_originales_paper/) (algoritmo original, ventana de exclusión temporal de 20 s,
parámetros del demo original) con un único cambio: el PnP de la verificación geométrica usa los intrínsecos
rectificados (las 3 primeras columnas de la matriz de proyección P izquierda) y sin distorsión, en vez de la K cruda +
coeficientes de distorsión.

## Código

Commit **59db499** ("Arreglo de Solver PnP en robot estatico"), que es d3fff0e + el arreglo. Los comandos exactos están en
[comandos.txt](comandos.txt).

## Sesiones

| Archivo | Sesión de FieldSAFE | Imágenes | Loops |
|---|---|---|---|
| `fs_dynamic1_results.yml` | Dinámica #1 (2016-10-25-11-41-21) | 11363 | 197 |
| `fs_static1_results.yml` | Estática #1 (2016-10-25-11-09-42) | 9475 | 198 |
| `fs_dynamic2_results.yml` | Dinámica #2 (2016-10-25-12-07-22) | 12182 | 122 |

Los loops (ids de query y match) son exactamente los mismos que en 01_originales_paper: el PnP no decide si se acepta un
loop. Solo cambia la traslación estimada.

## Evaluación

[translation_error.ipynb](translation_error.ipynb) compara, loop por loop, el error de traslación de 01_originales_paper
(K cruda) contra esta corrida (K rectificada). Gráficos: `translation_error_boxplot.png` y `translation_scatter.png`.

## Resultados

| Sesión | Grupo | n | Mediana del error, antes → después |
|---|---|---|---|
| Dinámica #1 | Detenido | 136 | 0,151 → 0,002 m |
| Dinámica #1 | En movimiento | 61 | 0,092 → 0,051 m |
| Estática #1 | Detenido | 187 | 0,149 → 0,004 m |
| Estática #1 | En movimiento | 11 | 0,163 → 0,033 m |
| Dinámica #2 | Detenido | 122 | 0,155 → 0,004 m |

- Con el robot detenido, ‖t_est‖ baja de ~155 mm a 1–3 mm de mediana: desaparece el sesgo.
- En movimiento la estimación deja de sobreestimar ‖t‖ y queda alrededor de la diagonal.
- Dos loops de Estática #1 (q=618 y q=541) siguen con ‖t‖ de varios metros con las dos K: son fallas de PnP con malas
  correspondencias, no del arreglo. Por uno de ellos la media en movimiento de Estática #1 sube (0,52 → 0,71 m).
