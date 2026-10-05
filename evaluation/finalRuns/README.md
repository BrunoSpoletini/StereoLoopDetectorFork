# finalRuns — corridas para el paper

Corridas cuyos resultados van al paper. Las corridas de prueba están en `../testRuns/`, que tiene el mismo
formato pero no se versiona.

## Formato

Una carpeta por corrida. Cada carpeta tiene:

- `README.md`: qué cambios se hicieron para esa corrida, con qué código (commit) y con qué parámetros.
- `<sesion>_results.yml`: los loops detectados en cada sesión.
- Opcionales, si la corrida los generó: `<sesion>.log` (salida del demo), `<sesion>_queries.csv`
  (diagnóstico por query), `<sesion>_config.yaml` (config usada), y los comandos o notebooks de evaluación.

Nombres de sesión:

| Prefijo | Dataset | Ejemplo |
|---|---|---|
| `fs_` | FieldSAFE | `fs_dynamic1`, `fs_static1`, `fs_dynamic2`, `fs_11-34`, `fs_12-37` |
| `ro_` | RosarioV2 a 640×360 | `ro_1222_1314` |
| `rof_` | RosarioV2 a 1280×720 | `rof_1222_1314` |

## Corridas

| Carpeta | Descripción |
|---|---|
| [originales_paper](originales_paper/) | Réplica del paper original de StereoLoopDetector en FieldSAFE |
| [pnp_rectificado](pnp_rectificado/) | originales_paper con el PnP usando los intrínsecos rectificados |
| [distancia_minima_20m](distancia_minima_20m/) | pnp_rectificado con exclusión por 20 m recorridos en vez de 20 s |
