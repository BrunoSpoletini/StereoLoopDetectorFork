# Bitácora — mejoras de StereoLoopDetector (rama `claude`)

Objetivo: mejorar la detección de loops de SLD en campo agrícola.
**FieldSAFE = train** (desarrollo y ajuste), **RosarioV2 = validación** (config congelada, sin reajustar).

## Protocolo de evaluación (`evaluation/sld_eval.py`)

- **Revisita GT**: frame con otro frame anterior a < 3 m (GPS), con ≥ 20 m de camino recorrido entre ambos
  y en el mismo sentido de marcha (|Δyaw| < 60°; una cámara frontal no puede cerrar un surco recorrido al revés).
- **Clasificación de cada loop detectado**: TP (< 3 m y ≥ 20 m de camino), **trivial** (< 20 m de camino:
  p. ej. robot detenido, correcto pero inútil para SLAM) y FP (≥ 3 m).
- **Métricas**: #TP, precisión = TP/(TP+FP), recall por frames revisitados, **cobertura** (fracción de tramos
  de 10 m de trayecto revisitado con al menos un TP) y error de pose de los TP: `e_mag` (|‖t‖−‖Δgt‖|, la
  métrica original), `e_vec` (error del vector de traslación en el plano) y `e_yaw`.
- Corridas: `python evaluation/run_sld.py --config evaluation/configs/<cfg>.yaml --sessions fieldsafe|rosario|all`.
  Cachea las features ORB (`/home/bruno/Desktop/tesina/sld_cache`), así que cambiar parámetros de detección
  cuesta ~3 min por sesión. La caché reproduce los resultados bit a bit.

---

## 2026-09-29 — Diagnóstico inicial

**RosarioV2 vs FieldSAFE con SLD original** (corridas previas, métricas nuevas):

| Secuencia | Frames revisitados | Loops | TP | Triviales (< 20 m de camino) |
|---|---|---|---|---|
| Rosario 12-22 13:14 | 8551 | 3604 | 2334 | ~1270 |
| FieldSAFE dinámica #1 | 8194 | 197 | 61 | 136 |
| FieldSAFE estática #1 | 5886 | 198 | 0 | 198 |
| FieldSAFE dinámica #2 | 7919 | 122 | 0 | 122 |

- En FieldSAFE casi todos los loops son **triviales**: robot detenido > 20 s viéndose a sí mismo
  (la ventana de exclusión es temporal). La "mediana de 15 cm" del paper sale de esos casos:
  con el robot quieto el PnP devuelve ‖t‖ ≈ 0.16 m cuando la verdad es ≈ 0 → hay un sesgo.
- Cero falsos positivos en ambos datasets: el problema es el **recall**, no la precisión.
- En FieldSAFE, de los candidatos rechazados, ~2500 eran correctos y los tiró la verificación geométrica,
  ~1300 los tiró la consistencia temporal; ~75 % de los candidatos de BoW son incorrectos.

**Problemas encontrados en el código**:
1. `near_distance`/`far_distance` eran `int`: el 0.4 m se truncaba a 0 (filtro de cercanía inactivo).
2. PnP usa la K cruda + distorsión, pero las imágenes y la triangulación están rectificadas (P).
   En FieldSAFE fx = 556 vs 581 (4 %): sesgo de escala en la traslación.
3. `solvePnP` sin RANSAC ni conteo de inliers: un outlier arruina la pose.
4. Matcher estéreo sin chequeo de disparidad: entran puntos detrás de la cámara.
5. Ventana de exclusión y consistencia temporal definidas en segundos, no en distancia recorrida.

**Infraestructura**: config YAML (`--config`), salida nombrada (`--output`), caché de features (`--cache`),
CSV de diagnóstico por query (estado, candidato, score, inliers, pose), rotación del PnP en el `.yml`.
Todos los defaults reproducen el comportamiento original.

**Datos**: extrayendo las 2 secuencias de Rosario faltantes (12-26 13:39 y 15:48) y 2 sesiones de FieldSAFE
nunca extraídas (11:34 y 12:37). Calibración de Rosario: una única calibración para todo el dataset
(repo oficial `CIFASIS/rosariov2`, Kalibr) → se usa la misma para todas las secuencias.

**Próximo paso**: baseline en las 7 sesiones disponibles; después Fase 1 (arreglos 1–4, uno por uno).
