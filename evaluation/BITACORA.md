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

---

## 2026-09-29 — Baseline completo (SLD original, frecuencia real de cada dataset)

| Dataset | Loops | TP | Triviales | FP | Recall | Cobertura | e_vec med |
|---|---|---|---|---|---|---|---|
| FieldSAFE (3 ses.) | 517 | 61 | 456 | 0 | 0.3 % | 1.7 % | 0.50 m |
| Rosario (4 sec.) | 3643 | 3441 | 202 | 0 | 15 % | 33 % | 0.04 m |

Rosario es muy desparejo: 12-22 13:14 da 3375 TP (cobertura 80 %); las otras 3 dan 55, 0 y 11 TP.

## 2026-09-29 — Fase 1: arreglos (FieldSAFE)

| Config | TP | Triviales | e_vec med (TP) | e_yaw med | ‖t‖ con robot quieto (verdad ≈ 0) |
|---|---|---|---|---|---|
| baseline | 61 | 456 | 0.50 m | 8.5° | 0.157 m |
| p1_near (0.4 m real) | 61 | 456 | 0.50 m | 8.5° | 0.157 m (sin efecto) |
| p1_rectK (K rectificada) | 61 | 456 | 0.60 m | 10.3° | **0.001 m** |
| p1_ransac (≥ 20 inliers a 3 px) | 16 | 449 | 0.39 m | 4.7° | 0.102 m |
| p1_disp (disparidad > 0, z > 0) | 45 | 456 | **0.26 m** | **4.1°** | 0.154 m |
| p1_all | 40 | 454 | 0.29 m | 4.8° | 0.001 m |

- **Confirmado**: el sesgo de 0.16 m con el robot quieto era el desajuste de intrínsecos (K cruda vs P
  rectificada, 4 % de focal). Con la K correcta cae a 1 mm.
- Con la K rectificada, e_vec de los TP en movimiento empeora (0.50 → 0.60 m): hay que revisar si el GT
  en movimiento (brazo de palanca antena–cámara, offset temporal GPS de 19.25 s) domina ese error.
- RANSAC con ≥ 20 inliers descarta 45 de 61 TP: la geometría de los loops reales es débil (pocos inliers).
- Los arreglos mejoran la pose (yaw 8.5° → ~4.5°) pero **no el recall**: el cuello de botella está antes
  (recuperación de candidatos, consistencia temporal y el umbral geométrico).

**Próximo paso**: Fase 2 sobre `p1_all`: exclusión por distancia con odometría visual, verificación sin
chequeo cruzado L-L/R-R, ratio test 0.8, 3 candidatos, bypass temporal con ≥ 60 inliers.
