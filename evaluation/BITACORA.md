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

---

## 2026-09-30 — Fase 2: gating por distancia y verificación más permisiva (FieldSAFE, 3 sesiones)

| Config (sobre p1_all) | Loops | TP | Triviales | FP | Precisión | Cobertura | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|---|---|
| baseline (original) | 517 | 61 | 456 | 0 | 1.00 | 1.7 % | 0.50 m | 8.5° |
| p1_all | 494 | 40 | 454 | 0 | 1.00 | 1.1 % | 0.29 m | 4.8° |
| **p2_excl** (exclusión 20 m por odometría visual) | 39 | 39 | **0** | 0 | 1.00 | 1.2 % | 0.27 m | 4.8° |
| p2_nocross (sin chequeo L-L ∧ R-R) | 565 | 93 | 472 | 0 | 1.00 | 1.8 % | 0.55 m | 7.0° |
| p2_ratio08 (ratio test 0.8) | 562 | 90 | 472 | 0 | 1.00 | 1.8 % | 0.56 m | 7.1° |
| p2_cands3 (3 candidatos) | 497 | 43 | 454 | 0 | 1.00 | 1.1 % | 0.31 m | 4.8° |
| p2_bypass (≥ 60 inliers sin consistencia temporal) | 505 | 40 | 465 | 0 | 1.00 | 1.1 % | 0.29 m | 4.8° |
| p2_combo (todo) | 165 | **160** | 0 | 5 | 0.97 | 3.2 % | 0.65 m | 8.7° |

- La **odometría visual estéreo elimina el 100 % de los loops triviales** sin perder TP y sin GPS.
- Relajar la geometría da ~2.3× más TP pero empeora la pose y el recall sigue < 1 %.

**Embudo sobre los frames revisita GT** (`funnel.py`): en qué etapa se pierde cada revisita y si el mejor
candidato de BoW estaba a < 3 m.

| | FieldSAFE (p2_combo) | Rosario (baseline) |
|---|---|---|
| Frames revisita | 22 000 | 22 800 |
| Candidato BoW correcto | **~8 %** | ~36 % |
| Perdidos por consistencia temporal | 71 % | 44 % |
| Perdidos por geometría | 25 % | 35 % |

→ **En FieldSAFE el cuello de botella es la recuperación**: el 92 % de los candidatos de BoW son de otro
lugar (aliasing), y ningún ajuste posterior puede recuperar esos loops.

**Datos**: extraídas y preparadas Rosario 12-26 13:39 (2.2 km) y 15:48 (1.7 km), FieldSAFE 11:34 (110 m, poco
útil) y 12:37 (4.7 km).

**Próximo paso (Fase 3)**: recuperación con descriptores globales aprendidos (DINOv2-SALAD) y medir recall@K
offline contra BoW; si mejora, integrarlo en el detector (reemplazo o fusión con BoW) manteniendo la
verificación estéreo.

---

## 2026-09-30 — Fase 3: recuperación con DINOv2-SALAD (offline)

Recall@K sobre frames revisita GT (base = frames con ≥ 20 m de camino detrás; acierto si alguno de los K
más similares está a < 3 m). PCA 256-D ajustada **solo con FieldSAFE**; en Rosario se aplica sin reajuste.

| | R@1 | R@5 | R@10 | R@25 | Candidato BoW correcto (SLD) |
|---|---|---|---|---|---|
| FieldSAFE (5 sesiones) | 0.34 | 0.61 | 0.72 | 0.83 | ~0.08 |
| Rosario (6 secuencias) | 0.63 | 0.73 | 0.77 | 0.83 | ~0.36 |

- SALAD multiplica ×4 (FieldSAFE) y ×1.8 (Rosario) el acierto del primer candidato.
- En FieldSAFE el top-1 "incorrecto" es casi siempre la pasada vecina: está a < 5 m el 80 % de las veces.
- p2_combo sobre las 5 sesiones de FieldSAFE: 165 TP, 14 FP (precisión 0.92), cobertura 2.4 %.
- Las sesiones nuevas agregan muchos loops triviales (p1_all: 2573 triviales sobre 5 sesiones), lo que
  refuerza el valor de la exclusión por distancia.

**Incidente**: un reinicio de la sesión mató los procesos en segundo plano; se relanzaron con `setsid`
(`evaluation/queue.sh`). En cola: p3_salad, p3_salad_c5, p3_salad_combo, p4_mask, p4_depth, p4_hybrid,
baseline en Rosario completo. En paralelo: extracción de Rosario a 1280×720 en el disco raíz.

---

## 2026-09-30 — Fase 3: SALAD integrado → el cuello de botella pasa a la verificación

p3_salad (p2_excl + recuperación SALAD) en FieldSAFE (5 ses.): **33 TP** (p2_excl: 39), 0 FP.
Embudo: candidato correcto 26 % (BoW: 7 %), la consistencia temporal ya no frena (pasa 81 %), pero la
**verificación ORB rechaza 7581 candidatos correctos** (mediana de 7 matches L-L en pares correctos).

**Experimento offline** (`verify_offline.py`, 150 pares correctos rechazados por ORB vs 150 pares a > 10 m;
inliers de matriz esencial RANSAC a 1 px, máscara del tractor):

| Método | Inliers med (correctos) | Inliers med (incorrectos) | Recall @ 0 FP | Recall @ 1 % FP |
|---|---|---|---|---|
| ORB (2000, ratio 0.8) | 76 | 32 | 0.11 | 0.15 |
| SuperPoint + LightGlue | 219 | 118 | 0.77 | 0.79 |
| DISK + LightGlue | 312 | 41 | 0.86 | 0.86 |
| **ALIKED + LightGlue** | **384** | 86 | **0.95** | **0.95** |

→ Features aprendidas + LightGlue recuperan el 95 % de los loops que ORB descartaba, sin falsos positivos
en la muestra. Próximo paso: etapa de verificación ALIKED+LightGlue con pose métrica (estéreo + PnP) sobre
los candidatos de SLD+SALAD, y evaluación completa.

**p3_salad_c5** (SALAD + verificar 5 candidatos, ORB): 56 TP, 1 FP (p2_excl: 39 TP). En curso: re-verificación
de los ~37 k candidatos de p3_salad_c5 con ALIKED+LightGlue + PnP estéreo + rayos lejanos
(`learned_verify.py`, guarda puntajes por candidato para barrer umbrales / curvas PR).

| Config (FieldSAFE, 5 ses.) | Loops | TP | FP | Precisión | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|---|---|
| p2_excl | 39 | 39 | 0 | 1.00 | 0.8 % | 30 | 0.28 m | 5.5° |
| p3_salad | 33 | 33 | 0 | 1.00 | 0.9 % | 15 | 0.67 m | 8.5° |
| p3_salad_c5 | 57 | 56 | 1 | 0.98 | 1.3 % | 22 | 0.63 m | 10.7° |
| p3_salad_combo (+ verificación ORB permisiva) | 418 | 374 | 43 | 0.90 | 4.2 % | 95 | 1.07 m | 11.6° |
| p4_mask (p2_excl + máscara del tractor) | 36 | 36 | 0 | 1.00 | 0.9 % | 21 | 0.44 m | 7.3° |

- Con ORB, aflojar la verificación sube TP (374) pero la precisión cae a 0.90 y la pose se degrada: ORB no
  alcanza para verificar en pasto. La máscara no cambia nada con BoW+ORB (se re-evaluará con SALAD/ALIKED).

---

## 2026-09-30 — Fase 4: features lejanos (FieldSAFE, 5 ses., sobre p2_excl)

| Config | TP | FP | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|
| p2_excl (referencia) | 39 | 0 | 0.8 % | 30 | 0.28 m | 5.5° |
| p4_depth (descartar puntos con error de profundidad > 10 %) | 19 | 0 | 0.5 % | 18 | **0.22 m** | **4.3°** |
| **p4_hybrid** (p4_depth + lejanos como rayos de profundidad desconocida) | **53** | 0 | 1.0 % | **32** | 0.42 m | 6.9° |

- Tirar los puntos lejanos mejora la pose pero pierde la mitad de los loops; usarlos como **rayos** los
  recupera y suma: +36 % de TP respecto de la referencia, con precisión 1.00. Confirma que los lejanos
  sirven para verificar (rotación / soporte) aunque su profundidad estéreo no sirva.
- Rosario a 1280×720: extraída 12-22 13:14 (mismos 13 745 frames que la versión reducida);
  calibración `resources/rosario_fullres_stereo_parameters.yaml` (intrínsecos ×2, del `camera_info`).

---

## 2026-09-30 — Fase 5: verificación ALIKED + LightGlue (parcial: FieldSAFE estática #1 y dinámica #1)

Candidatos de p3_salad_c5 re-verificados con `learned_verify.py` (umbral por defecto: ≥ 15 inliers
cercanos y ≥ 60 totales, incluyendo rayos lejanos).

| Sesión | Config | TP (< 3 m) | "FP" (≥ 3 m) | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|---|
| estática #1 | p2_excl (ORB) | 0 | 0 | 0 % | 0 | — | — |
| estática #1 | **p5_aliked** | 1853 | 2565 | **93.5 %** | 3189 | 0.30 m | 1.9° |
| dinámica #1 | p2_excl (ORB) | 39 | 0 | 3.5 % | 30 | 0.27 m | 5.5° |
| dinámica #1 | **p5_aliked** | 2167 | 2733 | **79.7 %** | 3069 | 0.28 m | 1.7° |

- Los "FP" del criterio de 3 m **no son falsos**: con cualquier umbral, el 100 % de los loops aceptados está a
  < 10 m, y la distancia estimada coincide con la GT (mediana 3.38 m vs 3.38 m en el tramo 3–4 m). Son loops
  con la pasada vecina, con pose relativa correcta → restricciones válidas para SLAM. El criterio de 3 m
  queda chico; hay que evaluar por pose (propuesta #2 del agente).
- Error por distancia GT: 2–5 m → e_vec 0.24–0.37 m, yaw ~1.5°; < 2 m → yaw 8–11° (sospecha: giros en
  cabeceras, donde el heading GT derivado de la trayectoria es malo) ; > 6 m → degrada.
- pose_ok (e_vec < 1 m ∧ e_yaw < 10°) ≈ 70 % de los aceptados; el umbral de inliers casi no lo cambia
  → el techo parece estar en la GT de heading, a revisar con el heading del GPS dual (GPHDT) de FieldSAFE.

---

## 2026-09-30 — Fase 5 completa en FieldSAFE + GT de orientación del sensor

**GT de yaw**: ahora se usa el heading del GPS de doble antena (`$GPHDT`, `fieldsafe_heading.py`) en FieldSAFE y
el cuaternión 6DoF en Rosario, con el offset de montaje estimado como la mediana contra la trayectoria en
movimiento (FieldSAFE: ~84°, antena transversal). El heading derivado de la trayectoria se desviaba > 8–12°
en el 10 % de los frames (giros) y en Rosario hasta 140° (marcha atrás).

| Config (FieldSAFE, 5 ses.) | Loops | TP < 3 m | Cobertura | **pose_ok** | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|
| baseline (original) | 2716 | 65 | 1.3 % | 47 | 0.57 m | 5.8° |
| p2_excl | 39 | 39 | 0.8 % | 33 | 0.27 m | 4.2° |
| p4_hybrid | 53 | 53 | 1.0 % | 39 | 0.41 m | 5.7° |
| p3_salad_c5 | 57 | 56 | 1.3 % | 33 | 0.63 m | 8.3° |
| **p5_aliked** (SALAD + ALIKED/LightGlue + estéreo + rayos) | **16 948** | **7125** | **72.1 %** | **11 644** | **0.33 m** | **2.0°** |

(pose_ok = loops no triviales con e_vec < 1 m y e_yaw < 10°, a cualquier distancia; el baseline incluye los
2651 loops triviales en "Loops".)

- **× 250 loops con pose correcta y cobertura de 1.3 % → 72 %** respecto del SLD original.
- 69 % de los loops aceptados tienen pose correcta; el resto falla sobre todo por traslación (92 %), con el
  error correlacionado negativamente con el desplazamiento GT (−0.69) y sesgo lateral por sesión.
  Ángulo de montaje cámara–vehículo estimado: ~3° (mejora poco: 0.45 → 0.38 m).
  **Pendiente**: verificar la sincronización GPS–cámara (offset de 19.25 s estimado a mano; 0.1 s = 0.25 m)
  comparando la velocidad de la odometría visual con la del GPS.
