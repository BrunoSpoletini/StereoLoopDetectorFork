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

---

## 2026-09-30 — Corrección del GT de FieldSAFE: desfase GPS–cámara de +1.5 s

Dos estimaciones independientes coinciden en que el GPS estaba asociado a la imagen equivocada:
1. La velocidad GPS está retrasada 1.4–1.7 s respecto de la de la odometría visual estéreo (3 de 4 sesiones).
2. El % de loops de p5_aliked con pose correcta es máximo con el GT corrido 15 frames (1.5 s) en las 4 sesiones
   (76→90 %, 66→84 %, 59→85 %, 73→85 %).

→ `fieldsafe_gt.py` regenera poses y heading con desfase 19.25 + 1.5 = 20.75 s, **manteniendo exactamente los
mismos pares/ids** (las cachés siguen valiendo). El GT se mueve 3–4.7 m en promedio; el retardo contra la
odometría queda en 0. Poses viejas: `poses_<sesion>_offset19.25.csv`. **Todas las métricas de FieldSAFE de
arriba quedan superadas por esta tabla**:

| Config (FieldSAFE, 5 ses.) | Loops no triviales | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|
| baseline (SLD original) | 65 (+ 2651 triviales) | 1.3 % | 65 | 0.20 m | 1.3° |
| p1_all (arreglos) | 40 | 0.6 % | 40 | 0.06 m | 1.1° |
| p2_excl (+ exclusión por distancia) | 39 | 0.7 % | 39 | **0.05 m** | 1.2° |
| p2_combo | 179 | 2.6 % | 177 | 0.13 m | 1.4° |
| p3_salad_combo | 417 | 4.7 % | 404 | 0.15 m | 1.8° |
| p4_hybrid | 53 | 0.9 % | 53 | 0.08 m | 1.3° |
| **p5_aliked** | **16 919** | **74.0 %** | **16 270 (96 %)** | **0.14 m** | **1.2°** |

- Con el GT bien sincronizado **el 96 % de los loops de p5_aliked tiene pose correcta** y ninguno está a > 10 m.
- La rareza de Fase 1 ("K rectificada empeora e_vec en movimiento") era un artefacto de la sincronización.
- Los arreglos de Fase 1 bajan el error del SLD original de 0.20 m a 0.05 m.

---

## 2026-09-30 — Validación en Rosario (parcial: 4 secuencias cortas)

Fix del GT: el cuaternión de Rosario es la orientación **de la cámara** (eje z = óptico, ~19° hacia abajo); el
yaw se toma como la dirección del eje óptico en el plano (antes se usaba Euler zyx → errores de 20–40°).

| Config (Rosario, 4 sec., sin reajuste) | Loops | TP | Triviales | FP | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|---|---|
| baseline (SLD original) | 3643 | 3441 | 202 | 0 | 32.7 % | 3364 | 0.035 m | 0.8° |
| p3_salad_c5 | 4001 | 3959 | 42 | 0 | 32.0 % | 3691 | 0.047 m | 1.9° |

- Con ORB en la verificación, SALAD suma +15 % de TP en Rosario pero no cambia la cobertura (la secuencia
  12-22 13:14 domina; 16:31 sigue en 0 loops). En curso: re-verificación con ALIKED+LightGlue (~45 k candidatos).

**p5_aliked en Rosario 12-22 13:14** (validación, sin reajuste): 6775 loops, 6646 TP (baseline 3375), recall 0.79
(0.40), cobertura 92 % (82 %), pose_ok 6606 (3350), e_vec 0.037 m, e_yaw 1.1°; 84 loops a > 3 m (precisión por
distancia 0.988). La mejora de FieldSAFE se traslada a Rosario.

**p5_aliked en Rosario 12-22 14:29 y 16:31** (validación):

| Secuencia | Config | Loops | TP | ≥ 3 m | Cobertura | pose_ok | e_vec med (TP) | e_yaw med |
|---|---|---|---|---|---|---|---|---|
| 14:29 | baseline | 142 | 55 | 0 | 11.5 % | 4 | 0.20 m | 11.9° |
| 14:29 | p5_aliked | 999 | 941 | 56 | **84.6 %** | 524 | 0.13 m | 8.7° |
| 16:31 | baseline | 0 | 0 | 0 | 0 % | 0 | — | — |
| 16:31 | p5_aliked | 672 | 137 | **534** | 16 % | **0** | 1.29 m | 2.0° |

- La sincronización de Rosario está bien (barrido de desfase: óptimo en 0).
- **16:31: aliasing a lo largo del surco.** Los loops falsos unen frames del mismo surco a 20–35 m (justo
  pasada la exclusión de 20 m): el horizonte no cambia y las hileras cercanas se repiten, así que el PnP
  estima ‖t‖ ≈ 0.2 m cuando la verdad es ≈ 22 m (inliers cercanos: mediana 24; lejanos: 58).
- **Remedio (propuesta #1 del agente, consistencia con odometría)**: la odometría visual sabe que entre m y q
  el robot avanzó ~22 m en línea recta; un loop que dice 0.2 m es incompatible. Rechazar si
  ‖Δ_VO − Δ_loop‖ > α·camino(m, q) + β (tolerancia que crece con el drift, no afecta revisitas lejanas).
  Se agregó la pose integrada de la odometría al CSV por query y se re-corre p3_salad_c5 (`p3c5vo`) para
  tenerla; el filtro se aplica sobre las verificaciones ALIKED ya calculadas.

---

## 2026-09-30 23:30 — Pausa (PC apagada). Estado y cómo retomar

Resultados del filtro de odometría en FieldSAFE (`odo_filter.py`, solo para caminos < 100 m, tolerancia
0.2·camino + 1 m): no pierde ningún loop con pose correcta (16 270) y elimina 70 dudosos (todos los triviales).
Sin límite de camino descartaba vueltas completas de ~700 m (drift de la odometría ~25 %).

**Pendiente, en orden** (todo se cortó a mitad; las corridas incompletas se rehacen solas porque su
`_results.yml` quedó vacío):
1. `run_sld.py --config configs/p3_salad_c5.yaml --name p3c5vo --sessions rosario --jobs 5`
   (detección idéntica a p3_salad_c5 + pose de la odometría en el CSV; incluye 12-26 13:39).
2. `lg_venv/bin/python learned_verify.py p3c5vo --out p5_aliked --sessions ro_1226_1548,ro_1226_1339`
   (las otras 4 secuencias ya están verificadas en `runs/p5_aliked/*_verify.csv`).
3. `lg_venv/bin/python odo_filter.py p5_aliked --vo-run p3c5vo --out p6_odo --sessions rosario` y evaluar con
   `run_sld.py ... --name p6_odo --sessions rosario --eval-only` → ver si limpia el aliasing de 16:31.
4. Tabla final de validación Rosario (6 sec.): baseline vs p5_aliked vs p6_odo.
5. Extracción a 1280×720 (`pytorch-NetVlad-GPS/extract_rosario_fullres.sh`): hechas 13:14, 14:29, 16:31, 15:10;
   faltan 13:39 (se borró la parcial) y 15:48 — sacar del script las ya hechas antes de relanzar.
Lanzar con `setsid nohup ... &` (sobrevive a reinicios de la sesión).

## 2026-10-01 — Retomado

Validación de Rosario encadenada en `evaluation/chain_validation.sh`. Los reinicios de la sesión de Claude
matan también los procesos lanzados con `setsid` (cgroup), así que ahora las corridas largas se lanzan como
servicios de usuario: `systemd-run --user --unit=<nombre> ...` (seguimiento con `systemctl --user status`).

**Filtro de odometría en Rosario 12-22 14:29** (validación, parámetros fijados en FieldSAFE):

| Config | Loops | TP | ≥ 3 m | Precisión | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|---|---|
| baseline | 142 | 55 | 0 | 1.00 | 11.5 % | 4 | 0.20 m | 11.9° |
| p5_aliked | 999 | 941 | 56 | 0.94 | 84.6 % | 524 | 0.13 m | 8.7° |
| **p6_odo** | 941 | 941 | **0** | **1.00** | **84.6 %** | 524 | 0.13 m | 8.7° |

El filtro saca exactamente los 56 loops a ≥ 3 m (y 2 triviales) sin perder ningún TP. Pendiente: el yaw de
esta secuencia (mediana 8.7° también en el baseline) — revisar el GT de orientación de 14:29.

**Rosario 12-22 16:31 con filtro de odometría**:

| Config | Loops | TP | ≥ 3 m | Precisión | Cobertura | pose_ok |
|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 0 | — | 0 % | 0 |
| p5_aliked | 672 | 137 | 534 | 0.20 | 16 % | 0 |
| p6_odo | 228 | 137 | 91 | 0.60 | 16 % | 0 |

- El filtro saca 443 de 534 FP (los de camino corto, aliasing en el mismo surco) sin perder TP.
- Los 91 FP restantes están a ~450 m de camino (fuera del alcance del filtro: drift de la odometría) y
  estiman ‖t‖ ≈ 0.2 m. Se descartó que sea algo fijo a la cámara (p. ej. el emisor IR de la RealSense): en los
  pares falsos los matches se desplazan 5–27 px y < 6 % queda a < 2 px. Es **aliasing entre surcos**: otra
  hilera, misma imagen. Los TP de esta secuencia también tienen pose mala (e_vec 1.29 m): 16:31 es una
  secuencia degenerada para la verificación geométrica (sin estructura cercana distintiva).
- Limitación a reportar; posible solución: test de ambigüedad por desplazamiento de ±k espaciados de hilera
  (propuesta R2-6 del agente) o un prior de trayectoria con covarianza (ROVER).

**Rosario 12-22 13:14 y 12-26 15:10 con filtro de odometría**:

| Secuencia | Config | Loops | TP | ≥ 3 m | Precisión | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|---|---|---|
| 13:14 | baseline | 3490 | 3375 | 0 | 1.00 | 82 % | 3350 | 0.032 m | 0.6° |
| 13:14 | p5_aliked | 6775 | 6646 | 84 | 0.99 | 92 % | 6606 | 0.037 m | 1.1° |
| 13:14 | **p6_odo** | 6691 | 6646 | **2** | **1.00** | **92 %** | 6606 | 0.037 m | 1.1° |
| 15:10 | baseline | 11 | 11 | 0 | 1.00 | 19 % | 10 | 0.085 m | 1.5° |
| 15:10 | p5_aliked | 2624 | 2052 | 545 | 0.79 | 100 % | 1834 | 0.059 m | 3.3° |
| 15:10 | **p6_odo** | 2100 | 2052 | **46** | **0.98** | **100 %** | 1834 | 0.059 m | 3.3° |

**Rosario 12-26 13:39** (2.2 km): baseline 534 TP / cobertura 16.9 % / 0 FP; p5_aliked 1961 TP / 42.4 % / 1604 FP;
**p6_odo** 1961 TP / 42.4 % / 883 FP (precisión 0.69). El filtro saca 45 % de los FP; el resto es aliasing
entre surcos a largo camino, como en 16:31.

---

## 2026-10-01 — RESUMEN: train (FieldSAFE) y validación (Rosario, sin reajuste)

Pipeline final **p6_odo** = SLD con arreglos (K rectificada, disparidad > 0, exclusión de 20 m por odometría
visual estéreo) + recuperación DINOv2-SALAD (PCA ajustada en FieldSAFE) + verificación ALIKED + LightGlue con
PnP estéreo y rayos lejanos + filtro de consistencia con la odometría (caminos < 100 m).
Figuras: `figs/final_fieldsafe.png`, `figs/final_rosario.png`.

| Dataset | Config | Loops | TP (< 3 m) | Triviales | FP (≥ 3 m) | Precisión | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|---|---|---|---|
| FieldSAFE (5 ses.) | SLD original | 2716 | 65 | 2651 | 0 | 1.00 | 1.3 % | 65 | 0.20 m | 1.3° |
| FieldSAFE (5 ses.) | **p6_odo** | 16 878 | 7556 | 0 | 9322* | 0.45* | **74.0 %** | **16 270** | 0.14 m | 1.2° |
| Rosario (6 sec.) | SLD original | 4607 | 3975 | 632 | 0 | 1.00 | 27.6 % | 3892 | 0.04 m | 0.7° |
| Rosario (6 sec.) | **p6_odo** | 13 740 | **12 466** | 67 | 1207 | 0.91 | **59.5 %** | **11 556** | 0.07 m | 2.2° |

\* En FieldSAFE los loops a ≥ 3 m son la pasada vecina (3–10 m) con pose relativa correcta: 96 % de los loops
no triviales tienen e_vec < 1 m y e_yaw < 10°, ninguno está a > 10 m. Son restricciones válidas para SLAM.

Por secuencia de Rosario (cobertura SLD original → p6_odo): 13:14 82 → 92 %, 14:29 12 → 85 %, 15:10 19 → 100 %,
15:48 0 → 75 %, 13:39 17 → 42 %, 16:31 0 → 16 %.

**Conclusiones**
1. Las mejoras logradas en FieldSAFE se trasladan a Rosario sin reajuste: ×3.1 TP, ×3.0 loops con pose
   correcta y cobertura 28 % → 60 %.
2. Cambio de mayor impacto: verificación con features aprendidas (ALIKED + LightGlue). Recuperación con SALAD
   es condición necesaria (candidato correcto 7 % → 26 % en FieldSAFE).
3. Odometría visual estéreo, sin GPS: elimina los loops triviales (robot detenido) y, como filtro de
   consistencia, la mayoría del aliasing a lo largo del surco.
4. Limitación: aliasing entre surcos a largo camino en Rosario (16:31, 13:39; 1207 FP en total). Línea futura:
   test de ambigüedad por desplazamiento de hilera o prior de trayectoria con covarianza (ROVER).
5. Hallazgos colaterales: bug de intrínsecos en el PnP original (sesgo de 0.16 m), GT de FieldSAFE
   desincronizado 1.5 s, y el "15 cm" del paper original dominado por loops con el robot detenido.

**Pendiente**: Rosario a 1280×720 (5 de 6 extraídas); evaluar si la resolución completa mejora 16:31/13:39.

---

## 2026-10-01 — Rosario a 1280×720 (resolución nativa de las IR)

SALAD reutilizado (mismos frames; SALAD redimensiona igual). ALIKED a 1280×720: 0.19 s por candidato.

| Secuencia | Resolución | Loops | TP | FP | Precisión | Cobertura | pose_ok | e_vec med |
|---|---|---|---|---|---|---|---|---|
| 16:31 | 640×360 (p6_odo) | 228 | 137 | 91 | 0.60 | 16 % | 0 | 1.29 m |
| 16:31 | **1280×720** (p6_odo) | 198 | **155** | **43** | **0.78** | **28 %** | **32** | 1.21 m |

Con el doble de focal×baseline (16 → 32 px·m) baja a la mitad el aliasing que sobrevive y sube la cobertura,
pero la pose de esta secuencia sigue siendo mala (e_vec ~1.2 m).

| Secuencia | Resolución | Loops | TP | FP | Precisión | Cobertura | pose_ok | e_vec med |
|---|---|---|---|---|---|---|---|---|
| 13:39 | 640×360 (p6_odo) | 2866 | 1961 | 883 | 0.69 | 42 % | 1870 | 0.07 m |
| 13:39 | **1280×720** (p6_odo) | 2115 | **2067** | **15** | **0.99** | 32 % | **2062** | 0.07 m |

→ A resolución nativa el aliasing entre surcos de 13:39 prácticamente desaparece (883 → 15 FP), con más TP y
más loops con pose correcta; la cobertura baja (42 → 32 %). Se corre la resolución nativa en las 4 secuencias
restantes para la validación completa.

**Rosario completo a 1280×720** (p6_odo, sin reajuste):

| Secuencia | Loops | TP | FP | Precisión | Cobertura | pose_ok | e_vec med | e_yaw med |
|---|---|---|---|---|---|---|---|---|
| 13:14 | 7064 | 6924 | 8 | 0.999 | 92 % | 6897 | 0.037 m | 1.1° |
| 14:29 | 1265 | 1257 | 8 | 0.994 | 85 % | 824 | 0.12 m | 7.6° |
| 16:31 | 198 | 155 | 43 | 0.78 | 28 % | 32 | 1.21 m | 2.2° |
| 13:39 | 2115 | 2067 | 15 | 0.993 | 32 % | 2062 | 0.07 m | 2.0° |
| 15:10 | 2284 | 2284 | 0 | 1.000 | 100 % | 2070 | 0.06 m | 3.3° |
| 15:48 | 741 | 737 | 4 | 0.995 | 100 % | 730 | 0.05 m | 0.9° |
| **Total** | 13 667 | **13 424** | **78** | **0.994** | **60.0 %** | **12 615** | 0.07 m | 2.2° |

vs 640×360: 12 466 TP / 1207 FP / precisión 0.91 / cobertura 59.5 %. La resolución nativa elimina el 94 % de
los FP manteniendo la cobertura. En curso: SLD original a 1280×720 (comparación justa).

---

## 2026-10-02 — Evaluación por ATE (grafo de poses SE(2))

`ate_eval.py`: nodos = 1 de cada 5 frames; aristas de odometría = odometría visual estéreo (sin GPS); aristas de
loop = pose relativa estimada por cada loop (≤ 3000 por sesión, submuestreo uniforme); mínimos cuadrados L2
(una pérdida robusta descarta todos los loops cuando el drift es de decenas de metros); ATE = RMSE de
posición tras alinear con una transformación rígida 2D.

| FieldSAFE | Recorrido | Solo odometría | + loops SLD original | **+ loops p6_odo** |
|---|---|---|---|---|
| estática #1 | 2.1 km | 84.4 m | 84.4 m | **38.9 m (−54 %)** |
| dinámica #1 | 3.1 km | 24.6 m | 23.7 m | **11.1 m (−55 %)** |
| dinámica #2 | 4.2 km | 44.2 m | 44.2 m | **8.1 m (−82 %)** |
| 12:37 | 4.7 km | 46.7 m | 46.4 m | **7.7 m (−84 %)** |

Los loops del SLD original no cambian el ATE (triviales o concentrados en pocos lugares); los del pipeline
nuevo lo reducen entre 54 y 84 %.
