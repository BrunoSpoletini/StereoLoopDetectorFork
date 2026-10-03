# Investigación bibliográfica — mejoras para SLD

Estado: primera versión completa (2026-09-30). Registro del agente investigador; foco en el cuello de botella observado
(recuperación con aliasing perceptual en FieldSAFE, pocos inliers geométricos, cobertura despareja en Rosario).

## 1. Papers leídos

### 1.1 Verificación con prior de trayectoria / secuencias

**[P1] ROVER: Robust Loop Closure Verification with Trajectory Prior in Repetitive Environments** — J. Yu et al.,
2025, arXiv [2508.13488](https://arxiv.org/abs/2508.13488) (código: github.com/jarvisyjw/ROVER).
Para cada candidato ya verificado geométricamente, corre una PGO con ese loop, alinea (Sim3) la trayectoria
optimizada con la de odometría pura y puntúa el RMSE: un loop correcto deforma la trayectoria "suavemente", uno
falso la rompe. AP 99.3 % vs 95.4 % (verificación con LoFTR) y **recall@100 %P 87 % vs 51 %** en escenarios
repetitivos (hotel, depósito, escaleras).
*Relevancia*: SLD ya tiene odometría visual estéreo (`StereoOdometry.hpp`); con ella se puede reemplazar la
consistencia temporal "k=5 por índice" por una consistencia *métrica* con la odometría, que es lo que permite
relajar la verificación geométrica sin perder precisión.

**[P9] Probabilistic Appearance-Invariant Topometric Localization with New Place Awareness** — M. Xu, T. Fischer,
N. Sünderhauf, M. Milford, RA-L 2021, [arXiv 2107.07707](https://arxiv.org/abs/2107.07707).
Filtro bayesiano (tipo HMM) sobre los nodos del mapa: modelo de movimiento con odometría 3-DoF completa +
modelo de observación con similitud VPR + estado "fuera de mapa". Supera a SeqSLAM/matching de secuencias en
RobotCar tanto en loop closure como en localización global, justamente porque usa la odometría métrica.
*Relevancia*: reemplazo principista de la consistencia temporal k=5 (que pierde 71 % de las revisitas en
FieldSAFE). La observación puede ser la similitud SALAD sobre una ventana de candidatos.

**[P10] Pairwise Consistent Measurement Set Maximization (PCM)** — J. Mangelson, D. Dominic, R. Eustice,
R. Vasudevan, ICRA 2018; y **RRR** (Latif, Cadena, Neira, IJRR 2013). Aceptan loops sólo si son
*mutuamente consistentes* con la odometría: para dos loops (i→j) y (k→l), la composición
odom(j→l)·T_loop(l→k)·odom(k→i)·T_loop(i→j) debe ser ≈ identidad (distancia de Mahalanobis). PCM busca la
clique máxima de loops consistentes (verificado: PCM supera a DCS, SCGP y RANSAC; problema de clique
máxima resoluble en tiempo real). *Corrección (Ronda 2)*: RRR no es "PCM de a pares": agrupa los loops en
clusters temporales y acepta/rechaza clusters con tests χ² de consistencia intra- e inter-cluster contra la
odometría (IJRR 32(14), 2013; versión RSS 2012).
*Relevancia*: SLD ya devuelve pose relativa 6-DoF por loop y tiene VO → el chequeo por pares es barato y es
exactamente una "consistencia temporal métrica".

**[P11] Visual Loop Closure Detection Through Deep Graph Consensus (LoopGNN)** — M. Büchner, ..., A. Valada, 2025,
[arXiv 2505.21754](https://arxiv.org/abs/2505.21754). GNN sobre cliques de keyframes recuperados por VPR que
reemplaza la verificación RANSAC; más barato que la verificación geométrica clásica en TartanDrive 2.0 (off-road)
y NCLT. *Relevancia*: idea de "consenso de vecindario" (usar varios frames vecinos del candidato, no uno solo);
la red en sí requiere entrenamiento → baja prioridad.

### 1.2 VPR con descriptores globales

**[P2] BoQ: A Place is Worth a Bag of Learnable Queries** — A. Ali-bey, B. Chaib-draa, P. Giguère, CVPR 2024,
[arXiv 2405.07364](https://arxiv.org/abs/2405.07364), pesos públicos (github.com/amaralibey/Bag-of-Queries,
torch.hub, backbone DINOv2 y ResNet50). Consultas globales aprendidas que atienden (cross-attention) los tokens
del backbone. SOTA en 14 benchmarks urbanos.
*Relevancia*: candidato directo a comparar con SALAD en `global_desc.py` (misma interfaz torch.hub).

**[P3] MegaLoc: One Retrieval to Place Them All** — G. Berton, C. Masone, 2025, [arXiv 2502.17237](https://arxiv.org/abs/2502.17237),
`torch.hub.load("gmberton/MegaLoc","get_trained_model")`. DINOv2+SALAD entrenado sobre la unión de datasets de
VPR, landmark retrieval y SfM (más diversidad de puntos de vista que GSV-Cities, que es sólo urbano).
*Relevancia*: la variante más "generalista" disponible; probable mejora fuera de dominio respecto de SALAD.

**[P4] Cross-Modal Benchmarking for Robotic Perception in Natural Environments (WildCross)** — D. Hall, J. Knights,
M. Cox, P. Moghadam, ICRA 2026 WS, [arXiv 2606.11563](https://arxiv.org/abs/2606.11563).
Benchmark de VPR en bosque (476k frames). R@1 zero-shot: NetVLAD 39 %, MixVPR 47 %, SALAD 46 %, BoQ 46 %;
**fine-tuned en dominio: +12 a +19 puntos** (BoQ 63 %, SALAD 62 %). En urbano los mismos modelos superan 90 %.
*Relevancia*: evidencia de que (a) BoQ≈SALAD fuera de dominio, no hay "bala de plata"; (b) el fine-tuning en
dominio es la palanca grande. FieldSAFE+Rosario tienen GPS/GT → se pueden armar tuplas de entrenamiento.

**[P12] AnyLoc: Towards Universal Visual Place Recognition** — N. Keetha, A. Mishra, J. Karhade, K. M. Jatavallabhula, S. Scherer, M. Krishna, S. Garg, RA-L 2023 (presentado
en ICRA 2024),
[arXiv 2308.00688](https://arxiv.org/abs/2308.00688). DINOv2 ViT-G (capa 31, faceta *value*) + VLAD no supervisado
con vocabulario por dominio; fuerte en entornos no estructurados (aéreo, subacuático, cuevas). Pesos públicos.
*Relevancia*: la variante `dinov2_gem` de `global_desc.py` ya es un AnyLoc-GeM reducido; el VLAD con vocabulario
entrenado *sobre imágenes agrícolas* es la versión barata (sin backprop) de "adaptar al dominio".

**[P13] SelaVPR (ICLR 2024) / SelaVPR++ (2025)** — F. Lu et al., [arXiv 2402.14505](https://arxiv.org/abs/2402.14505),
[2502.16601](https://arxiv.org/abs/2502.16601). Adaptadores livianos sobre DINOv2 congelado: descriptor global +
features densas locales para re-ranking *sin* RANSAC. *Relevancia*: el re-ranking sin geometría no da pose
relativa; lo útil es la idea de adaptadores (fine-tuning barato con 8 GB de GPU).

**[P14] JIST (RA-L 2024) / SeqNet (RA-L 2021) / SeqMatchNet (CoRL 2021)** — Berton et al.; Garg & Milford.
Descriptores o matching a nivel de *secuencia* (SeqSLAM aprendido). JIST: descriptores 8× más chicos que
SeqVLAD y mejor recall en MSLS/RobotCar. *Relevancia*: en FieldSAFE la pasada vecina es igual a lo largo de
toda la secuencia, así que la secuencia sola no desambigua 3 m laterales; sirve para suavizar la recuperación
(ver P9, más apropiado porque usa odometría métrica).

**[P15] Multi-modal Loop Closure Detection with Foundation Models in Severely Unstructured Environments (MPRF)** —
L. Encinar González, J. Folkesson, R. Triebel, R. Giubilato, 2025, [arXiv 2511.05404](https://arxiv.org/abs/2511.05404).
DINOv2-SALAD para screening de candidatos + verificación 6-DoF explícita (LiDAR) en entornos planetarios de baja
textura. *Relevancia*: validación independiente de la arquitectura "SALAD recupera, geometría verifica" que SLD
acaba de adoptar en la Fase 3.

### 1.3 Matching local y verificación geométrica

**[P16] GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure
Detection** — J. Yu, H. Ye, J. Jiao, P. Tan, H. Zhang, IROS 2024, [arXiv 2407.11736](https://arxiv.org/abs/2407.11736).
Evalúa verificación geométrica de loops con AP y **MR = max recall @100 % precisión**. MR promedio: SP+SuperGlue
44 %, DISK+LightGlue 35 %, SP+NN 33 %, LoFTR 27 %, SIFT+NN 24 % (ORB no evaluado, sería peor que SIFT).
*Relevancia*: (a) cuantifica la ganancia esperable de features aprendidas en la verificación (~×1.8 en MR);
(b) define la métrica que falta en SLD (MR/AP barriendo un score de confianza).

**[P17] LightGlue** — P. Lindenberger, P.-E. Sarlin, M. Pollefeys, ICCV 2023 (github.com/cvg/LightGlue);
**XFeat** — G. Potje et al., CVPR 2024 (tiempo real en CPU, 5× más rápido que otras features aprendidas, con
LighterGlue); **Efficient LoFTR** — Y. Wang et al., CVPR 2024: 40 ms FP32 / 25 ms mixto vs SP+LG 56 ms (RTX 3090,
Aachen). *Relevancia*: SP/ALIKED/XFeat + LightGlue es tiempo real en la RTX 2070 (≈20-60 ms por par); exportable
a ONNX (repo LightGlue-ONNX) para llamarlo desde C++. En stitching agrícola terrestre (AgRowStitch,
[arXiv 2503.21990](https://arxiv.org/abs/2503.21990)) SP+LightGlue es lo que funciona sobre hojas repetitivas.

**[P18] MASt3R-SLAM** — R. Murai, E. Dexheimer, A. Davison, CVPR 2025, [arXiv 2412.12392](https://arxiv.org/abs/2412.12392).
Matching denso 3D-consciente (MASt3R) + recuperación ASMK incremental; 15 FPS en total en GPU de escritorio.
*Relevancia*: el entorno conda `mast3r-slam` ya existe; útil como **oráculo offline** para saber cuántas
revisitas son verificables en principio (cota superior del matching), no como componente de tiempo real.

**[P19] Flow Separation for Fast and Robust Stereo Odometry** — M. Kaess, K. Ni, F. Dellaert, ICRA 2009.
En exteriores, puntos lejanos restringen sólo la rotación (RANSAC de 2 puntos) y los cercanos la traslación
(RANSAC de 1 punto); evita la inconsistencia de RANSAC-3pt en datos casi degenerados. Relacionado: solvers
mínimos de movimiento plano (Choi & Kim, IVC 2018; Scaramuzza 1-point, IJCV 2011).
*Relevancia*: ataca directo "pocos inliers cercanos + horizonte lejano" de FieldSAFE; hoy los puntos lejanos
se descartan (`far_distance`) o entran a un EPnP 6-DoF mal condicionado.

*Verificado (Ronda 2)*: Choi & Kim, "Fast and reliable minimal relative pose estimation under planar motion",
Image and Vision Computing 69:103-112, 2018 (≈9× más rápido que el solver plano previo); Scaramuzza, "1-Point-RANSAC
SfM for vehicle-mounted cameras by exploiting non-holonomic constraints", IJCV 95(1):74-85, 2011.

**[P20] OverlapNet** — X. Chen, T. Läbe et al. (PRBonn), RSS 2020 / Autonomous Robots 2021. *Corrección (Ronda 2)*:
es un método **LiDAR** (red siamesa sobre range images que predice solapamiento y yaw relativo); el "solapamiento"
como GT se define allí sobre range images. Para cámara, el análogo es la co-visibilidad por reproyección con
poses GT (usado p.ej. en benchmarks recientes de loop closure visual). Se mantiene la idea: protocolos que definen el GT de loop por
**solapamiento de campo visual** (reproyección con poses GT) en lugar de umbral de distancia.
*Relevancia*: con cámara frontal y pasadas a 3-5 m, "<3 m" es un GT inadecuado: pares a 4 m con solapamiento
alto son loops válidos si la pose relativa estimada es correcta.

### 1.4 Agricultura

**[P5] Addressing the challenges of loop detection in agricultural environments (SLD)** — Soncini, Civera, Pire,
JFR 42(1) 2025, [arXiv 2408.15761](https://arxiv.org/abs/2408.15761). Sistema base. Argumenta que los descriptores
globales fallan por aliasing y prefiere features locales + verificación estéreo; error mediano 15 cm.

**[P6] The Rosario Dataset v2** — Soncini, Cremona, Vidal, García, Castro, Pire, 2025, [arXiv 2508.21635](https://arxiv.org/abs/2508.21635).
ORB-SLAM3 con loops habilitados cierra 7 loops falsos y corrompe la trayectoria (errores de rotación de hasta
170°); por eso el benchmark se corre con loops deshabilitados.
*Relevancia*: justifica que la precisión debe seguir ~100 % (un FP destruye el mapa) → cualquier relajación de
recall necesita una verificación extra (métrica/odometría), no sólo más inliers.

**[P7] Keypoint Semantic Integration for Improved Feature Matching in Outdoor Agricultural Environments** —
R. de Silva, ..., C. Cadena, C. Stachniss, R. Polvara, RA-L 2025. Enriquece descriptores de keypoints con un
embedding de la máscara semántica (troncos de vid) para romper aliasing; mejora matching para todos los
detectores probados, sobre varios meses.
*Relevancia*: baja para FieldSAFE (pasto sin instancias semánticas); moderada para Rosario (plantas de soja).

**[P8] Semantic Landmark Particle Filter for Robot Localisation in Vineyards** — R. de Silva et al., 2026 (IROS
sub.), [arXiv 2603.10847](https://arxiv.org/abs/2603.10847). El aliasing es *a nivel de hilera*: hileras paralelas
dan observaciones casi idénticas; resuelven con landmarks semánticos + GNSS como prior.
*Relevancia*: confirma que el error típico es "la hilera/pasada vecina" (igual que el top-1 de SALAD a 3-5 m en
FieldSAFE); un prior métrico (odometría/GNSS) es la herramienta estándar para desambiguarlo.

**[P21] PointNetPGAP-SLC** — T. Barros et al., RA-L 2024, [arXiv 2405.19038](https://arxiv.org/abs/2405.19038).
LiDAR PR en huertos/frutillas (HORTO-3DLM); pérdida auxiliar de "consistencia por segmento" (hilera) en el
entrenamiento para separar hileras adyacentes casi idénticas. *Relevancia*: idea transferible al fine-tuning de
SALAD: usar la pasada vecina como *hard negative* (o, mejor, no penalizarla y dejar la separación a la geometría).

## 2. Propuestas de mejora (ranking)

Criterio de orden: (ganancia esperada en recall/cobertura sobre el cuello de botella medido) / (costo), sin
sacrificar la precisión ~100 % (P6: un FP corrompe el mapa).

### #1 Seguimiento del loop guiado por odometría ("loop tracking") + consistencia temporal métrica
- **Idea**: (a) reemplazar la consistencia k=5 "por índice de isla" por una consistencia *métrica*: una hipótesis
  de loop (i↔j, T_ij) predice la del frame siguiente como T_{i+1,j'} = odom(i→i+1)⁻¹·T_ij·odom(j→j'), con j' el
  frame viejo más cercano a la pose predicha. Se acepta cuando ≥ m hipótesis sucesivas verificadas son
  consistentes (error < ~0.5 m / 5°, al estilo PCM). (b) Una vez aceptado un loop, **no volver a recuperar**:
  verificar directamente contra j' (el frame viejo predicho) en cada frame nuevo, mientras la verificación
  estéreo siga pasando. Así un único loop "semilla" cubre todo el tramo revisitado.
- **Evidencia**: PCM/RRR [P10] (consistencia por pares con odometría), ROVER [P1] (MR 87 % vs 51 % usando prior de
  trayectoria), filtro topométrico con odometría [P9]; ORB-SLAM3 hace lo análogo tras relocalizar.
- **Problema que ataca**: 71 % de revisitas perdidas en consistencia temporal (FieldSAFE), cobertura 1-3 %, y
  cobertura despareja en Rosario (secuencias con pocos TP aislados). La consistencia k=5 exige que BoW/SALAD
  acierte 5 veces seguidas, cuando el candidato correcto aparece ~8 %/39 % de las veces.
- **Implementación**:
  1. `demo/StereoOdometry.hpp`: exponer en `Result` la pose acumulada SE(3) del frame (T_w_c, ya se compone
     internamente entre keyframes) además de `odometer`.
  2. `TemplatedLoopDetector.h`: nuevo parámetro `metric_consistency` (bool) y `detectLoop(..., const cv::Matx44d&
     pose)`; guardar `m_poses` junto a `m_odometers`. Reemplazar `updateTemporalWindow/getConsistentEntries`
     (líneas ~940-946) por un buffer de hipótesis verificadas {query, match, T} y la función
     `bool isMetricallyConsistent(hyp_prev, hyp_cur)`.
  3. Estado `m_tracking` {old_id, T}: si está activo, antes de la recuperación llamar
     `isGeometricallyConsistent_Exhaustive_Stereo` sobre j' (y j'±1..2); si falla N frames seguidos, desactivar.
  4. Flags en `configs/*.yaml` y lectura en `demoDetectorStereo.h` (~línea 300).
- **Costo**: medio (1-2 días). La geometría ya existe; el trabajo es el bookkeeping de poses.
- **Riesgo**: la deriva de VO entre i y j' es chica en tramos cortos (se compara odometría *local*, no global),
  pero en giros cerrados de cabecera la VO puede fallar (`tracked=false`) → cortar el tracking ahí. Riesgo de
  propagar un FP a lo largo de un tramo: mitigado exigiendo verificación geométrica en cada paso.
- **Medición**: cobertura (métrica principal que debería subir más), TP, FP/precisión, e_vec/e_yaw; embudo
  (`funnel.py`) con una nueva etapa "aceptado por tracking".

### #2 Redefinir el GT y reportar MR/AP (evaluación)
- **Idea**: (a) considerar TP a todo loop cuya **pose relativa estimada coincide con la GT** (p.ej. e_vec < 0.5 m y
  e_yaw < 5°) y con solapamiento visual (distancia < ~8-10 m y |Δyaw| < ~30°), no sólo los a < 3 m; un loop con la
  pasada vecina a 4 m y pose correcta es una restricción válida para el grafo de poses. FP = pose incorrecta.
  (b) Agregar a cada detección un score de confianza (inliers PnP, similitud SALAD) y reportar **AP y MR (recall
  @100 % precisión)** barriendo ese score, como GV-Bench/ROVER.
- **Evidencia**: GT por solapamiento [P20]; MR/AP en [P16] y [P1]. Dato propio: el top-1 de SALAD está a < 5 m el
  80 % de las veces — muchos "incorrectos" son loops métricamente útiles.
- **Problema**: el criterio actual subestima el recall de la recuperación global y castiga candidatos que la
  verificación estéreo puede resolver bien; además un único punto operativo impide comparar configuraciones.
- **Implementación**: `evaluation/sld_eval.py` (`evaluate`, `pose_errors` ya calcula e_vec/e_yaw por loop): nueva
  clase de resultado "TP métrico" y curva PR con `geom_inliers` como score (ya se escribe en el CSV de
  `demoDetectorStereo.h`, línea ~434). Mantener también la métrica vieja para comparar con el paper original.
- **Costo**: bajo (horas). **Riesgo**: nulo en el sistema; hay que justificar el umbral de solapamiento en la tesina.
- **Medición**: es la medición; recalcular todas las corridas existentes (runs/) sin re-ejecutar SLD.

### #3 Verificación con features aprendidas (SuperPoint/ALIKED/XFeat + LightGlue)
- **Idea**: en la verificación estéreo, reemplazar ORB + fuerza bruta + ratio por keypoints aprendidos + LightGlue,
  tanto para el match L-R del par viejo como para viejo-L ↔ actual-L. Mantener triangulación + PnP.
- **Evidencia**: GV-Bench [P16]: MR 44 % (SP+SG) vs 24 % (SIFT) en verificación de loops; LightGlue/XFeat tiempo real
  [P17]; SP+LG funciona en imágenes de cultivos repetitivos (AgRowStitch). ORB en pasto da pocos matches correctos.
- **Problema**: 25-35 % de revisitas perdidas en geometría; loops reales con pocos inliers (RANSAC ≥20 descarta 45/61
  TP); relajar ORB (ratio 0.8, sin chequeo cruzado) aumentó TP pero empeoró la pose (0.65 m, 8.7°) → faltan
  matches *buenos*, no umbrales más laxos.
- **Implementación**: dos etapas.
  1. Experimento offline en Python (entorno `mast3r-slam`, torch): script nuevo en `evaluation/` que toma los pares
     (query, candidato) que SLD logueó como `NO_GEOMETRICAL_CONSISTENCY` o que SALAD propone, corre SP+LG (o
     ALIKED+LG), triangula con P rectificada y hace `cv2.solvePnPRansac`; comparar #inliers y error de pose vs GT
     con la versión ORB. Decide si vale la pena integrar.
  2. Integración: precomputar keypoints+descriptores por frame (como `global_desc.py` con los `.npy`) y hacer el
     matching con LightGlue-ONNX desde C++ (ONNX Runtime) en una nueva rama `GEOM_LEARNED_STEREO` de
     `isGeometricallyConsistent_Exhaustive_Stereo`; o, más simple, resolver la verificación entera en Python.
- **Costo**: offline 1 día; integración C++ 2-4 días. **Riesgo**: SuperPoint entrenado en MegaDepth/COCO puede
  detectar poco en pasto uniforme (probar ALIKED/DISK/XFeat); tiempo en la 2070 ~30-60 ms por par (OK a 10 Hz
  para 1-3 candidatos).
- **Medición**: #inliers en revisitas verdaderas, fracción de revisitas que pasan geometría (embudo), e_vec/e_yaw,
  MR.

### #4 PnP contra un mapa local multi-frame del lugar viejo + estimación desacoplada rotación/traslación
- **Idea**: (a) en vez de triangular sólo el par estéreo `cand`, acumular puntos 3D de los frames cand-n..cand+n
  (expresados en el frame `cand` con las poses de VO) y hacer PnP contra ese mapa local; de ser útil, también
  usar 2-3 frames actuales. Más puntos cercanos y mejor distribución espacial. (b) Estimar rotación con los puntos
  lejanos (los que hoy se descartan por `far_distance`/disparidad baja; bearing-only, RANSAC de 2 puntos) y
  traslación con los cercanos (1 punto dada R), o PnP plano de 3 DoF (x, z, yaw) cuando el terreno es plano, y
  luego refinar en 6-DoF.
- **Evidencia**: flow separation [P19] y solvers de movimiento plano (menos DoF → menos inliers necesarios,
  RANSAC más estable); ORB-SLAM3 verifica loops contra ventanas de keyframes covisibles, no contra un único frame.
- **Problema**: pocos inliers y horizonte lejano (FieldSAFE); baseline de 5 cm en Rosario (triangulación útil sólo
  a pocos metros).
- **Implementación**: `isGeometricallyConsistent_Exhaustive_Stereo` (TemplatedLoopDetector.h ~1560-1675): separar
  `good_3d_points` en cercanos/lejanos por disparidad; función nueva `estimateRelPoseDecoupled(...)`. El mapa
  local requiere las poses de VO por frame (mismo cambio que #1 en `StereoOdometry.hpp`) y guardar keypoints del
  vecindario (ya se guardan `m_image_keys/descriptors[2]`).
- **Costo**: (b) 1-2 días; (a) 2-3 días. **Riesgo**: deriva de VO dentro de la ventana (pequeña para ±1-2 m);
  terreno no plano en FieldSAFE → usar 3-DoF sólo como inicialización.
- **Medición**: e_vec/e_yaw (debería bajar yaw), fracción de revisitas que pasan geometría con `min_pnp_inliers`
  estricto, ‖t‖ con robot quieto (que debe seguir ≈ 1 mm).

### #5 Mejor descriptor global: comparar MegaLoc/BoQ y adaptar al dominio
- **Idea**: (a) comparar zero-shot SALAD vs MegaLoc vs BoQ-DINOv2 vs AnyLoc-VLAD en R@1/R@10 con
  `retrieval_eval.py` (misma interfaz torch.hub). (b) Fine-tuning de SALAD (o sólo la cabeza SALAD/adaptadores,
  cabe en 8 GB) sobre FieldSAFE con positivos por GPS/RTK (< 3-5 m, mismo rumbo) y negativos > 25 m; validar en
  Rosario. Alternativa sin backprop: vocabulario VLAD de AnyLoc entrenado sobre imágenes agrícolas.
- **Evidencia**: WildCross [P4]: fine-tuning en dominio natural +12-19 puntos de R@1 para todos los métodos, y
  BoQ≈SALAD≈MixVPR zero-shot; MegaLoc [P3] el más generalista; PointNetPGAP-SLC [P21] para el manejo de hileras
  adyacentes; adaptadores SelaVPR [P13].
- **Problema**: recuperación (candidato BoW correcto 8 % en FieldSAFE; SALAD R@1 39 %).
- **Implementación**: `evaluation/global_desc.py::load_model` (agregar `megaloc`, `boq`); script de fine-tuning
  aparte. El detector no cambia (usa `global_desc` .npy). Cuidar la partición: FieldSAFE train / Rosario val.
- **Costo**: (a) horas; (b) 2-4 días. **Riesgo**: (b) sobreajuste a FieldSAFE (un solo campo, un día) → la
  validación en Rosario es imprescindible. Ganancia de (a) probablemente chica (WildCross).
- **Medición**: R@1/R@5/R@10 a 3 m y a "pose verificable" (#2), luego TP/cobertura end-to-end.

### #6 Filtro bayesiano de hipótesis de loop sobre la similitud global
- **Idea**: filtro discreto (HMM) sobre los frames viejos: predicción con la odometría (desplaza la creencia según
  la distancia recorrida en la VO), actualización con la similitud SALAD de toda la base, estado "lugar nuevo".
  Verificar geométricamente sólo el máximo de la creencia cuando supera un umbral.
- **Evidencia**: [P9] (supera a SeqSLAM y a VPR frame-a-frame en RobotCar); secuencias [P14].
- **Problema**: aliasing en recuperación y consistencia temporal frágil; integra evidencia débil a lo largo del
  tiempo (R@10 = 81 % → la respuesta correcta está casi siempre en el top-10).
- **Implementación**: bloque `if (m_params.global_retrieval)` en `detectLoop` (~línea 863): mantener un vector de
  creencia sobre entradas (o sobre "tramos" de 1 m de odómetro para acotar el tamaño) y reemplazar el ranking
  por similitud por el MAP del filtro. Solapa con #1: #1 es la versión "dura" (una hipótesis), esta es la
  "blanda" (multihipótesis) para la *adquisición* del primer loop.
- **Costo**: medio (2-3 días). **Riesgo**: ajuste de hiperparámetros (sensor model, probabilidad de lugar nuevo);
  en campo con pasadas paralelas la creencia puede quedar bimodal entre pasadas vecinas (lo resuelve la geometría).
- **Medición**: fracción de frames revisita con candidato correcto (embudo), cobertura, precisión.

### #7 Score de confianza de loop y reporte de incertidumbre de pose
- **Idea**: exportar, por loop, #inliers, distribución espacial de inliers (cobertura de la imagen, fracción
  cercanos) y la covarianza del PnP (Jacobiano de `solvePnPRefineLM`), para usar como score de la curva PR (#2) y
  como información para un back-end (PGO robusto tipo GNC/switchable constraints).
- **Evidencia**: [P16] (MR requiere un score), [P1]/[P10] (consistencia requiere covarianza).
- **Costo**: bajo. **Riesgo**: nulo. **Medición**: AP/MR; correlación score ↔ e_vec.

## 3. Ideas descartadas

- **Re-ranking sin geometría (SelaVPR, R2Former, LoopGNN)**: no devuelven pose relativa y requieren entrenamiento;
  SLD ya hace verificación geométrica que es más informativa.
- **Matching denso (MASt3R, RoMa, LoFTR) en línea**: 100-300 ms por par y 8 GB justos; sólo como oráculo offline
  para acotar cuántas revisitas son verificables (ver P18).
- **Enriquecimiento semántico de descriptores (KSI, P7) y landmarks semánticos (P8)**: requieren instancias
  semánticas (troncos, postes) que el pasto de FieldSAFE no tiene; podría retomarse para plantas de soja en Rosario.
- **Descriptores de secuencia aprendidos (SeqNet/JIST)**: no resuelven el aliasing entre pasadas paralelas (toda
  la secuencia vecina es parecida); el filtro con odometría (#6) aprovecha mejor la secuencia.
- **Discriminar la pasada vecina (3-5 m laterales) sólo con el descriptor global**: problema mal planteado en campo
  abierto frontal; mejor dejarlo a la geometría o aceptarlo como loop válido (#2).
- **Revisitas en sentido opuesto**: con cámara frontal el solapamiento es casi nulo (WildCross reporta lo mismo
  para "reverse revisits"); mantener el filtro de mismo sentido en el GT.
- **Métodos LiDAR (MPRF-SONATA, PointNetPGAP, AgriGS-SLAM)**: sensor fuera del alcance de SLD (estéreo pura).
- **Seguir relajando umbrales de ORB (ratio, chequeo cruzado, inliers)**: la Fase 2 ya mostró que sube TP a costa
  de la pose y sin mover el recall (< 1 %).

---

# Ronda 2 (2026-09-30)

## R2.0 Verificación de citas dadas de memoria en la Ronda 1

| Cita | Estado |
|---|---|
| PCM (Mangelson, Dominic, Eustice, Vasudevan, ICRA 2018) | Correcta. |
| RRR (Latif, Cadena, Neira, IJRR 32(14) 2013) | Referencia correcta; **descripción corregida** en P10 (clusters temporales + tests χ², no pares). |
| OverlapNet (Chen et al., RSS 2020 / AURO 2021) | **Corregido** en P20: es LiDAR; el GT por co-visibilidad para cámara es un análogo, no lo que hace el paper. |
| AnyLoc (Keetha et al.) | **Corregido**: RA-L 2023, presentado en ICRA 2024. ViT-G/14, capa 31, faceta *value*, VLAD 32 clusters. |
| SeqNet (Garg & Milford, RA-L 2021) | Correcta (TCN 1-D sobre secuencias de descriptores, jerárquico). |
| SeqMatchNet (Garg, Vankadari, Milford) | CoRL 2021 (PMLR 164, publicado 2022). Triplet loss con distancia de *sequence matching*. |
| JIST (Berton, Trivigno, Caputo, Masone) | Correcta: RA-L 9(2):1310-1317, 2024, [arXiv 2403.19787](https://arxiv.org/abs/2403.19787); agregador SeqGeM. |
| Kaess, Ni, Dellaert, ICRA 2009 | Correcta: rotación con puntos lejanos (RANSAC 2 pt), traslación con cercanos (RANSAC 1 pt). |
| Choi & Kim IVC 2018; Scaramuzza IJCV 2011 | Correctas (ver P19). |

## R2.1 Observaciones propias sobre los datos (no son experimentos de SLD)

Mirando imágenes crudas y las calibraciones (`resources/*_stereo_parameters.yaml`):

1. **FieldSAFE: el propio tractor ocupa ~15-20 % de la imagen izquierda** (rueda delantera, capó con texto, y su
   sombra junto a la del mástil del sensor), además de *flares* del sol. Es contenido idéntico en todas las
   imágenes. Un chequeo rápido con ORB entre 15 pares de frames aleatorios no encontró matches "estáticos"
   (desplazamiento < 4 px) con distancia Hamming < 40, así que el daño probable no está en la geometría sino en
   **BoW y SALAD**: una fracción fija de palabras/tokens es común a todos los frames y reduce la
   discriminabilidad (sube el "piso" de similitud de todo el mapa). También se ven claramente los **carriles
   cortados / sin cortar** (bordes rectos, paralelos): estructura explotable (R2-6).
2. **Rosario: f·B = 16.2 px·m** (f=323.5 px a 640×360, B=5 cm; las imágenes extraídas son 640×360 aunque el
   paper del dataset reporta 1280×720). Con σ_d = 0.5 px, σ_Z = Z²σ_d/(fB): 0.28 m a 3 m, 0.77 m a 5 m, 3.1 m a
   10 m. **FieldSAFE: f·B = 121.7 px·m** → σ_Z = 0.41 m a 10 m, 1.6 m a 20 m. Con `far_distance = 50 m` y
   `stereo_min_disparity = 0.5`, en Rosario entran al PnP puntos con profundidad casi arbitraria: la
   triangulación sólo es confiable a < ~3-4 m (σ_Z/Z < 10 % ⇔ Z < 0.1·fB/σ_d ≈ 3.2 m Rosario, 24 m FieldSAFE).
   Esto explica en parte por qué Rosario rinde tan desparejo y por qué la pose depende de pocos puntos cercanos.
3. En Rosario aparecen personas/vehículos adelante en el surco (objeto dinámico) y la línea de horizonte con
   árboles lejanos es visible en IR.

## R2.2 Papers nuevos

**[P22] SCENES: Subpixel Correspondence Estimation With Epipolar Supervision** — D. Kloepfer, J. Henriques,
D. Campbell, 2024, [arXiv 2401.10886](https://arxiv.org/abs/2401.10886). Fine-tuning de matchers existentes en un
dominio nuevo usando sólo poses (de odometría) con una pérdida epipolar, sin profundidad ni correspondencias GT;
incluye bootstrapping cuando no hay poses. *Relevancia*: permite adaptar SP+LightGlue (o XFeat) a pasto/soja con
la VO estéreo de SLD o el GT GPS como supervisión.

**[P23] RoMa: Robust Dense Feature Matching** — J. Edstedt et al., CVPR 2024, [arXiv 2305.15404](https://arxiv.org/abs/2305.15404).
Usa features **DINOv2 congeladas** para el matching grueso (muy robustas pero gruesas, parches de 14 px) + CNN
fina; +36 % mAA en WxBS. *Relevancia*: los tokens DINOv2 ya se calculan para SALAD → se pueden reutilizar como
matching grueso/guía de la verificación (R2-3) casi gratis.

**[P24] Reloc3r** — S. Dong et al., CVPR 2025, [arXiv 2412.08376](https://arxiv.org/abs/2412.08376). Regresión de
pose relativa entrenada en ~8 M pares; 24 FPS a 512 px. **VGGT** (Wang et al., CVPR 2025 best paper): reconstrucción
feed-forward de cámaras/profundidad/tracks en < 1 s. **MASt3R/DUSt3R** (P18). *Relevancia*: oráculos o
*teachers* offline; como verificador en línea son peligrosos (regresan una pose aunque no haya solapamiento).

**[P25] FoundationStereo** — B. Wen et al. (NVIDIA), CVPR 2025, [arXiv 2501.09898](https://arxiv.org/abs/2501.09898);
**MonSter** (2025, [arXiv 2501.08643](https://arxiv.org/abs/2501.08643)); **Depth Anything V2 / UniDepthV2 /
Metric3D v2**. Estéreo zero-shot con priors monoculares (DepthAnythingV2 congelado dentro de FoundationStereo) y
fusión mono+estéreo. *Relevancia*: extender el rango útil del estéreo de 5 cm de Rosario (R2-5).

**[P26] Closed-Form Solutions to Minimal Absolute Pose Problems with Known Vertical Direction** — Z. Kukelova,
M. Bujnak, T. Pajdla, ACCV 2010. P2P: pose absoluta con **2 correspondencias 2D-3D** si se conoce la vertical
(IMU o punto de fuga). *Relevancia*: ambos datasets tienen IMU; con la vertical conocida el PnP pasa de 6 a 4 DoF →
RANSAC mucho más estable con pocos inliers (R2-2).

**[P27] Robust Long-Term Registration of UAV Images of Crop Fields for Precision Agriculture** — N. Chebrolu,
T. Läbe, C. Stachniss, RA-L 2018. Registra imágenes a lo largo de la temporada usando la **geometría de la
disposición de plantas** (estática) en vez de la apariencia (cambiante). *Relevancia*: en Rosario, la posición de
plantas/huecos a lo largo del surco es una "huella" métrica (R2-6).

**[P28] HD Ground / Micro-GPS** — J. F. Schmid et al., ICRA 2022; L. Zhang et al., ICRA 2019
([arXiv 1710.10687](https://arxiv.org/abs/1710.10687)). Localización con cámara hacia el suelo, precisión
milimétrica; evaluado en asfalto, adoquín, alfombra, laminado (no pasto ni suelo agrícola) y con degradación en
superficies mojadas. *Relevancia*: respalda la idea de "huella de suelo" pero no en vegetación; alto riesgo (R2-12).

**[P29] Skyline-based localisation for aggressively manoeuvring robots using UV sensors and spherical harmonics**
— Stone, Mangan, Wystrach, Webb (U. Edinburgh), ICRA 2016. Segmentación cielo/suelo como firma de lugar
invariante a iluminación y (con armónicos esféricos) a rotación. *Relevancia*: en campo abierto la línea de árboles
del horizonte es el único contenido estable y lejano → brújula visual (R2-7). (Autores citados de memoria; título y
afiliación verificados.)

**[P30] SMART / SeqSLAM con odometría** — Pepperell, Corke, Milford (ICRA 2014) y fusión VO+SeqSLAM (Sensors
2018): la odometría resuelve la variabilidad de velocidad en el matching de secuencias; error de posición 0.22 %
vs 0.45 % con VO sola. *Relevancia*: refuerza #6 (filtro con odometría) frente a SeqSLAM de velocidad constante.

**[P31] DARE-SLAM** — K. Ebadi et al., J. Intell. Robot. Syst. 2021, [arXiv 2102.05117](https://arxiv.org/abs/2102.05117).
Loop closing consciente de degeneración en entornos subterráneos perceptualmente degradados. *Relevancia*:
analogía para detectar geometría degenerada (puntos todos lejanos o todos en el plano del suelo) y rechazar o
restringir la pose en vez de aceptar un PnP 6-DoF mal condicionado.

## R2.3 Ideas nuevas

### R2-1 Máscara del ego-vehículo, sombra propia y flares (FieldSAFE; objetos dinámicos en Rosario)
- **Idea**: máscara binaria fija por dataset (rueda+capó+mástil del tractor en FieldSAFE; borde inferior si
  aparece el robot en Rosario) aplicada a la detección ORB, al vocabulario BoW, a la VO y a los tokens de
  DINOv2 antes de SALAD (poner a cero/ignorar esos parches, o recortar la imagen). Opcional: máscara de sombra
  propia y de saturación (flares).
- **Evidencia**: práctica estándar en SLAM vehicular (máscara de capó); observación R2.1-1. Sin paper específico:
  es higiene de datos.
- **Problema**: aliasing en recuperación (BoW correcto 8 %): palabras/tokens comunes a todo el mapa.
- **Implementación**: `demo/demoDetectorStereo.h` (pasar `mask` a `ORB::detectAndCompute`; parámetro `mask_image`
  en el YAML), `StereoOdometry.hpp` igual; `global_desc.py`: recortar o enmascarar la imagen antes del modelo.
  La máscara se genera una vez (desvío estándar por píxel sobre N frames + edición manual).
- **Costo**: muy bajo (horas). **Riesgo**: bajo; la rueda gira con la dirección → máscara generosa.
- **Medición**: R@1/R@10 de SALAD y de BoW (`retrieval_eval.py`), candidato BoW correcto en el embudo, TP.

### R2-2 Pose relativa robusta con baseline chico: vertical conocida, matriz esencial + escala estéreo, y filtrado por incertidumbre de profundidad
- **Idea** (tres piezas combinables, en orden de costo):
  (a) reemplazar `far_distance` fijo por un umbral de **incertidumbre de profundidad** (σ_Z/Z < 10-15 %, o sea Z <
  ~3 m en Rosario y ~24 m en FieldSAFE) y ponderar residuos del PnP por 1/σ;
  (b) estimar R y la dirección de t con la **matriz esencial 2D-2D** (`cv::findEssentialMat` + `recoverPose`) usando
  *todos* los matches L-L, incluidos los lejanos sin profundidad útil, y fijar sólo la **escala** con los pocos
  puntos estéreo cercanos (RANSAC de 1 punto sobre la escala);
  (c) si se usa la IMU: vertical conocida → P2P (4 DoF) de Kukelova et al.
- **Evidencia**: P19 (separar lejanos/cercanos), P26 (P2P), solvers planos (Choi & Kim), cálculo propio de f·B.
- **Problema**: pocos inliers (RANSAC ≥20 descarta 45/61 TP), pose peor al relajar, Rosario desparejo.
- **Implementación**: en `isGeometricallyConsistent_Exhaustive_Stereo` (`TemplatedLoopDetector.h` ~1600-1675):
  (a) filtro por disparidad mínima dependiente de fB (nuevo parámetro `max_depth_rel_error`); (b) nueva rama
  `pose_mode: essential_scale` que no descarta matches sin profundidad; (c) requiere leer la IMU en
  `demoDetectorStereo.h` (más costo; dejar para después).
- **Costo**: (a) horas; (b) 1-2 días; (c) 2-3 días. **Riesgo**: matriz esencial degenerada con traslación casi
  nula (robot quieto; ya filtrado por exclusión con VO) o escena plana (usar homografía como alternativa, test
  tipo ORB-SLAM de H vs F).
- **Medición**: e_vec/e_yaw, fracción de revisitas que pasan geometría con umbrales estrictos, ‖t‖ con robot
  quieto; separar resultados FieldSAFE vs Rosario.

### R2-3 Reutilizar los tokens DINOv2 de SALAD como matching grueso para la verificación
- **Idea**: guardar además del descriptor global el mapa de tokens de parche (reducido con PCA a 64-128 D); en la
  verificación, emparejar parches por mutual-nearest-neighbour entre query y candidato, y usar esos pares
  (i) como correspondencias gruesas para un PnP inicial o (ii) para restringir el matching ORB/LightGlue a
  keypoints dentro de parches correspondientes (anula el aliasing local de hojas/pasto, porque el contexto del
  parche de 14 px·ViT es más amplio).
- **Evidencia**: RoMa (P23) usa exactamente DINOv2 congelado como matcher grueso robusto; AnyLoc (P12) muestra que
  los tokens son descriptivos en entornos no estructurados.
- **Implementación**: `global_desc.py` guarda `x_norm_patchtokens` (float16, p.ej. 37×73 tokens a 518 px ≈ 2.7k ×
  128 D por imagen → ~0.7 MB/imagen: guardar sólo para un subconjunto o recalcular a demanda en Python). Prueba
  offline primero, como la #3.
- **Costo**: medio. **Riesgo**: resolución gruesa; el almacenamiento para miles de frames.
- **Medición**: #inliers y e_vec en pares revisita; comparación con ORB y SP+LG.

### R2-4 Adaptación autosupervisada al dominio (VPR y matching) con la propia odometría/GPS; destilación
- **Idea**: (a) fine-tuning del matcher local (SP+LG, XFeat) con pérdida epipolar usando las poses de VO/GPS de
  FieldSAFE (P22); (b) usar MASt3R/RoMa como *teacher* offline sobre pares revisita (con poses GT para filtrar
  pseudo-labels) y destilar a un matcher liviano; (c) para SALAD, lo propuesto en #5 con positivos por GPS.
- **Evidencia**: P22 (SCENES), P4 (WildCross, +12-19 R@1 con fine-tuning), P13 (adaptadores).
- **Costo**: alto (1-2 semanas). **Riesgo**: sobreajuste a un solo campo/día → validar sólo en Rosario.
- **Medición**: igual que #3 y #5; siempre en la sesión de validación.

### R2-5 Profundidad densa para el estéreo de 5 cm (Rosario)
- **Idea**: precomputar offline mapas de profundidad con FoundationStereo/MonSter, o profundidad monocular métrica
  (Depth Anything V2 metric, UniDepthV2) re-escalada con los puntos estéreo cercanos; muestrear la profundidad en
  los keypoints del frame viejo y hacer PnP con más puntos confiables a 3-10 m.
- **Evidencia**: P25; cálculo de f·B (R2.1-2).
- **Implementación**: script Python que escribe `.npy` de profundidad por frame; en SLD, alternativa a
  `cv::sfm::triangulatePoints` que lee la profundidad (análogo a como se lee `global_desc`).
- **Costo**: medio. **Riesgo**: sesgos de escala de la profundidad aprendida en vegetación; tiempo (≥100 ms por
  frame, sólo offline). Evaluar primero contra el estéreo en FieldSAFE (baseline grande = referencia).
- **Medición**: e_vec/e_yaw en Rosario; % de revisitas que pasan geometría.

### R2-6 Explotar la estructura de surcos/carriles: chequeo de alias periódico y prior de yaw
- **Idea**: (a) estimar la dirección de las hileras (Rosario) o bordes de corte (FieldSAFE) por punto de fuga /
  Hough en la mitad inferior; el yaw relativo del PnP debe coincidir con la diferencia de direcciones de hilera
  (chequeo barato de consistencia). (b) **Test de ambigüedad periódica**: tras el PnP, re-evaluar el conteo de
  inliers para hipótesis desplazadas lateralmente ±1 espaciado de hilera (≈0.5 m en soja); si una hipótesis
  desplazada tiene un soporte parecido, el loop es ambiguo → rechazarlo o marcarlo con baja confianza. (c)
  "Código de barras" del surco: perfil de plantas/huecos a lo largo de la hilera como firma métrica (P27).
- **Evidencia**: aliasing a nivel de hilera (P8, P21), geometría de disposición de plantas (P27), análogo a un
  test de "segundo mejor" (ratio test) pero a nivel de pose.
- **Implementación**: (a)/(b) en `isGeometricallyConsistent_Exhaustive_Stereo` después del PnP (reproyectar los 3D
  con la pose desplazada y contar inliers con el matching ya hecho o con búsqueda guiada); parámetro
  `row_spacing`. (c) es investigación aparte.
- **Costo**: (a)+(b) 1-2 días. **Riesgo**: en FieldSAFE el "espaciado" es el ancho de corte (variable); en
  Rosario con plantas chicas (V3-V4) las hileras se ven bien, con V8 se cierran.
- **Medición**: precisión cuando se relaja la verificación (el objetivo es poder bajar `min_pnp_inliers` sin FP),
  e_vec lateral.

### R2-7 Brújula visual por la línea de horizonte/árboles
- **Idea**: extraer el perfil de horizonte (altura de la frontera cielo/tierra por columna, 1-D) y correlacionarlo
  circularmente/por desplazamiento entre query y candidato → yaw relativo casi puro (objetos lejanos). Usarlo como
  prior/chequeo del yaw del PnP y como inicialización de la rotación en R2-2(b).
- **Evidencia**: P29 (skyline como firma robusta), P19 (lejanos ⇒ rotación).
- **Implementación**: Python offline primero (segmentación por umbral + morfología; IR de Rosario tiene cielo
  oscuro, FieldSAFE cielo saturado); luego función C++ ligera. **Costo**: bajo-medio. **Riesgo**: horizontes sin
  estructura (campo abierto sin árboles en esa dirección), flares.
- **Medición**: error de yaw del perfil vs GT; e_yaw del loop con y sin prior.

### R2-8 Evaluación orientada a SLAM: ATE tras optimizar el grafo de poses
- **Idea**: armar un grafo VO + loops aceptados (con su pose relativa y covarianza, #7) y optimizar (GTSAM/g2o en
  Python, con y sin kernel robusto); reportar ATE/RPE vs GT antes y después. Es lo que al final importa: pocos
  loops bien distribuidos pueden valer más que muchos en un tramo (conecta con la métrica de cobertura).
- **Evidencia**: ROVER (P1) y Rosario v2 (P6) evalúan con ATE; GV-Bench (P16) con MR/AP.
- **Implementación**: `evaluation/` script nuevo; requiere exportar las poses SE(3) de la VO (mismo cambio que #1).
- **Costo**: bajo-medio (1 día). **Riesgo**: la VO pura puede tener deriva grande en giros → el ATE mide también la
  VO; comparar siempre contra "VO sin loops".
- **Medición**: ATE, RPE, y "ATE vs número de loops" para cada configuración.

### R2-9 Loops multi-sesión
- **Idea**: correr SLD con la base de datos persistida entre sesiones del mismo campo (FieldSAFE 12-07 y 12-37 el
  mismo día; en Rosario, las sesiones de un mismo día *si* comparten campo, a verificar con el GPS: el dataset cubre dos campos) para detectar revisitas entre sesiones.
- **Evidencia**: práctica de SLAM multi-mapa (ORB-SLAM3 Atlas); WildCross evalúa inter-secuencia (61.9 % R@1 vs
  63.2 % intra), o sea el caso inter-sesión no es mucho más difícil en el mismo día/semana.
- **Problema**: pocas revisitas en el mismo sentido por sesión → estadística pobre; en la práctica agrícola la
  revisita inter-sesión es el caso de uso principal.
- **Implementación**: `demoDetectorStereo.h`: aceptar varias listas de imágenes concatenadas con un id de sesión
  (o save/load de la base), `sld_eval.py`: GT en un marco común (GPS). **Costo**: bajo-medio. **Riesgo**: cambios
  de iluminación entre horas del día; en FieldSAFE el pasto ya cortado cambia de apariencia.
- **Medición**: TP/cobertura inter-sesión separadas de intra-sesión.

### R2-10 Registro BEV del suelo cercano (exploratoria)
- **Idea**: con el plano del suelo (estéreo), proyectar la región cercana a vista cenital métrica (IPM) y registrar
  query vs candidato en SE(2) por correlación de fase/Fourier-Mellin → (x, y, yaw) sin keypoints.
- **Evidencia**: Micro-GPS/HD Ground (P28, pero en superficies rígidas), IPM virtual ([arXiv 2303.05192](https://arxiv.org/abs/2303.05192)).
- **Riesgo**: alto: pasto no rígido, viento, el corte cambia la textura entre pasadas; soja crece. **Costo**: medio.
  Sólo si sobra tiempo; útil como verificación independiente de la PnP.

## R2.4 Ranking global actualizado (Ronda 1 + Ronda 2)

| # | Propuesta | Ataca | Costo | Riesgo |
|---|---|---|---|---|
| 1 | **#1** Loop tracking guiado por VO + consistencia métrica (PCM/ROVER) | consistencia temporal (71 % perdido), cobertura | medio | bajo-medio |
| 2 | **#2 + R2-8** GT por pose/co-visibilidad, MR/AP y ATE tras PGO | la medición misma | bajo | nulo |
| 3 | **R2-1** Máscara de ego-vehículo/sombra/flares | aliasing en recuperación (FieldSAFE) | muy bajo | bajo |
| 4 | **R2-2 + #4(b)** Filtro por incertidumbre de profundidad, E + escala estéreo, vertical conocida | pocos inliers, Rosario (fB=16 px·m) | bajo→medio | medio |
| 5 | **#3** Features aprendidas (SP/ALIKED/XFeat + LightGlue), probar offline primero | pérdidas en geometría | medio | medio |
| 6 | **#5 + R2-4** MegaLoc/BoQ zero-shot y luego fine-tuning autosupervisado | recuperación | bajo→alto | medio |
| 7 | **R2-9** Multi-sesión | pocas revisitas; caso de uso real | bajo-medio | bajo |
| 8 | **R2-6** Estructura de surcos: prior de yaw y test de alias periódico | FP al relajar geometría | medio | medio |
| 9 | **#4(a)** PnP contra mapa local multi-frame | pocos inliers | medio | medio |
| 10 | **#6** Filtro bayesiano/HMM con odometría (SMART, Xu 2021) | recuperación + consistencia | medio | medio |
| 11 | **R2-7** Brújula por horizonte | yaw | bajo-medio | medio |
| 12 | **R2-3** Tokens DINOv2 como matching grueso | aliasing local en verificación | medio | medio |
| 13 | **R2-5** Profundidad densa aprendida para Rosario | baseline 5 cm | medio | medio-alto |
| 14 | **R2-10** Registro BEV del suelo | verificación independiente | medio | alto |

(#7, score de confianza y covarianza, queda absorbido por #2/R2-8.)

Orden sugerido de ejecución: R2-1 y #2/R2-8 (baratos, cambian la línea de base), luego #1 (mayor ganancia
esperada), y R2-2(a) en paralelo; #3 y #5 como experimentos offline en Python antes de integrarlos en C++.

## R2.5 Ideas descartadas en esta ronda

- **Sol/sombras como prior absoluto de orientación**: entre dos pasadas separadas por minutos la dirección del
  sol casi no cambia, pero la sombra propia aparece en posiciones distintas según el rumbo y está en la zona que
  conviene enmascarar (R2-1); aporta menos que la IMU o el horizonte (R2-7).
- **Reloc3r/VGGT/MASt3R como verificador en línea**: regresan una pose aunque no haya solapamiento → sin un
  test de co-visibilidad no sirven para decidir un loop; quedan como oráculo/teacher (R2-4).
- **Cámara mirando al suelo**: fuera del alcance (los datasets no la tienen); R2-10 es el sustituto con la cámara
  frontal.
- **Filtro de partículas sobre la trayectoria completa (FastSLAM-like)**: más pesado que #6 y que PCM sobre loops
  verificados para el mismo beneficio en esta escala (unos km, una sola hipótesis de odometría razonable).

---

# Capítulo: interior del campo y codificación de hileras por huecos de cultivo

Estado: en progreso (2026-10-02). Idea del usuario: usar los huecos del cultivo (plantas muertas/faltantes) como
"código de barras" natural de cada hilera para cerrar loops en el interior del lote, donde SALAD+ALIKED+PnP falla
por aliasing periódico (13:39: cobertura interior 5 % vs 44 % en borde; loops falsos con ~0.1 m estimado vs 5-9 m
reales a lo largo y 2-4 m laterales).

## C.1 Literatura

**[C1] Robot Localization Based on Aerial Images for Precision Agriculture Tasks in Crop Fields** — N. Chebrolu,
P. Lottes, T. Läbe, C. Stachniss, ICRA 2019 ([pdf](https://www.ipb.uni-bonn.de/pdfs/chebrolu2019icra.pdf)). **La
evidencia más directa a favor de la idea.** Mapa aéreo de tallos de cultivo, malezas **y huecos**; el robot
(cámara inclinada, no nadir) detecta tallos con una FCN reentrenada, los proyecta al plano del suelo y **acumula
~15 m² de observaciones** (un frame da sólo ~30 detecciones, insuficiente contra el aliasing) antes de evaluarlas
en un filtro de partículas (5000 partículas, campo de verosimilitud por transformada de distancia, por tipo
semántico). Ablación (Tabla I): crops+malezas+**huecos** → error medio 4.3-5.1 cm; **sólo crops → 54.5 cm: el filtro
converge a la hilera equivocada (corrida dos hileras)**. Funciona a lo largo de varias semanas de crecimiento
(mapa con filtro de persistencia).
*Relevancia*: demuestra (i) que la posición de plantas sola no rompe el aliasing periódico y los huecos sí;
(ii) que hay que acumular varios metros; (iii) que la proyección al suelo desde cámara inclinada funciona.

**[C2] Robust Long-Term Registration of UAV Images of Crop Fields** — Chebrolu, Läbe, Stachniss, RA-L 2018 (ya
citado en P27): la *disposición geométrica* de las plantas es estable en el tiempo aunque la apariencia cambie.

**[C3] From Plants to Landmarks: Time-invariant Plant Localization that uses Deep Pose Regression** — F. Kraemer,
A. Schaefer, A. Eitel, J. Vertens, W. Burgard, IROS 2017 WS AGROB, [arXiv 1709.04751](https://arxiv.org/abs/1709.04751).
El punto de emergencia del tallo es invariante en el tiempo; FCN sobre RGB+NIR, precisión centimétrica en BoniRob.
*Relevancia*: alternativa más fina que la ocupación de vegetación (tallos en vez de manchas), útil entre sesiones.

**[C4] Registration of spatio-temporal point clouds of plants for phenotyping** — Chebrolu, Magistri, Läbe,
Stachniss, PLOS ONE 2021: correspondencias temporales entre plantas con un **HMM** sobre esqueletos.
*Relevancia*: el alineamiento de secuencias de plantas como inferencia HMM/DTW es estándar en fenotipado.

**[C5] End-to-End Deep Learning Models for Gap Identification in Maize Fields** — Waqar et al., CVPR 2024 WS
(Vision4Ag). Conteo de plantas + detección de huecos multitarea; multiespectral > RGB. **Stand counting/huecos
en soja**: flujos comerciales de "gap analysis" en V2-V3. *Relevancia*: los huecos son un objeto agronómico de
interés en sí (mapa de huecos = subproducto útil), y la detección es más fácil en estadios tempranos.

**[C6] Semantic Landmark / Semantic-Aware Particle Filter for vineyard localisation** — R. de Silva et al.,
[arXiv 2509.18342](https://arxiv.org/abs/2509.18342) (ICRA 2026 sub.) y [arXiv 2603.10847](https://arxiv.org/abs/2603.10847):
el aliasing es por hilera; landmarks semánticos + "muros" por hilera + prior GNSS; RTAB-Map visual falla.
*Relevancia*: confirma que la identidad de hilera es el problema central y que un filtro con observaciones
estructurales lo resuelve.

**[C7] Índices de vegetación**: ExG = 2g−r−b (Woebbecke et al., Trans. ASAE 1995) y ExG−ExR con umbral fijo en 0
(Meyer & Camargo Neto, Comput. Electron. Agric. 2008; calidad 0.88 vs 0.53 de ExG+Otsu). En NIR, el suelo arenoso
también satura (observado abajo), por lo que el IR solo no separa planta/suelo por intensidad.

**[C8] Matching de firmas 1D**: SeqSLAM/SMART (P30), alineamiento con DTW/HMM (C4), y localización por perfil
longitudinal de la ruta (perfil de rugosidad estimado con IMU, emparejado contra perfiles indexados; Sensors 2018,
PMC6210071). En patentes de guiado agrícola aparece "elegir la hilera del mapa que maximiza la correlación
cruzada" para georreferenciar la hilera detectada (familia US 11,277,956 / 11,789,459, "Vehicle controllers for
agricultural and industrial applications"). *Relevancia*: correlación cruzada normalizada sobre una firma
acumulada con odometría es la técnica base; DTW/HMM si la escala de la odometría deriva.

## C.2 Experimento de factibilidad (scripts en `evaluation/research/`)

**Montaje.** Sesión 2023-12-26-13-39-43, tramo interior (se descartan 15 m en cada cabecera). El GT se usa sólo
para elegir pasadas y para evaluar (y, en esta primera prueba, para proyectar).
- Geometría medida: cámara a **1.44 m** sobre el suelo (SGBM estéreo + orientación GT, desvío 1.8 cm entre 14
  frames; `cam_height.py`), inclinada ~17.5° hacia abajo; canopeo ~0.25 m. Hileras a **0.58-0.60 m**, con una
  orientación de 1.05° respecto de x del GT, igual en las 4 pasadas (estimada maximizando el contraste del perfil
  lateral del BEV).
- **Firma**: máscara de vegetación por frame → proyección al plano del canopeo bajo (h = 1.30 m) con la pose →
  acumulación en una grilla cenital de 2 cm en el **marco de las hileras** (u a lo largo, w lateral) → filas
  detectadas como picos del perfil lateral → **firma 1D por fila** = ocupación media en una banda de ±10 cm vs u
  (`bev.py`, `sig1d.py`).
- **Matching** (`barcode.py`): una ventana de L m de la pasada consulta con todas las filas observadas por ambas
  pasadas (|w − traza| ≤ 1.6 m) se compara contra el mapa de una pasada anterior. El score es la NCC media entre
  filas en función de (k = corrimiento en índice de fila, du = corrimiento a lo largo, ±10 m). Correcto = k=0 y
  |du| < 1 m.
- Dos máscaras: **ExG** en la cámara color (640×360, sincronizada por timestamp) e **IR** del pipeline
  (1280×720, intensidad suavizada < 240: en NIR el suelo cercano satura y las plantas no).

**Observaciones cualitativas.**
- En IR el suelo arenoso cercano satura (≈255) y las hojas también son brillantes. La textura **no** separa planta
  de suelo (el suelo tiene marcas de surco), pero un umbral de saturación sí sirve en la franja cercana (v ≥ 400 px,
  1.2-3.2 m adelante).
- Hay **huecos visibles** en casi todas las filas que no están pegadas a la cámara: 0.3-0.56 huecos ≥10 cm por
  metro y 8-16 % de la longitud en hueco. Las dos filas inmediatas al robot dan casi 0 (hojas en primer plano lo
  tapan todo) y las más lejanas se emborronan (`out/firmas_AC.png`).

**Resultado 1: pasadas cercanas (A+ vs C−, sentidos OPUESTOS, trazas a ~1 m).** Ventanas de 10 m, 7 posiciones:

| Firma | Correctas | score medio | 2º mejor (otra fila u otro du) | margen |
|---|---|---|---|---|
| Color ExG, continua | 7/7 | 0.67 | 0.38 | 0.28 |
| Color, **sólo huecos** (binaria, 20 % más bajo) | 7/7 | 0.47 | 0.27 | 0.20 |
| Color, sin huecos (recortada al percentil 20) | 7/7 | 0.61 | 0.35 | 0.26 |
| **IR**, continua | 7/7 | 0.59 | 0.22 | **0.37** |
| IR, sólo huecos | 7/7 | 0.45 | 0.21 | 0.25 |

Además, con ventana de 5 m: 9/9 correctas (margen 0.06-0.26); con 15 m: 4/4 (margen ≈0.27).
- **La firma identifica a la vez la fila (k) y la posición a lo largo (du)**, que es justo la ambigüedad
  periódica que hace fallar a SALAD+PnP.
- **Funciona en sentido opuesto**: el BEV en el marco de las hileras no depende del sentido de marcha. El
  pipeline actual no puede cerrar esos loops (la cámara frontal ve escenas distintas).
- **Los huecos solos alcanzan** (binaria, 7/7), pero la densidad continua da más margen: la "codificación" es
  el perfil de densidad de plantas, con los huecos como su parte más saliente.

**Sesgo sistemático.** du* = +0.36 m (color) / −0.34 m (IR), constante en todas las ventanas. Es un sesgo
dependiente del sentido de marcha: puede ser paralaje de proyectar hojas de altura variable sobre un plano, un
desfase temporal imagen-GT de ~0.2 s, o el brazo de palanca del GT. Se corrige calibrando h por dirección o
proyectando con profundidad estéreo. No afecta la identificación.

**Sensibilidad.** El umbral IR es sensible a la exposición: 225 y 240 andan; con 250, 4/4 ventanas siguen
siendo correctas pero el margen cae a ~0.07 (score 0.22 vs 0.16). En la integración hay que usar un umbral
relativo (percentil por frame).

**Límite observado (D− vs A+, trazas a 2.7 m).** 0/30 ventanas correctas y score ≈ 0.2-0.4 (nivel impostor):
casi no hay filas bien observadas por ambas pasadas. Con la franja cercana actual el alcance lateral útil es
≈ ±1.6 m alrededor de cada traza.
