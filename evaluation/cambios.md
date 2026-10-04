4 de Octubre de 2026:
- El PnP de la verificación geométrica ahora usa los intrínsecos rectificados (las 3 primeras columnas de la matriz de proyección P izquierda) y sin distorsión, en vez de la K cruda + coeficientes de distorsión. Las imágenes y la triangulación están rectificadas, así que la K cruda introducía un sesgo en la traslación: con el robot detenido en FieldSAFE daba ‖t‖ ≈ 0.16 m cuando la verdad es ≈ 0 (fx 556 vs 581, 4 %). Con la K rectificada baja a ≈ 1 mm (arreglo traído de la branch claude).

28 de Septiembre de 2026:
- Se agregó el parámetro de frecuencia de imágenes a demoStereo (antes estaba hardcodeado en 10)
- Ahora se guardan los resultados de las corridas en un archivo de texto
- 