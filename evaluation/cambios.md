28 de Septiembre de 2026:
- Se agregó el parámetro de frecuencia de imágenes a demoStereo (antes estaba hardcodeado en 10)
- Ahora se guardan los resultados de las corridas en un archivo de texto
- 

Branch distanciaRecorrida:
- Se reemplazó el dislocal (no matchear contra las imágenes de los últimos 20 s) por una distancia mínima recorrida: solo se matchea contra imágenes desde las que el robot recorrió al menos `--min-distance` metros (parámetro obligatorio de demoStereo). La distancia se calcula sobre el plano xy con el archivo de poses, que hace de odometría de las ruedas.
- En el archivo de resultados se guardan también el vocabulario usado (`vocabulary`) y la distancia mínima (`min_travel_distance`).
