# face-id

Iteración 1: detección y comparación local de dos imágenes. El motor (`src/facial_recognition/`) recibe imágenes RGB como arrays NumPy; no conoce archivos, cámara ni empleados. El script (`scripts/compare_images.py`) carga archivos y muestra las caras detectadas con OpenCV.

## Instalación

Se requiere Python 3.10+ y un entorno con ventanas gráficas para la visualización. `face_recognition` depende de `dlib`, que puede requerir compilador C++ y CMake para instalarse.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

En macOS, si la compilación de `dlib` falla porque el SDK activo es incompatible con el compilador, seleccioná un SDK funcional para la instalación (por ejemplo, `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk python -m pip install -r requirements.txt` si ese SDK existe).

## Comparación

```sh
python -m scripts.compare_images --image-a ruta/a.jpg --image-b ruta/b.jpg --threshold 0.6
```

Cada imagen debe contener **exactamente un rostro** para este script. El core admite varios rostros (`extract_faces` devuelve una lista), pero si falta un rostro, hay varios o no se puede generar un encoding válido, el script informa el problema y termina con código 2. Una distancia euclidiana menor o igual al umbral se muestra como `MATCH`; una mayor, como `NO_MATCH`. El umbral 0.6 es un valor inicial configurable, no una garantía de identidad. Para cerrar las ventanas, presioná una tecla.

## Pruebas

```sh
python -m pytest -q
```

Las pruebas automatizadas cubren el cálculo del umbral, el contrato de detección y los errores/resultados del CLI con encodings controlados. `tests/test_demo_images.py` también valida pares reales de la misma persona y de personas distintas cuando encuentra las fotos en `demo-images/`; omite esos casos si no están disponibles. Las fotografías de `demo-images/` no se incluyen en Git. Usá solo imágenes autorizadas y no publiques fotografías faciales sin autorización.

Fuera de alcance de esta iteración: empleados, persistencia CSV y webcam.
