# face-id

Iteraciones 1 y 2: comparación de imágenes e identificación de empleados registrados.

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

## Registro e identificación de empleados (Iteración 2)

Colocá una imagen autorizada por empleado en `data/employees/` con el nombre `ID__Nombre_Apellido.jpg` (también se admiten `.jpeg` y `.png`). Por ejemplo, `EMP001__Ana_Garcia.jpg` representa el ID `EMP001` y el nombre `Ana Garcia`. Las imágenes de este directorio están ignoradas por Git: no se publican con el código. Para probar con los presidentes, hay una foto registrada por persona (`AR001` Alberto Fernández, `AR002` Fernando de la Rúa y `AR003` Carlos Menem) y las segundas fotos están en `data/test-images/`, también ignorado. Ejecutá `.venv/bin/python -m pytest tests/test_employee_photos.py -v` para probar las tres identificaciones y un rostro no registrado (usa `demo-images/trump-1.jpeg`). Si faltan fotos locales, los casos correspondientes se omiten.

```python
import face_recognition

from facial_recognition.employee_registry import load_employees
from facial_recognition.engine import extract_faces, find_best_match

employees = load_employees("data/employees")  # una vez al iniciar; encodings en memoria
faces = extract_faces(face_recognition.load_image_file("data/test-images/alberto-2.jpeg"))
for face in faces:
    result = find_best_match(face.encoding, employees, threshold=0.6)
    print(result.employee_id, result.distance)
```

`employee_id` será `None` si ninguna distancia cumple el umbral; `distance` conserva la distancia al candidato más cercano. Sin candidatos, ambos valores son `None`. El archivo de cada empleado debe tener exactamente un rostro: las fotos ilegibles, sin rostro, con varios rostros o con un nombre inválido se omiten con una advertencia. Un ID repetido detiene la carga con un error; el directorio inexistente o sin empleados válidos también provoca un error controlado. La carga no se actualiza automáticamente: llamá de nuevo a `load_employees()` si cambian las imágenes.

Fuera de alcance: webcam, asistencia y persistencia CSV.
