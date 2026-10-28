# face-id

Iteraciones 1–4: comparación de imágenes, identificación por webcam y registro diario de asistencias en CSV.

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

Las pruebas automatizadas cubren el cálculo del umbral, el contrato de detección y los errores/resultados del CLI con encodings controlados. `tests/test_demo_images.py` valida pares reales de la misma persona y de personas distintas con las fotos de ejemplo incluidas en `demo-images/`. Usá solo imágenes que tengas permiso para redistribuir.

## Registro e identificación de empleados (Iteración 2)

Colocá una imagen autorizada por empleado en `data/employees/` con el nombre `ID__Nombre_Apellido.jpg` (también se admiten `.jpeg` y `.png`). Por ejemplo, `EMP001__Ana_Garcia.jpg` representa el ID `EMP001` y el nombre `Ana Garcia`. Las fotos de ejemplo autorizadas de este directorio se incluyen en el repositorio para poder probarlo en otras computadoras. No agregues retratos privados sin consentimiento, incluso si el repositorio es privado. Para probar con los presidentes, hay una foto registrada por persona (`AR001` Alberto Fernández, `AR002` Fernando de la Rúa y `AR003` Carlos Menem) y las segundas fotos están en `data/test-images/`, también incluidas en el repositorio. Ejecutá `.venv/bin/python -m pytest tests/test_employee_photos.py -v` para probar las tres identificaciones y un rostro no registrado (usa `demo-images/trump-1.jpeg`). Si faltan fotos locales, los casos correspondientes se omiten.

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

## Reconocimiento en vivo (Iteración 3)

Con las fotos de referencia cargadas en `data/employees/`, conectá una webcam y ejecutá desde la raíz del proyecto:

```sh
python -m scripts.attendance --camera 0 --threshold 0.6
```

Si usás el entorno virtual sin activarlo, reemplazá `python` por `.venv/bin/python`. `--camera` selecciona el dispositivo (0 por defecto), `--threshold` cambia el umbral (0.6 por defecto) y `--employees` permite usar otro directorio de retratos. El sistema solicita 640 × 480, procesa frames localmente con el detector HOG y reutiliza los encodings del registro. En la ventana activa, presioná **Q** para salir; también se liberan la cámara y las ventanas ante fallos de captura o interrupción. Si el sistema pide permiso para usar la cámara, concedelo al proceso desde el que ejecutás el comando.

Un único rostro conocido muestra un rectángulo verde y el nombre; uno no registrado, rojo y `UNKNOWN`. Sin rostros se muestra el video sin marcas. Con varios rostros se dibujan rectángulos amarillos y `MULTIPLE_FACES`, pero no se identifica a nadie en ese frame. Para probar apertura, visualización y cierre hace falta una webcam real: las pruebas automáticas (`python -m pytest tests/test_camera.py tests/test_attendance.py -q`) usan frames y cámaras simuladas.

## Asistencia diaria (Iteración 4)

El mismo comando registra automáticamente un ingreso al reconocer **tres frames consecutivos** con el mismo empleado. Un rostro desconocido, varios rostros o un empleado distinto reinician la confirmación. Tras guardar el registro aparece `Attendance registered` con la hora; si ya existe para ese ID y fecha local, aparece `Already registered today` y no se agrega otra fila. La asistencia se solicita en el tercer frame confirmado, no al iniciar el programa.

El archivo predeterminado es `data/attendance.csv`; se crea con encabezados `employee_id,employee_name,check_in_at` y timestamps ISO 8601 con desplazamiento horario. Los registros anteriores se cargan al iniciar y se conservan; un empleado puede volver a registrar al día siguiente. Podés cambiar la ruta con `--attendance-csv ruta/archivo.csv`. El sistema asume una sola instancia escribiendo el archivo: si la escritura falla o el CSV existente es inválido, informa el error y cierra la cámara sin confirmar una asistencia falsa. No hay turnos, cálculo de horas ni control antispoofing: tres frames son solo una comprobación de estabilidad. `data/attendance.csv` se excluye de Git para evitar publicar identidades y horarios de ingreso inadvertidamente.
