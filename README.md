# face-id

MVP académico: comparación de imágenes, identificación local por webcam y registro diario de asistencias en CSV. El motor (`src/facial_recognition/`) recibe imágenes RGB como arrays NumPy; no conoce archivos ni cámara. El procesamiento permanece en el equipo, sin envío de imágenes a servicios externos.

## Instalación

Se requiere Python 3.11+ y un entorno con ventanas gráficas para la visualización. `requirements.txt` fija las versiones comprobadas en macOS con Python 3.11; no se verificó una instalación limpia en otros sistemas. `face_recognition` depende de `dlib`, que puede requerir compilador C++ y CMake para instalarse.

```sh
python3.11 -m venv .venv
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

Las pruebas automatizadas cubren el cálculo del umbral, el contrato de detección y los errores/resultados del CLI con encodings controlados. `tests/test_demo_images.py` valida pares reales de la misma persona y de personas distintas con las fotos de ejemplo incluidas en `demo-images/`. `tests/test_mvp_integration.py` prueba el recorrido foto real → reconocimiento → CSV → reinicio, sin webcam. Usá solo imágenes que tengas permiso para redistribuir.

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

Si usás el entorno virtual sin activarlo, reemplazá `python` por `.venv/bin/python`. `--camera` selecciona el dispositivo (0 por defecto), `--threshold` cambia el umbral (0.6 por defecto) y `--employees` permite usar otro directorio de retratos. El sistema solicita 640 × 480, procesa frames localmente con el detector HOG y reutiliza los encodings del registro. En la ventana activa, presioná **Q** para salir. Un fallo de captura reinicia la confirmación de identidad; se permiten dos reintentos seguidos y al tercero se informa el error y se liberan la cámara y las ventanas. Un frame con encoding inválido se omite, se informa por stderr y reinicia la confirmación. Si el sistema pide permiso para usar la cámara, concedelo al proceso desde el que ejecutás el comando.

Un único rostro conocido muestra un rectángulo verde y el nombre; uno no registrado, rojo y `UNKNOWN`. Sin rostros se muestra el video sin marcas. Con varios rostros se dibujan rectángulos amarillos y `MULTIPLE_FACES`, pero no se identifica a nadie en ese frame. Para probar apertura, visualización y cierre hace falta una webcam real: las pruebas automáticas (`python -m pytest tests/test_camera.py tests/test_attendance.py -q`) usan frames y cámaras simuladas.

## Asistencia diaria (Iteración 4)

El mismo comando registra automáticamente un ingreso al reconocer **tres frames consecutivos** con el mismo empleado. Un rostro desconocido, varios rostros o un empleado distinto reinician la confirmación. Tras guardar el registro aparece `Attendance registered` con la hora; si ya existe para ese ID y fecha local, aparece `Already registered today` y no se agrega otra fila. La asistencia se solicita en el tercer frame confirmado, no al iniciar el programa.

El archivo predeterminado es `data/attendance.csv`; se crea con encabezados `employee_id,employee_name,check_in_at` y timestamps ISO 8601 con desplazamiento horario. Los registros anteriores se cargan al iniciar y se conservan; un empleado puede volver a registrar al día siguiente. Podés cambiar la ruta con `--attendance-csv ruta/archivo.csv`. El sistema asume una sola instancia escribiendo el archivo: si la escritura falla o el CSV existente es inválido, informa el error y cierra la cámara sin confirmar una asistencia falsa. No hay turnos, cálculo de horas ni control antispoofing: tres frames son solo una comprobación de estabilidad. `data/attendance.csv` se excluye de Git para evitar publicar identidades y horarios de ingreso inadvertidamente.

## Componentes y algoritmo

| Componente | Responsabilidad |
| --- | --- |
| `scripts/compare_images.py` | Carga dos imágenes y usa el motor compartido; exige exactamente un rostro por imagen. |
| `scripts/attendance.py` | Integra registro, motor, reglas, CSV y webcam; convierte frames BGR de OpenCV a RGB. |
| `src/facial_recognition/engine.py` | Detecta con HOG en CPU, extrae encodings y compara por distancia euclidiana. |
| `src/facial_recognition/employee_registry.py` y `models.py` | Cargan retratos `ID__Nombre.jpg` y definen los datos compartidos. |
| `src/facial_recognition/attendance_service.py` y `attendance_csv.py` | Confirman tres frames, deduplican por ID y fecha local, y persisten registros. |
| `src/facial_recognition/camera.py` | Abre, lee y libera cámara y ventanas. |

El motor compara encodings de 128 valores y elige el empleado de **menor distancia** si cumple el umbral inclusivo (0.6 por defecto). La comparación 1:1 exige un rostro en cada foto; el motor puede detectar varios, pero la asistencia solo identifica cuando aparece uno. La coincidencia no demuestra identidad ni presencia real: no hay detección de liveness.

El CSV usa UTF-8 y las columnas `employee_id,employee_name,check_in_at`. Ejemplo de fila ficticia: `EMP001,Ana Garcia,2026-10-28T08:05:00-03:00`. La fecha para evitar duplicados se obtiene del timestamp local con su zona horaria. Si el CSV es inválido, el script informa el error **sin sobrescribirlo**; si no existe, se crea al primer ingreso confirmado.

## Demostración reproducible

1. Instalá las dependencias como se indica arriba. Las fotos autorizadas de `demo-images/`, `data/employees/` y `data/test-images/` ya están incluidas en el repositorio privado.
2. Ejecutá `python -m scripts.compare_images --image-a demo-images/trump-1.jpeg --image-b demo-images/trump-2.jpeg`: debe indicar `MATCH` y mostrar dos rectángulos; cerrá las ventanas con una tecla. Repetí con `--image-b demo-images/biden-1.jpeg` para `NO_MATCH`.
3. Ejecutá `python -m pytest -q` para verificar los casos automáticos, incluido el recorrido completo con la foto `data/test-images/alberto-2.jpeg` y un CSV temporal.
4. Para la prueba física, prepará un retrato autorizado `EMP004__Nombre_Apellido.jpg` en un directorio **fuera del repositorio**, por ejemplo `~/face-id-demo-employees/`; usá ese directorio con `--employees`. Ejecutá `python -m scripts.attendance --camera 0 --employees ~/face-id-demo-employees --attendance-csv data/attendance.csv`. Mirá a la cámara al menos tres frames procesados: comprobá nombre y `Attendance registered`, una sola fila en el CSV con zona horaria y el mensaje `Already registered today` tras reiniciar. Verificá que `UNKNOWN` y varios rostros no agreguen filas; salí con **Q**. Si no querés registrar una asistencia real, usá un CSV temporal fuera del repositorio y borralo al terminar.

La prueba automática no necesita cámara. En la comprobación física realizada durante la Iteración 5, la webcam abrió, capturó un frame de 640 × 480 y el script abrió y cerró correctamente; no apareció un rostro conocido, así que **el ingreso real desde webcam y la revisión visual de las ventanas siguen pendientes de confirmación manual**.

## Matriz de verificación del MVP

Los **11 requisitos funcionales planificados** tienen implementación: cargar/procesar imágenes y comparar rostros (`compare_images.py`, `engine.py`), convertir BGR a RGB (`attendance.py`), detectar/codificar/calcular distancias (`engine.py`), identificar empleados (`employee_registry.py`, `engine.py`), capturar webcam (`camera.py`), dibujar nombres y rectángulos (`attendance.py`) y registrar nombre/hora en CSV (`attendance_service.py`, `attendance_csv.py`). Cobertura del diseño **11/11**; no equivale a una prueba manual completa.

La lista de **19 escenarios** queda distribuida así (los tests reales están en `tests/`; la prueba física indicada arriba aún debe confirmarse con un rostro registrado):

| Área | Escenarios cubiertos automáticamente |
| --- | --- |
| Reconocimiento (5) | Misma persona, personas distintas, sin rostro, múltiples rostros, umbral configurable. |
| Registro (4) | Carga de retratos, archivo inválido, ID duplicado, directorio vacío o ausente. |
| Webcam (5) | Apertura, lectura y reintentos limitados, conversión RGB y anotación, desconocidos o múltiples rostros, liberación al salir o fallar. |
| Asistencia (5) | Tres frames y reinicios, desconocidos y múltiples, duplicado diario y cambio de día, CSV persistente con zona horaria, fallo de escritura sin confirmación. |

Se prueban la carga de fotos reales y el recorrido reconocimiento → CSV sin hardware. Una instalación limpia fuera de este macOS y el flujo visual con un empleado registrado frente a la webcam no están verificados.
