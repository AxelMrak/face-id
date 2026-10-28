# Face ID

A local desktop demo for comparing faces, recognizing registered people through a webcam, and saving one attendance check-in per person per day. Images are processed on your computer; they are not sent to a cloud service. This is an academic demo, **not** a secure identity or liveness check.

## Install and open

Use Python 3.11 or newer. Run these commands from the project folder:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.gui
```

If `.venv` is already installed, just run the last command. Installing `face_recognition`/`dlib` may require a C++ compiler and CMake. A desktop environment is required for the GUI; the documented setup was tested on macOS, not on a clean installation of another OS. Allow camera access if your system asks.

## People included in the demo

`data/employees/` contains one registered reference photo for each of these people:

| ID | Registered person | Reference photo | Different photo for testing |
| --- | --- | --- | --- |
| AR001 | Alberto Fernández | `data/employees/AR001__Alberto_Fernandez.jpg` | `data/test-images/alberto-2.jpeg` |
| AR002 | Fernando de la Rúa | `data/employees/AR002__Fernando_De_La_Rua.jpeg` | `data/test-images/delarua-2.jpeg` |
| AR003 | Carlos Menem | `data/employees/AR003__Carlos_Menem.jpeg` | `data/test-images/menem-2.jpeg` |

These photos are included in the private repository for the demo. `demo-images/` also contains photos of Biden and Trump for comparisons; **they are not registered people** and should not create attendance. There is no pre-filled attendance history: `data/attendance.csv` is created after the first successful check-in and is ignored by Git.

To try another photo, search the web for a picture of **Carlos Menem**, **Alberto Fernández**, or **Fernando de la Rúa**, and save a clear image with only that person's face on your computer. Use a different photo from the registered reference. A random web image is **not guaranteed to match**: face angle, lighting, resolution, other faces, and the matching threshold all matter. Use images you are allowed to use; do not add downloaded or private photos to the repository without permission.

## Use the four GUI screens

1. **Asistencia (Attendance):** Keep the default camera `0`, matching threshold `0.6`, registered-photo folder `data/employees`, and CSV path `data/attendance.csv`. Click **Iniciar cámara**. Face the webcam with one registered person, or, for a demo, show it one of the included test photos (or a suitable web photo) on a phone or another screen. Keep one face clearly visible for **three consecutive processed frames**. A successful check-in shows **Ingreso registrado**; repeating it on the same day shows **Ya registrado hoy** and does not add another row. An unknown face, several faces, or a camera/CSV error does not register attendance. Click **Detener** to release the camera. Showing a photo to the camera may work because this demo has **no anti-spoofing**; never use it as proof that a person was physically present.
2. **Comparar fotos (Compare photos):** Choose two local images, each containing exactly one face, then click **Comparar fotos**. The screen displays both photos with their detected faces outlined, a match/no-match status, the face distance, and the threshold. For example, compare `data/employees/AR003__Carlos_Menem.jpeg` with `data/test-images/menem-2.jpeg` for a known person; compare `demo-images/trump-1.jpeg` with `demo-images/biden-1.jpeg` for different people. A match is a similarity result, not proof of identity.
3. **Registros (Records):** See the saved CSV as a table of employee ID, name, and check-in time with timezone. It reads the CSV path set on **Asistencia**. Click **Actualizar registros** after a new check-in. If there is no CSV yet, the list is empty. An invalid CSV displays an error rather than overwriting the file.
4. **Pruebas (Tests):** Click **Ejecutar pruebas** to run the automated test suite without a webcam. The output appears on this screen; you can cancel a run.

The screens are labeled in Spanish in the app. On a small window, scroll down to see the comparison result.

## Run from the command line (optional)

Use the same virtual environment and run commands from the project folder. To compare two photos:

```sh
.venv/bin/python -m scripts.compare_images --image-a demo-images/trump-1.jpeg --image-b demo-images/trump-2.jpeg --threshold 0.6
```

The command prints `MATCH` or `NO_MATCH` and opens image windows; press a key to close them. For webcam attendance instead of the GUI:

```sh
.venv/bin/python -m scripts.attendance --camera 0 --threshold 0.6
```

Press **Q** in the camera window to exit. Use `--employees path/to/photos` for another registered-photo folder or `--attendance-csv path/to/file.csv` to save elsewhere. Each registered photo must contain one face and use the filename format `ID__First_Last.jpg` (JPEG and PNG are also accepted). Do not commit private photos or attendance records.

Run all automated tests without a physical camera:

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

The tests cover face matching, the registered demo photos, three-frame confirmation, CSV persistence, and the GUI with a simulated camera. They do **not** replace a manual webcam check-in with a registered person; that end-to-end hardware check and installation on another computer have not yet been verified.
