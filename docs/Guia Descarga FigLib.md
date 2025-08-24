FIgLib: Guía de Descarga del Fire Ignition
Library 🔄
El Fire Ignition Library (FIgLib) es un conjunto de datos especializado en secuencias
temporales de imágenes de humo inicial de incendios, capturadas por cámaras fijas de la
red HPWREN (High Performance Wireless Research and Education Network) en el sur de
Californiawifire-data.sdsc.eduwifire-data.sdsc.edu. A diferencia de PyroSDIS y FASDD,
FIgLib está orientado a la detección de humo en tiempo real utilizando series de frames
(antes y después de la ignición), lo cual lo hace ideal para entrenar modelos con
componentes espaciotemporales (por ejemplo, arquitecturas CNN+LSTM como
SmokeyNet ). Consta de ~ 25,000 imágenes etiquetadas extraídas de cientos de secuencias
de incendios reales (aprox. 200 eventos), equilibrando datos positivos (humo visible) y
negativos (sin humo) para cada evento. Cada secuencia típicamente abarca alrededor de 40
minutos antes y 40 minutos después de una ignición confirmadawifire-data.sdsc.edu, con
imágenes capturadas a intervalos regulares (p. ej., 1 foto cada 30s o 1 min, dependiendo de
la cámara).

Etiquetado: Importante resaltar que FIgLib no utiliza bounding boxes ni segmentaciones.
Las imágenes están etiquetadas a nivel de imagen con una clase binaria: “ smoke ”
(presencia de humo de incendio inicial) o “ no-smoke ” (ausencia de humo). Todas las
imágenes posteriores al instante de aparición del humo en una secuencia se consideran
positivas, mientras que las previas (y algunas de días sin incidentes similares) son
negativas. No se distinguen niveles de densidad de humo ni hay anotaciones de falsos
positivos; es simplemente presencia/ausencia de humo. En los conjuntos liberados por los
autores, suele proporcionarse un archivo CSV con columnas como camera_id,
datetime, label (1 = smoke, 0 = no-smoke) para cada frame.

FIgLib fue introducido junto con el modelo SmokeyNet (Dewangan et al. 2022)mdpi.com.
Los autores proporcionaron herramientas para obtener el dataset desde la fuente HPWREN.
Detallaremos dos vías para adquirir FIgLib:

1. Descarga mediante el script oficial de SmokeyNet (recomendado)
El equipo de SmokeyNet desarrolló un método automatizado para compilar FIgLib desde el
archivo de imágenes de HPWREN. En el repositorio oficial de SmokeyNet (alojado en
GitLab de UC San Diego, con una posible réplica en GitHub), se incluye un script – por
ejemplo, referenciado como download_figlib.py – que permite descargar todas las
imágenes relevantes de manera programática. Este enfoque es ideal para reproducir
exactamente el mismo conjunto de datos que utilizaron en su investigación.

Pasos básicos:

Obtener el repositorio SmokeyNet: Clona o descarga el código fuente de
SmokeyNet. El código fue publicado en 2022 bajo PyTorch Lightning. Puedes
buscar "SmokeyNet SDSC UCSD" o el usuario khalooei/SmokeyNet en GitHub.
Alternativamente, la versión multimodal se encuentra en GitLab (mhnguyen/smoke-
detection-multimodal-smokeynet). Asegúrate de tener acceso al script de descarga
de FIgLib (generalmente en una carpeta de scripts/ o notebooks/ de ingeniería
de datos).
Instalar requerimientos: Revisa el README del repo para instalar dependencias.
Usualmente necesitarás Python 3 con bibliotecas como requests o urllib (para
descargar imágenes), posiblemente pandas (para manipular listas/csv) y otras
utilidades. Un entorno virtual es aconsejable.
Reunir metadata de eventos: El script de SmokeyNet probablemente necesita
como entrada una lista de cámaras y timestamps de ignición. En la publicación, los
autores mencionan intervalos de tiempo por cámara para ~101 cámaras en ~
sitios. Es posible que el repo incluya directamente un archivo con estas listas (ej.
cams.txt y timestamps.txt). De lo contrario, podrían estar en material
suplementario. Cada línea suele indicar una cámara HPWREN y la hora de inicio de
fuego para un evento.
Ejecutar el script de descarga: Invoca el script proporcionándole los parámetros
requeridos. Por ejemplo, según la documentación, podría usarse un comando
semejante a:
python download_figlib.py --camera_list cams.txt --timestamps
timestamps.txt --output ./FIgLib
donde cams.txt contiene los IDs de cámaras (ej.: mlo-n-mobo-c ) y
timestamps.txt las fechas/hora de ignición correspondidas. El script entonces se
conectará a la base de datos de imágenes de HPWREN (posiblemente vía la API de
AlertWildfire o por URL directas) para descargar frame por frame cada
secuencia deseada. Esto puede tardar, ya que son ~25k imágenes HD en total
(decenas de GB). No obstante, evita descargas manuales y te asegura la misma
recopilación filtrada que SmokeyNet.
Organización de datos: Al finalizar, deberías obtener una estructura organizada,
típicamente agrupada por cámara o por evento. Por ejemplo, es común estructurarlo
así:
FIgLib/
├── cam1/....jpg
├── cam1/....jpg
├── cam2/....jpg
├── cam2/....jpg
└── labels.csv
(donde cada carpeta de cámara contiene sus frames, y un labels.csv global indica
cuáles frames son humo/no-humo). En SmokeyNet, además, sincronizaron
temporalmente los frames de distintas cámaras que observaban el mismo incendio,
para eventualmente fusionar vistas; por eso hablan de calibraciones de tiempo entre
cámaras. El dataset en sí puede ser utilizado sin esta sincronización si vas a entrenar
solo con secuencias de una cámara a la vez (que es lo más común). SmokeyNet en
su arquitectura avanzó a usar múltiples cámaras en paralelo, pero eso es opcional.
Verificación: Una vez descargado, deberías tener ~ 25,000 archivos JPEG. Cada
secuencia de un incendio normalmente aporta decenas de imágenes antes/después.
Comprueba que el conteo se aproxime a lo esperado y que los ejemplos incluyen
tanto casos positivos como negativos equilibrados (SmokeyNet seleccionó
secuencias con el inicio del humo y frames previos sin humo, balanceando ambas
clases).
Usando esta vía, obtienes la versión original y exacta de FIgLib usada por SmokeyNet.
Esto garantiza compatibilidad con experimentos descritos en su paper. El script automatiza
la descarga desde HPWREN, evitando esfuerzos manuales. Ten en cuenta que no hay un
único archivo .zip público con FIgLib completo (debido al tamaño); por eso el enfoque
oficial es usar su herramienta para construirlo localmente.

2. Descarga directa desde el portal de HPWREN (alternativa manual)
Si prefieres o necesitas obtener FIgLib sin usar el código SmokeyNet, puedes hacerlo
directamente a través de los recursos públicos de HPWREN/UCSD. HPWREN mantiene
un archivo web de imágenes donde las secuencias de ignición están disponibles. La
WIFIRE Data Commons (UCSD/SDSC) ofrece una página para FIgLib con enlaces de
acceso directo:

Página principal: HPWREN Fire Ignition Image Library (FIgLib) en WIFIRE
Data Catalogwifire-data.sdsc.edu. Allí se describe el dataset y se proporciona un
enlace “Go to resource”. Al hacer clic, serás dirigido a un índice de archivos en el
servidor de HPWRENwifire-data.sdsc.edu.
Estructura de los datos: La biblioteca está organizada por evento de incendio.
Cada secuencia de imágenes corresponde a un incendio en una fecha/lugar
determinados. Al navegar el índice, verás probablemente carpetas o archivos
agrupados por nombre de incendio o por cámara+fecha. Por ejemplo, podrían estar
nombrados con la fecha del incidente o el apodo del fuego (p.ej. HolyFire_2018- 08 -
06 ). Dentro de cada carpeta de evento, encontrarás todos los frames JPEG de esa
secuencia, más un video MP4. En efecto, por cada evento los curadores incluyen
un video en MP4 que concatena las imágenes en forma de timelapse, útil para
visualizar rápidamente el desarrollo del humohpwren.ucsd.eduhpwren.ucsd.edu.
Los nombres de archivo siguen un patrón:
origin_timestamp__offset_(segundos)_from_visible_plume_appearance.j
pg
Por ejemplo: 2021 - 07 - 12T15- 30 - 00__offset_-
300_from_visible_plume_appearance.jpg podría ser una imagen 5 minutos
antes de que el humo fuera visible, mientras que ...offset_60...jpg sería 1
minuto después de aparición. Este esquema indica claramente cuáles frames son
antes (offset negativo) y después (offset positivo) del punto de inicio del
humowifire-data.sdsc.edu.
Descarga de secuencias: Desde el índice web, puedes descargar manualmente cada
secuencia. Para muchos eventos, HPWREN ha facilitado archivos comprimidos: es
común encontrar enlaces a archivos .tgz (tarball comprimido) que contienen todas
las imágenes de un evento, y el archivo .mp4 correspondientehpwren.ucsd.edu. Por
ejemplo, podrías ver SomeFire_event123.tgz junto a SomeFire_event123.mp4.
Si haces clic en .tgz, se iniciará la descarga del paquete de imágenes de ese
incendio (estos paquetes pueden pesar desde unos pocos MB hasta cientos de MB
dependiendo de la duración de la secuencia). Descarga todos los eventos que
necesites. Nota: También existe la posibilidad de un mega-archivo FIgLib.tgz que
compile todo (se reportó un ~100 GB tar file)hpwren.ucsd.edu, pero no es
aconsejable a menos que quieras absolutamente todo y tengas conexión muy
robusta.
Reconstrucción manual del dataset: Tras bajar varias secuencias, organízalas en
carpetas. Puedes replicar la estructura de SmokeyNet: crear un directorio por
cámara o por evento. Asegúrate de distinguir las clases: dado el esquema de
nombres, podrías programar una simple rutina que etiquete como no-smoke los
frames con offset negativo y como smoke aquellos con offset ≥ 0 (incluso offset = 0,
momento de aparición, se considera ya positivo)hpwren.ucsd.edu. Alternativamente,
en la página de WIFIRE mencionan que el momento “origin” se eligió
precisamente como la aparición visible del humowifire-data.sdsc.edu, así que
cualquier imagen con “offset_0” o positivo tiene humo. Puedes generar un CSV
consolidado con este criterio.
Crédito y licencia: Al usar los datos de HPWREN, debes atribuir la fuente.
HPWREN solicita dar crédito a hpwren.ucsd.edu en trabajos derivadoswifire-
data.sdsc.edu. Aunque la página indica "License Not Specified", se trata de datos
abiertos para investigación con la condición de mención. Evita usos comerciales sin
permiso expreso. Siempre es bueno contactar a HPWREN (vía su formulario de
feedback) para informarles de tu uso, como ellos animan, fortaleciendo
colaboracioneswifire-data.sdsc.edu.
Esta vía manual te permite descargar solo eventos específicos que te interesen o
inspeccionar el contenido antes de bajarlo todo. Podría ser útil si, por ejemplo, sabes que
quieres entrenar con ciertos incendios emblemáticos o probar el modelo en eventos
particulares. También es una forma de obtener los videos ilustrativos.

Consideraciones Finales (FIgLib)
Formato de las etiquetas: Como mencionado, la etiqueta es binaria por imagen.
No existen bounding boxes en FIgLib. Por tanto, no es adecuado entrenar YOLO
directamente con FIgLib a menos que decidas anotarle cajas manualmente (lo cual
no es el objetivo, ya que se asume que el modelo aprenda dónde mirar mediante su
arquitectura de división de imagen en celdas, como hizo SmokeyNet). FIgLib está
pensado para clasificación (detección de humo vs no-humo en la imagen completa)
y para aprovechar la dimensión temporal (secuencia de frames).
Uso con YOLOv8: Si insistes en usar YOLOv8, podrías tratarlo como un caso
especial de una sola clase (“smoke”) y entrenarlo en modo classification en vez de
detection (YOLOv8 tiene un modo de clasificación de imágenes). Sin embargo, es
más provechoso usar arquitecturas CNN+LSTM o 3D CNN que exploten la
secuencia temporal, ya que el valor de FIgLib está en los cambios sutiles en los
primeros minutos de un fuego.
Uso con CNN+LSTM: Este es el escenario natural. Por ejemplo, SmokeyNet toma
5 frames consecutivos de una misma cámara (que abarcan ~5 minutos) como
entrada a un modelo que combina extracción espacial (CNN en cada frame) y fusión
temporal (LSTM). Para replicar esto, puedes dividir cada secuencia en segmentos
de 5 minutos (o X frames) y alimentar esos stacks al modelo. Asegúrate de
mantener el orden cronológico y de sincronizar la etiqueta (basta con que uno de los
frames tenga humo para que el segmento se considere positivo en la tarea de
detección temprana ). SmokeyNet incluso incorporó múltiples cámaras, pero
empezar con una sola cámara a la vez es más sencillo.
Tamaño y almacenamiento: FIgLib completo ocupa alrededor de 100+ GB en
imágenes (muchas cámaras son de 1080p o más). Ten esto en cuenta al descargar.
Puedes optar por reducir resolución de las imágenes si tu modelo no requiere el full
HD, para acelerar entrenamiento.
Actualizaciones: La versión original abarcó incendios hasta ~2020. Es posible que
HPWREN continúe añadiendo eventos (la página indica última actualización en
agosto 2024)wifire-data.sdsc.edu. Puedes incorporar nuevos eventos si lo deseas,
siguiendo el mismo procedimiento de descarga (por ejemplo, si quieres ampliar el
dataset con incendios de 2021-2023, que HPWREN haya archivado). Solo mantén
la consistencia de etiquetas y divisiones.
En conclusión, FIgLib se obtiene mejor usando el script de SmokeyNet para garantizar un
conjunto de datos listo y equivalente al de la literatura. Como alternativa, el portal de
HPWREN permite explorar y descargar las secuencias manualmente, con archivos .tgz por
evento y videos MP4 ilustrativoswifire-data.sdsc.eduwifire-data.sdsc.edu. Recuerda que es
un dataset de clasificación binaria secuencial , orientado a detectar humo de ignición en
sus primeros instantes; no contiene anotaciones de localización precisas, pero su riqueza
temporal y necesidad de detección sutil lo hacen un recurso único para investigación en
early fire detection. Siempre da crédito a HPWREN/UCSD al usar FIgLib, y respeta el
espíritu de uso no comercial a menos que obtengas permiso. ¡Con estos datos en mano,
estarás listo para entrenar modelos avanzados que combinen visión e inteligencia temporal
para detectar incendios de forma temprana y automática! 🚀

Referencias: Para más detalles, ver Dewangan et al. , Remote Sensing 14(4):1007, 2022
(FIgLib & SmokeyNet)mdpi.com y Wang et al. , 2022 (FASDD,
DOI:10.57760/sciencedb.j00104.00103)essd.copernicus.org, así como la documentación
oficial en Hugging Face (PyroSDIS)huggingface.co y Kaggle (FASDD). Cada fuente de
datos aquí descrita incluye instrucciones y licencias que se han resumido en esta guía.
¡Buena descarga y mejor entrenamiento! 🔥
