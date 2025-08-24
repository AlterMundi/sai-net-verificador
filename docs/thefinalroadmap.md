Revisión del repositorio SmokeyNet y preparación del conjunto de datos FIgLib
1. Contexto y estructura del repositorio

El repositorio de GitHub denominado smokeynet‑engineering fue creado como parte de una iniciativa académica para analizar detección 
de humo de incendios forestales. El archivo README.md explica que el objetivo del proyecto es mejorar un modelo de detección de humo 
denominado SmokeyNet, desarrollado originalmente en la Supercomputadora de la Universidad de California en San Diego
raw.githubusercontent.com
. Este repositorio, sin embargo, sirve sobre todo como base de trabajo: contiene notebooks para procesar metadatos de cámaras, 
scripts auxiliares para preparar datos y un README con directrices de codificación y entorno virtual
raw.githubusercontent.com
. En la carpeta data hay subdirectorios raw y processed, pero no incluye el conjunto de imágenes; según el README, el conjunto de 
datos debe descargarse externamente
raw.githubusercontent.com
.

El código para el entrenamiento de SmokeyNet no está en este repositorio, sino en el repositorio GitLab 
pytorch‑lightning‑smoke‑detection (enlace indicado en el artículo original). Al clonar este segundo repositorio se obtienen 
implementaciones en PyTorch Lightning de los modelos de clasificación, detección y segmentación, incluidas las variantes de SmokeyNet 
y sus baselines.

2. Conjunto de datos FIgLib
2.1 Descripción general

El artículo FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real‑Time Wildland Fire Smoke Detection presenta el Fire Ignition 
Library (FIgLib) como un conjunto de datos público para detección de humo de incendios forestales
mdpi.com
. FIgLib contiene alrededor de 24 800 imágenes de alta resolución (1536×2048 px o 2048×3072 px) capturadas por 101 cámaras en 30 
estaciones de la red HPWREN en el sur de California. Las imágenes cubren 315 secuencias de incendios ocurridas entre junio de 2016 y 
julio de 2021
mdpi.com
. Cada secuencia proporciona unos 40 minutos antes y 40 minutos después del inicio del fuego, y cada imagen está etiquetada 
binariamente (humo/no‑humo). Para 144 incendios también se proporcionan cajas delimitadoras y máscaras de contorno del humo
mdpi.com
.

Los autores eliminaron incendios nocturnos, imágenes en blanco y negro y casos ambiguos; tras esta depuración, quedan 24 800 
imágenes, de las cuales 11 300 pertenecen a los incendios con anotaciones (conjunto de entrenamiento) y 9 800 al conjunto de 
evaluación
mdpi.com
. El artículo recomienda dividir los datos por incendios para evitar que imágenes del mismo fuego aparezcan en entrenamiento y prueba
mdpi.com
. Las listas exactas de incendios asignados a entrenamiento, validación y prueba están disponibles en un snippet de GitLab (según el 
artículo), pero la regla general es que los incendios con anotaciones se usan para entrenar y el resto para evaluar
mdpi.com
.

2.2 Preprocesamiento y generación de etiquetas por mosaicos (tiling)

Las imágenes completas contienen un amplio paisaje, por lo que detectar un penacho pequeño de humo resulta difícil con un etiquetado 
a nivel de imagen. Para proporcionar una señal de entrenamiento más localizada, las imágenes se redimensionan a 1392×1856 px, se 
recortan las 352 filas superiores (para eliminar cielo y nubes) y se recortan a un tamaño final de 1040×1856 px, lo que permite 
dividirlas en 45 mosaicos de 224×224 px superpuestos
mdpi.com
. A continuación se aplican augmentations aleatorias (reflejo horizontal, recorte vertical, variaciones de color/brillo/contraste y 
desenfoque) y se normalizan los canales RGB con media 0.5 y desviación estándar 0.5
mdpi.com
.

Para generar etiquetas a nivel de mosaico, se rellena la máscara de contorno (o la caja delimitadora si no hay contorno) y se cuenta 
el número de píxeles de humo en cada mosaico; un mosaico se etiqueta como positivo si contiene más de 250 píxeles de humo (0,5 % de 
sus píxeles)
mdpi.com
. Los mosaicos con menos humo se consideran negativos. Este etiquetado granular proporciona retroalimentación adicional durante el 
entrenamiento y permite al modelo localizar penachos pequeños
mdpi.com
.

2.3 Cómo descargar el conjunto de datos

El conjunto de datos FIgLib se distribuye desde los servidores HPWREN/UCSD. La página oficial ofrece un archivo comprimido (~4 GB) 
que contiene las imágenes y las anotaciones. Para descargarlo:

Acceder al sitio de descarga: diríjase a la página de la biblioteca Fire Ignition (hpwren.ucsd.edu/HPWREN-FIgLib), donde se ofrece la 
descarga del paquete completo. Si el navegador devuelve un error de certificado, añada la opción --no-check-certificate al usar 
herramientas como wget.

Descargar el archivo ZIP: haga clic en el enlace de descarga (o ejecute desde la terminal: wget --no-check-certificate 
<URL-del-zip>). El tamaño aproximado es 4,1 GB, por lo que la descarga puede tardar.

Descomprimir los archivos: tras descargar el ZIP, descomprímalo (unzip HPWREN-FIgLib.zip) en una carpeta de trabajo 
(~/datasets/FIgLib). La estructura típica es FIgLib/<fire_id>/<imagen.jpg> y un archivo JSON con etiquetas binaria y anotaciones de 
contorno por incendio.

Verificar integridad: compruebe que todas las carpetas de incendios están presentes (deberían ser 315) y que las imágenes se abren 
correctamente.

Si no es posible acceder al servidor HPWREN (por ejemplo, por restricciones geográficas), existen alternativas:

El repositorio de WIFIRE Commons replica el conjunto de datos FIgLib y permite la descarga previa autenticación.

Hugging Face hospeda un subconjunto de prueba (4880 imágenes) en el dataset leon-se/FIgLib‑Test. Puede usarse para validar procesos, 
aunque no contiene los datos de entrenamiento.

3. Preparación del conjunto de datos para SmokeyNet

Para entrenar un modelo con la misma arquitectura que SmokeyNet se deben seguir varios pasos de preprocesamiento:

Recopilación de imágenes y anotaciones: después de descomprimir FIgLib, identifique las imágenes de los 144 incendios con anotaciones 
y sus archivos de contorno/cajas. Estas conforman el conjunto de entrenamiento
mdpi.com
. Las imágenes de los 126 incendios restantes se destinarán a validación y prueba. Asegúrese de registrar la correspondencia entre 
cada imagen y su incendio para realizar divisiones consistentes.

Redimensionado y recorte: utilice una biblioteca como OpenCV o PIL para leer cada imagen, redimensionarla a 1392×1856 px y recortar 
las 352 filas superiores. Esto reduce el tamaño de la imagen y elimina la parte superior del cielo, lo que disminuye falsos positivos
mdpi.com
.

División en mosaicos: sobre el tamaño recortado (1040×1856 px) aplique un generador de ventanas deslizantes para crear 45 mosaicos de 
224×224 px con solapamiento de 20 px
mdpi.com
. Conserve para cada mosaico su posición en la imagen original para poder contar los píxeles de humo dentro.

Cálculo de etiquetas por mosaico: para cada mosaico, use la máscara de contorno o la caja delimitadora asociada al incendio para 
rellenar los píxeles de humo y cuente cuántos de ellos caen dentro del mosaico. Asigne etiqueta 1 si la cuenta supera 250 píxeles y 0 
en caso contrario
mdpi.com
. Si el incendio no tiene máscara de contorno para esa imagen, utilice la caja delimitadora como polígono y aplique el mismo criterio
mdpi.com
.

Almacenamiento de etiquetas: guarde las etiquetas binaria de la imagen (humo/no‑humo) y las etiquetas de los 45 mosaicos en un 
archivo (por ejemplo, JSON o CSV) que asocie el nombre del archivo de imagen con sus etiquetas. Estas etiquetas se usarán durante el 
entrenamiento.

Augmentación y normalización: en el proceso de carga de datos aplique aleatoriamente reflejo horizontal, recorte vertical, cambios de 
color/brillo/contraste y desenfoque para aumentar la variabilidad. Posteriormente normalice los valores de los canales RGB a media 
0.5 y desviación estándar 0.5
mdpi.com
. Esto es coherente con las transformaciones utilizadas en la implementación original.

División en conjuntos: cree divisiones de entrenamiento, validación y prueba a nivel de incendio. Según la guía del artículo, utilice 
todos los incendios anotados para entrenar (aprox. 11 300 imágenes) y divida los incendios restantes (9 800 imágenes) de forma 
equitativa entre validación y prueba
mdpi.com
. Al dividir, asegúrese de que las imágenes de un mismo incendio no se repartan entre distintos conjuntos
mdpi.com
.

4. Arquitectura y entrenamiento de SmokeyNet
4.1 Arquitectura

SmokeyNet es un modelo híbrido que combina redes convolucionales (CNN), memorias a largo plazo (LSTM) y un transformador de visión 
(ViT)
mdpi.com
. Su funcionamiento general es el siguiente:

Entrada: el modelo recibe dos fotogramas secuenciales de la misma escena, cada uno dividido en 45 mosaicos de 224×224 px
mdpi.com
. Usar dos fotogramas permite captar el movimiento del humo entre frames.

Extracción local con CNN: cada mosaico pasa por un CNN (backbone ResNet34 preentrenado en ImageNet) que genera representaciones de 
características para ambos fotogramas de forma independiente
mdpi.com
. Estos vectores resumen la apariencia de cada mosaico.

Modelado temporal con LSTM: para cada mosaico se toma la representación del frame actual y del frame anterior y se alimenta a un LSTM 
que combina información temporal, captando cómo evoluciona el humo entre los dos frames
mdpi.com
.

Modelado espacial con ViT: las salidas temporales de los 45 mosaicos se concatenan y se introducen en un Vision Transformer que 
incorpora atención entre mosaicos para comprender relaciones espaciales y producir una representación global de la imagen
mdpi.com
.

Cabezas de salida:

Tile heads: para cada mosaico, se aplican tres capas totalmente conectadas con activaciones ReLU y una capa sigmoidal que emite la 
probabilidad de humo en el mosaico
mdpi.com
. Se calcula una pérdida binaria entre esta predicción y la etiqueta de mosaico correspondiente, con ponderación de las clases para 
abordar el desequilibrio (peso de 40 para los mosaicos positivos)
mdpi.com
.

Image head: la representación CLS del ViT (que resume toda la imagen) pasa por tres capas fully‑connected (dimensiones 256‑64‑1) 
seguidas de una sigmoide para producir la probabilidad de humo en toda la imagen
mdpi.com
. La pérdida de imagen es una entropía cruzada binaria con un peso de 5 para los ejemplos positivos
mdpi.com
.

La combinación de pérdidas de mosaicos e imagen guía el entrenamiento para que el modelo aprenda tanto señales locales (mosaicos) 
como globales (imagen completa). La arquitectura modular permite sustituir la CNN o el LSTM por variantes más ligeras o eficientes
mdpi.com
.

4.2 Detalles de entrenamiento

El artículo detalla que los autores probaron distintos hiperparámetros y llegaron a la siguiente configuración óptima
mdpi.com
:

Optimizador: SGD con learning rate 0,001 y weight decay 0,001.

Batch size: se usaron lotes de 2 o 4 imágenes, acumulando gradientes para obtener un tamaño efectivo de 32 ejemplos
mdpi.com
. Esto se debe al alto consumo de memoria por el uso de dos frames y 45 mosaicos por imagen.

Épocas: 25 épocas, seleccionando el modelo con menor pérdida de validación.

Ponderaciones: para la pérdida a nivel de imagen se usó un peso 5 para las muestras positivas, y para la pérdida a nivel de mosaico 
se ponderó la clase positiva con peso 40
mdpi.com
.

Augmentaciones y normalización: ver sección 2.2.

A nivel práctico, el entrenamiento puede ejecutarse en PyTorch Lightning utilizando la implementación original. Se requerirá al menos 
una GPU de 11 GB (por ejemplo, NVIDIA 2080Ti) para sostener los dos frames y la arquitectura completa
mdpi.com
.

4.3 Pasos para entrenar un modelo con la misma arquitectura

Clonar el repositorio de código:

git clone https://gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection.git
cd pytorch-lightning-smoke-detection


Crear un entorno virtual e instalar dependencias:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Preparar el conjunto de datos:

Coloque las imágenes de FIgLib en un directorio (data/raw/images). Almacene también los archivos de máscara/caixas en 
data/raw/annotations.

Ejecute el script de preparación (en el repositorio hay notebooks como 1_process_camera_metadata.ipynb) o escriba su propio script 
para redimensionar, recortar, generar mosaicos y etiquetar cada mosaico siguiendo la metodología descrita en la sección 2.2. Guarde 
los resultados en data/processed/.

Crear listas de splits: Genere listas de archivos para cada conjunto (entrenamiento, validación y prueba) a nivel de incendio. Estas 
listas se guardan en data/splits/.

Entrenar SmokeyNet: Ejecute el script de entrenamiento en PyTorch Lightning, especificando la arquitectura SmokeyNet (ResNet34 + LSTM 
+ ViT) y las rutas a los datos procesados y a los splits. Ajuste el batch size según la capacidad de su GPU. Durante el 
entrenamiento, habilite la acumulación de gradientes para obtener el tamaño de batch efectivo de 32
mdpi.com
.

Evaluación: Después de entrenar, valide el modelo usando el conjunto de validación y finalmente evalúe en el conjunto de prueba. Los 
principales métricas son exactitud, precisión, recall, F1 y tiempo hasta detección. Utilice las mismas divisiones para poder comparar 
con los resultados del artículo (exactitud 83,49 %, F1 = 82,59 %, precisión 89,84 %, recall 76,45 %)
mdpi.com
.

5. Conclusiones

La arquitectura SmokeyNet combina el análisis espacial de una CNN, la agregación temporal de un LSTM y la atención global de un 
Vision Transformer, permitiendo detectar humo de incendios forestales con alta precisión y rapidez. El éxito del modelo depende de la 
disponibilidad del conjunto de datos FIgLib, donde cada imagen está cuidadosamente anotada y preprocesada. Para reproducir los 
resultados del artículo se deben descargar las imágenes desde el portal HPWREN, aplicar el preprocesamiento descrito (redimensionado, 
recorte y mosaicos), generar etiquetas a nivel de mosaico y entrenar la red con los hiperparámetros propuestos. Aunque el repositorio 
de GitHub smokeynet‑engineering ofrece material auxiliar, el entrenamiento completo se realiza con el código de PyTorch Lightning 
alojado en GitLab.

Referencias

Artículo FIgLib & SmokeyNet
mdpi.com
mdpi.com
mdpi.com
.

README del repositorio smokeynet‑engineering
raw.githubusercontent.com
.

Tabla de generación de etiquetas por mosaico y estrategias de preprocesamiento
mdpi.com
.
