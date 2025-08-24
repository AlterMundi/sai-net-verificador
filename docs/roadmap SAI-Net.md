SAI — Informe final de arquitectura
(A100×2)
Detector (cajas): YOLOv8 entrenado con PyroSDIS + FASDD · Verificador
(secuencial): SmokeyNet-like (CNN + LSTM + ViT) entrenado con FIgLib
Readme técnico para implementar el entrenamiento en Python

1) Resumen ejecu,vo
Decisiones clave
o Detector : Ultralytics YOLOv8-s/m (anchor-free, C2f) para localizar humo
con bounding boxes. Entrenar solo con PyroSDIS + FASDD (mezcla
estratificada). Esta cabeza es rápida y estable en GPU y está probada en
fuego/humo.
o Verificador : arquitectura SmokeyNet-like : CNN (ResNet-34) para tiles +
LSTM para historia corta + ViT para razonamiento global sobre
embeddings de tiles. Entrenar solo con FIgLib (secuencias). SmokeyNet
demostró F1≈82.6 % y detección ≈3.1 min post-ignición.
o Orquestación : el detector propone candidatos; el verificador confirma
sobre una ventana temporal corta y emite alarma si la evidencia persiste
(≥ k de N frames). Esta separación reduce falsos positivos y mantiene
sensibilidad temprana.
Cómputo : entrenamiento distribuido DDP en 2× A100 (BF16/FP16), gradient
accumulation y mixed precision.
Descarga de datos : usar las guías internas para FIgLib y PyroSDIS/FASDD
(scripts de exportación a formato YOLO y reconstrucción de secuencias).
Notas operativas (Drive) : seguir el Index.md de “Database” para localizar
materiales y no usar el directorio “Falopa”.
Rationale : YOLOv8 ofrece localización robusta y eficiente en humo/fuego;
SmokeyNet-like aporta temporalidad + razonamiento global para validar casos sutiles
(humo incipiente) con evidencia empírica en FIgLib.

2) Estructura de proyecto (sugerida)
sai/
├─ configs/
│ ├─ yolo/pyro_fasdd.yaml # data.yaml combinado (YOLO)
│ ├─ yolo/hyp.sai.yaml # hiperparámetros YOLO
│ └─ smokeynet/*.yaml # config del verificador (tiles, L,
ViT, etc.)
├─ data/
│ ├─ raw/
│ │ ├─ pyro-sdis/ # export HF → YOLO
│ │ ├─ fasdd/ # convertido a YOLO

│ │ └─ figlib/ # secuencias (+ csv de labels)
│ ├─ yolo/ # images/{train,val},
labels/{train,val}
│ └─ figlib_seq/ # split por evento (train/val/test)
├─ src/
│ ├─ detector/ # wrappers YOLO (train/eval/export)
│ ├─ verifier/ # cnn_lstm_vit/ (LightningModule)
│ └─ dataio/ # datamodules (YOLO export, FIgLib
sequences)
├─ scripts/
│ ├─ download_pyrosdis_fasdd.sh
│ ├─ export_pyrosdis_to_yolo.py
│ ├─ convert_fasdd_to_yolo.py
│ ├─ download_figlib.py # script repo SmokeyNet adaptado
│ └─ build_figlib_sequences.py
└─ outputs/
├─ yolo/ckpts/
└─ smokeynet/ckpts/

3) Datasets y preparación (solo los requeridos)
3.1 PyroSDIS (detector)
Fuente : Hugging Face pyronear/pyro-sdis (imágenes con cajas YOLO ;
monoclase “smoke”).
Exportación : cargar con datasets, volcar imágenes y .txt YOLO a data/yolo/.
La guía interna detalla la exportación y un data.yaml listo.
3.2 FASDD (detector)
Contenido : ≈100 k imágenes de cámaras y drones con humo/fuego ; es adecuado
para detectores y aporta objetos pequeños. Conviene convertir a YOLO (si no
llega en ese formato).
Conversión : usar el script sugerido en la guía para normalizar anotaciones y
generar labels/ YOLO.
3.3 FIgLib (verificador)
Contenido : ~25 k imágenes en secuencias (antes/después de ignición) de cámaras
fijas HPWREN; etiquetas a nivel imagen (smoke/no-smoke). Es el dataset
utilizado por SmokeyNet.
Descarga : preferir el script oficial del repo SmokeyNet (download_figlib.py) y
luego construir secuencias con un csv por evento. La guía resume pasos.
Split recomendado : por evento (no mezclar frames de un mismo incendio entre
train/val/test). En figlib_seq/: train/, val/, test/ con metadatos (cámara,
timestamp, label).

4) Detector (A100×2) — YOLOv8 sobre PyroSDIS + FASDD
4.1 Configuración (Ultralytics)
Modelo : yolov8s.pt (inicial) → evalúa también yolov8m.pt si hay margen.
Data : configs/yolo/pyro_fasdd.yaml con rutas a images/ y labels/
combinadas; monoclase humo (usar single_cls=True si simplifican clases).
Augmentations : Mosaic, MixUp, CutMix + jitter de color/brillo (prácticas
sugeridas para humo/fuego).
Entrenamiento : DDP en 2×A100 , BF16/FP16, cosine LR , warmup.
Métricas : mAP@0.5 y Recall para objetos pequeños (threshold bajo en conf para
priorizar sensibilidad).
4.2 Receta (CLI)
Entrenar detector (DDP, 2xA100)
yolo detect train
data=configs/yolo/pyro_fasdd.yaml
model=yolov8s.pt imgsz=960 epochs=150 batch=64
device=0,1 workers=16 amp=bf16 cos_lr=True
hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=5 translate=0.1 scale=0.
shear=2.0
mosaic=1.0 mixup=0.1 copy_paste=0.0
name=sai_yolov8s_pyrofasdd

Justificación : YOLOv8 (anchor-free + C2f) ofrece mejor equilibrio precisión-latencia y ha
mostrado mAP altos en humo/fuego.

5) Verificador (A100×2) — SmokeyNet-like (CNN + LSTM + ViT) sobre FIgLib
5.1 Diseño del modelo
Entrada : imagen completa → rejilla de tiles 224×224 (p.ej., 45 tiles por frame,
como en SmokeyNet). Para cada frame de la ventana temporal L (2–3).
Backbone CNN por tile : ResNet- 34 pretrained (ImageNet).
Agregado temporal : LSTM bi-direccional (2 capas, hidden=512) aplicado a la
secuencia de embeddings por tile (concat de frames).
Agregado espacial/global : ViT-S (encoder 6–8 bloques, dim=768, heads=12) que
toma como tokens los embeddings finales por tile (más token [CLS]).
Cabezales :
o Head global (binaria smoke/no-smoke; BCE con label smoothing ).
o Heads auxiliares por tile (opcionales) para regularizar (como en
SmokeyNet).
Regularización : dropout 0.2–0.3 en MLP, LayerNorm en ViT.
Pérdida total : L = λ_global * BCE + λ_tiles * BCE_tiles (p.ej., 1.0 y 0.3).
Por qué así : SmokeyNet combinó textura local (CNN) + memoria corta (LSTM) +
razonamiento global (ViT) y reportó F1≈82.6 con tiempo-a-detección ≈3.1 min en
FIgLib.

5.2 Datos (FIgLib)
Secuencia por muestra : ventana deslizante de L=3 frames (paso=1), balanceando
positivos (post-ignición) y negativos (pre-ignición) por evento.
Split : por evento (70/15/15).
Augmentations : cortes suaves, cambios de brillo/niebla, sin alterar la estructura de
humo; prohibir flips que invaliden contexto si hay texto/torres.
5.3 Entrenamiento (PyTorch Lightning)
Estrategia : DDP 2×A100, BF16 autocast, grad-clip 1.0, OneCycle o cosine LR.
Batch efectivo : 4–8 secuencias por GPU (tiles×frames elevan memoria); usar
accumulation para alcanzar BS_eff≈ 64.
Objetivo : Recall ≥ 0.80, TTD ≤ 4 min en val.
Pseudocódigo (LightningModule simplificado):

class SmokeyNetLike(pl.LightningModule):
def init(self, cfg):
super().init()
self.cnn = torchvision.models.resnet34(weights='IMAGENET1K_V1')
self.cnn.fc = nn.Identity()
self.lstm = nn.LSTM(input_size= 512 , hidden_size= 512 ,
num_layers= 2 ,
batch_first=True, bidirectional=True,
dropout=0.1)
self.vit = ViTEncoder(dim= 768 , depth= 6 , heads= 12 ) # tokens =
tiles
self.head_global = nn.Sequential(nn.LayerNorm( 768 ),
nn.Linear( 768 , 1 ))
self.criterion = nn.BCEWithLogitsLoss()

def forward(self, frames): # [B, L, C, H, W]

1) Tilear cada frame y pasar cada tile por la CNN → emb_tile
2) Concatenar por tiempo y pasar por LSTM (por tile) →
emb_tile_temporal

3) Construir secuencia de tokens (uno por tile) → ViT
4) [CLS] → head_global
...

6) Pipeline de inferencia en producción
Detección por imagen (cadencia de captura):
o YOLOv8 genera cajas de humo con conf bajo (p.ej., 0.25) para maximizar
recall.
Consolidación temporal (verificador) por cámara:
o Mantener buffer de los N últimos frames (p.ej., 3–5).
o Correr el SmokeyNet-like sobre la ventana ; alarma si ≥ k de N ventanas
superan p≥τ (p.ej., k=2, τ=0.5).
Salida : score global + cajas, con trazabilidad (timestamps).
Registro : métricas TTD (time-to-detect) y F1 por cámara/evento (SAI-Benchmark).
7) Recetas de descarga y preparación (bash + Python)
Usar estas guías internas como referencia de pasos y formatos :
FIgLib (descarga/armado de secuencias) y PyroSDIS + FASDD (exportar a YOLO).

7.1 PyroSDIS (Hugging Face → YOLO)
Entorno
python -m venv .venv && source .venv/bin/activate
pip install -U datasets pillow numpy huggingface_hub ultralytics

Exportación (ejemplo mínimo)
python scripts/export_pyrosdis_to_yolo.py
--hf_repo pyronear/pyro-sdis
--out data/yolo
--split train val
--single-cls 1 # humo como clase única

La guía incluye un data.yaml listo para Ultralytics y ejemplo de exportación a images/ y
labels/.

7.2 FASDD (fuente original → YOLO)
Descarga según fuente del paper/hosting y descompresión (ver guía)
bash scripts/download_pyrosdis_fasdd.sh # placeholder

Conversión a YOLO
python scripts/convert_fasdd_to_yolo.py
--src data/raw/fasdd
--dst data/yolo
--split-ratios 0.9 0.1
--map-classes smoke # o single-cls si se unifica

La guía detalla cómo normalizar anotaciones y asegurar correspondencia imagen/label.

7.3 FIgLib (script SmokeyNet)
Requisitos y descarga (ver guía; requiere listas de cámaras/timestamps
provistas)
pip install -U requests pandas
python scripts/download_figlib.py
--camera_list configs/figlib/cams.txt
--timestamps configs/figlib/timestamps.txt \

--output data/raw/figlib

Construir secuencias (ventanas L=3) y csvs por split (por evento)
python scripts/build_figlib_sequences.py
--raw-root data/raw/figlib
--out-root data/figlib_seq
--L 3 --stride 1 --split-per-event 0.7 0.15 0.

La guía explica el formato de etiquetas smoke/no-smoke y organización por evento.

8) Entrenamiento (orden de fases)
Detector YOLOv8 en PyroSDIS+FASDD (con aug Mosaic/MixUp/CutMix).
Guardar best.pt.
Verificador SmokeyNet-like en FIgLib (secuencias L=3; tiles 224; heads
auxiliares opc.). Optimizar Recall y TTD.
Integración : umbrales (τ), reglas k/N y priorización de Recall → calibrar en val y
validar en test.
9) Hiperparámetros iniciales (sugeridos)
YOLOv8-s : imgsz=960, epochs=150, batch=64 (accum si falta memoria),
lr0=0.01, weight_decay=0.0005, mosaic=1.0, mixup=0.1.
SmokeyNet-like :
o tiles=45 de 224×224; L=3; backbone=resnet34.
o optimizer=AdamW, lr=2e- 4 , wd=0.05, sched=cosine, epochs=60– 80.
o amp=bf16, ddp=2 gpus, grad_clip=1.0, accum para BS_eff≈ 64.
Basado en evidencia interna y replicando el patrón de SmokeyNet (CNN+LSTM+ViT con
heads auxiliares).

10) Métricas y criterios de aceptación
Detector : mAP@0.5, Recall (priorizar objetos pequeños).
Verificador : F1 , Recall , TTD (minutos desde primera aparición). Objetivo inicial:
Recall ≥ 0.80 y TTD ≤ 4 min en validación (línea con SmokeyNet).
11) Exportación e integración
Detector : exportar a ONNX/TensorRT para inferencia en servidor.
Verificador : TorchScript/ONNX (secuencial).
SAI-CAM/SAI-Server : la arquitectura de sistema y BENCHMARK internos ya
contemplan ingestión, colas, y reportes reproducibles (YAML + JSON).
12) Riesgos y mi,gaciones
Dominio : FIgLib no tiene cajas; por eso se separa detector (PyroSDIS+FASDD)
del verificador (FIgLib). Mezcla de dominios se maneja en calibración de
umbrales.
Falsos positivos (nubes/niebla) : consolidación temporal (k/N), heads por tile , aug
de haze.
Complejidad : entrenar por etapas y validar cada módulo de forma independiente
antes de fusionar.
13) Checklist “entregar a code assistant”
Ejecutar descargas siguiendo guías y scripts; exportar PyroSDIS+FASDD a
YOLO; construir FIgLib en secuencias.
Entrenar YOLOv8-s/m (DDP 2×A100) y guardar best.pt.
Entrenar SmokeyNet-like (CNN+LSTM+ViT) en FIgLib ; registrar TTD y Recall.
Integrar pipeline detector→verificador; calibrar (τ, k/N).
Empaquetar export (ONNX/TensorRT) y escribir inference runner con colas.
Documentar rutas y data.yaml; dejar configs versionadas.

SmokeyNet (CNN+LSTM+ViT) —F1≈82.6 % y TTD≈3.1 min en FIgLib; uso de
tiles 224×224 y supervisión auxiliar. (Chernyavsky et al., 2022, resumido en
informes del SAI).
YOLOv8 (anchor-free, C2f) —eleva precisión y mantiene latencia baja en GPU
para humo/fuego. (Síntesis en informe SAI).
