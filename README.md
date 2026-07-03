# MultiNet — Clasificación de Mamografías CBIS-DDSM

Proyecto de clasificación de imágenes médicas usando el dataset **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**. Se comparan tres arquitecturas de redes neuronales convolucionales con transfer learning, y se evalúa el impacto de la segmentación con U-Net en la precisión de clasificación.

---

## Objetivo

Clasificar mamografías en dos categorías:

| Clase | Descripción |
|-------|-------------|
| `benigno` | Anomalía presente, no maligna |
| `maligno` | Anomalía maligna |

> CBIS-DDSM no incluye casos "normal" con anotación (esos vienen de una colección aparte de DDSM), por lo que el proyecto pasó de 3 a 2 clases.

---

## Dataset

**CBIS-DDSM** — ~3,103 imágenes DICOM (1,566 pacientes), curadas por un mamógrafo experto. Incluye casos de calcificaciones y masas, con vistas CC y MLO.

Se descarga desde [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) usando el NBIA Data Retriever, junto con los 4 CSVs de metadatos (`calc_case_description_*`, `mass_case_description_*`).

> El dataset no se incluye en este repositorio. Una vez descargado, coloca los CSVs y las carpetas DICOM en `CBIS-DDSM/`.

---

## Estructura del proyecto

```
Practicas/
├── CBIS-DDSM/            ← DICOMs + CSVs oficiales (no incluidos en el repo)
├── data/                 ← generado por los scripts
│   ├── images/           ← PNGs 224×224
│   ├── images_augmented/ ← train/val/test por clase
│   ├── splits/           ← train.csv, val.csv, test.csv
│   └── cbis_labels.csv   ← etiquetas completas
├── results/              ← modelos, métricas y gráficas
├── prepare_cbis.py       ← Fase 1: preparación de datos
├── split_and_augment.py  ← Fase 2: split por paciente + augmentation
├── train_classifiers.py  ← Fase 3: entrenamiento y comparación de modelos
└── README.md
```

---

## Pipeline

### Fase 1 — Preparación de datos (`prepare_cbis.py`)
- Lee los CSVs oficiales de CBIS-DDSM (calcificaciones y masas)
- Convierte DICOM → PNG y redimensiona a 224×224
- Mapea patología a etiqueta binaria (benigno/maligno) y genera `cbis_labels.csv` con `patient_id`

### Fase 2 — Split y Augmentation (`split_and_augment.py`)
- Split estratificado 70/15/15 **por paciente** (no por imagen, evita data leakage entre vistas CC/MLO del mismo caso)
- Data augmentation en train para la clase minoritaria
- Transformaciones: flip horizontal/vertical, rotación ±20°, variación de brillo/contraste, zoom aleatorio

### Fase 3 — Clasificación (`train_classifiers.py`)
Tres modelos con transfer learning desde ImageNet:

| Modelo | Backbone |
|--------|----------|
| MobileNetV2 | Ligero, eficiente |
| ResNet50 | Residual connections |
| EfficientNetB3 | Escalado compuesto |

Estrategia de entrenamiento en dos etapas:
1. **Cabeza** (10 épocas): backbone congelado, lr=1e-3
2. **Fine-tuning** (20 épocas): último 15% del backbone descongelado (BatchNorm excluida), lr=1e-5

Preprocesamiento específico por backbone (`preprocess_input` de cada modelo) y `class_weight` balanceado.

Métricas reportadas: Accuracy, F1 macro, AUC-ROC

### Fase 4 — Segmentación con U-Net *(próximamente)*
- Generación de máscaras desde coordenadas del dataset
- Entrenamiento de U-Net
- Re-clasificación sobre imágenes segmentadas
- Comparación de resultados con/sin segmentación

---

## Instalación

```bash
# Requiere Python 3.9–3.11
py -3.11 -m venv venv_mias
venv_mias\Scripts\activate
pip install tensorflow-cpu scikit-learn matplotlib pandas numpy Pillow pydicom
```

> Para entrenamiento con GPU reemplaza `tensorflow-cpu` por `tensorflow`.

---

## Uso

```bash
# Activar entorno
venv_mias\Scripts\activate

# Ejecutar en orden
python prepare_cbis.py
python split_and_augment.py
python train_classifiers.py
```

Los resultados se guardan en `results/`:
- `<modelo>_best.keras` — mejor checkpoint
- `<modelo>_confusion_matrix.png`
- `<modelo>_training_curves.png`
- `comparacion_modelos.csv` — tabla comparativa

---

## Resultados

Pendientes de entrenamiento sobre CBIS-DDSM. Los resultados previos sobre MIAS (322 imágenes, 3 clases) quedaron obsoletos tras la migración de dataset.

---

## Referencia

Lee, R.S., Gimenez, F., Hoogi, A. et al. *"A curated mammography data set for use in computer-aided detection and diagnosis research."* Sci Data 4, 170177 (2017).
