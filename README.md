# MultiNet — Clasificación de Mamografías MIAS

Proyecto de clasificación de imágenes médicas usando el dataset **MIAS (Mammographic Image Analysis Society) v1.21**. Se comparan tres arquitecturas de redes neuronales convolucionales con transfer learning, y se evalúa el impacto de la segmentación con U-Net en la precisión de clasificación.

---

## Objetivo

Clasificar mamografías en tres categorías:

| Clase | Descripción |
|-------|-------------|
| `normal` | Sin anomalías detectadas |
| `benigno` | Anomalía presente, no maligna |
| `maligno` | Anomalía maligna |

---

## Dataset

**MIAS Database v1.21** — 322 mamografías en formato PGM (1024×1024 px, escala de grises).

Distribución original:

| Clase | N | % |
|-------|---|---|
| Normal | 207 | 64.3% |
| Benigno | 63 | 19.6% |
| Maligno | 52 | 16.1% |

> El dataset no se incluye en este repositorio. Puede obtenerse en [MIAS Database](http://peipa.essex.ac.uk/info/mias.html). Una vez descargado, coloca los 322 archivos `.pgm` en `MIASDBV1.21/`.

---

## Estructura del proyecto

```
Practicas/
├── MIASDBV1.21/          ← 322 archivos .pgm (no incluidos en el repo)
├── data/                 ← generado por los scripts
│   ├── images/           ← PNGs 224×224
│   ├── images_augmented/ ← train/val/test por clase
│   ├── splits/           ← train.csv, val.csv, test.csv
│   └── mias_labels.csv   ← etiquetas completas
├── results/              ← modelos, métricas y gráficas
├── prepare_mias.py       ← Fase 1: preparación de datos
├── split_and_augment.py  ← Fase 2: split estratificado + augmentation
├── train_classifiers.py  ← Fase 3: entrenamiento y comparación de modelos
└── README.md
```

---

## Pipeline

### Fase 1 — Preparación de datos (`prepare_mias.py`)
- Parsea las anotaciones del README oficial de MIAS
- Convierte imágenes PGM → PNG y redimensiona a 224×224
- Genera `mias_labels.csv` con etiquetas y metadatos

### Fase 2 — Split y Augmentation (`split_and_augment.py`)
- Split estratificado 70/15/15 (train/val/test)
- Data augmentation en train para clases minoritarias (benigno y maligno hasta ~180 imágenes)
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
2. **Fine-tuning** (20 épocas): último 30% del backbone descongelado, lr=1e-4

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
pip install tensorflow-cpu scikit-learn matplotlib pandas numpy Pillow
```

> Para entrenamiento con GPU reemplaza `tensorflow-cpu` por `tensorflow`.

---

## Uso

```bash
# Activar entorno
venv_mias\Scripts\activate

# Ejecutar en orden
python prepare_mias.py
python split_and_augment.py
python train_classifiers.py
```

Los resultados se guardan en `results/`:
- `<modelo>_best.keras` — mejor checkpoint
- `<modelo>_confusion_matrix.png`
- `<modelo>_training_curves.png`
- `comparacion_modelos.csv` — tabla comparativa

---

## Resultados preliminares *(prueba rápida, 2 épocas)*

| Modelo | Accuracy | F1 Macro | AUC-ROC |
|--------|----------|----------|---------|
| MobileNetV2 | 0.549 | 0.418 | 0.552 |
| ResNet50 | — | — | — |
| EfficientNetB3 | — | — | — |

> Resultados completos pendientes de entrenamiento en GPU.

---

## Referencia

J Suckling et al (1994) *"The Mammographic Image Analysis Society Digital Mammogram Database"* — Excerpta Medica, International Congress Series 1069, pp375-378.
