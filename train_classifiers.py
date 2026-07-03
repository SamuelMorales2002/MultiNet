"""
Clasificación de mamografías MIAS - Fase 2
Modelos: MobileNetV2, ResNet50, EfficientNetB3
Estrategia: Transfer learning en dos etapas (cabeza → fine-tuning)

Estructura esperada:
    practicas/
    ├── data/
    │   ├── splits/train.csv, val.csv, test.csv
    │   └── images_augmented/train | val | test / <clase> / *.png
    └── train_classifiers.py   ← este script

Salida:
    practicas/results/
    ├── <modelo>_best.keras
    ├── <modelo>_history.csv
    └── comparacion_modelos.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB3
from tensorflow.keras.applications import mobilenet_v2, resnet50, efficientnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # sin interfaz gráfica

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
BASE_DIR    = Path("C:/Users/Samuel/Desktop/Practicas")   # <-- AJUSTA SI ES NECESARIO
DATA_DIR    = BASE_DIR / "data"
AUG_DIR     = DATA_DIR / "images_augmented"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
CLASSES     = ["benigno", "maligno", "normal"]   # orden alfabético = índices Keras
NUM_CLASSES = len(CLASSES)
SEED        = 42

# Función de preprocesamiento propia de cada backbone (reemplaza rescale=1./255)
PREPROCESS_FN = {
    "MobileNetV2":    mobilenet_v2.preprocess_input,
    "ResNet50":       resnet50.preprocess_input,
    "EfficientNetB3": efficientnet.preprocess_input,
}

# Épocas por etapa
EPOCHS_HEAD  = 10   # entrenar solo la cabeza
EPOCHS_FINE  = 20   # fine-tuning de capas superiores

tf.random.set_seed(SEED)

# ─── GENERADORES DE DATOS ─────────────────────────────────────────────────────
def make_generators(model_name):
    # Cada backbone requiere su propio preprocesamiento (no un rescale genérico)
    preprocess = PREPROCESS_FN[model_name]
    train_gen = ImageDataGenerator(preprocessing_function=preprocess)
    val_gen   = ImageDataGenerator(preprocessing_function=preprocess)

    train = train_gen.flow_from_directory(
        AUG_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=True,
        seed=SEED,
    )
    val = val_gen.flow_from_directory(
        AUG_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )
    test = val_gen.flow_from_directory(
        AUG_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )
    return train, val, test

# ─── CONSTRUCCIÓN DE MODELOS ──────────────────────────────────────────────────
def build_model(name: str) -> tf.keras.Model:
    input_tensor = layers.Input(shape=(*IMG_SIZE, 3))

    backbone_args = dict(
        include_top=False,
        weights="imagenet",
        input_tensor=input_tensor,
    )

    if name == "MobileNetV2":
        base = MobileNetV2(**backbone_args)
    elif name == "ResNet50":
        base = ResNet50(**backbone_args)
    elif name == "EfficientNetB3":
        base = EfficientNetB3(**backbone_args)
    else:
        raise ValueError(f"Modelo desconocido: {name}")

    base.trainable = False   # congelar backbone para etapa 1

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=input_tensor, outputs=output, name=name)
    return model, base

# ─── ENTRENAMIENTO ────────────────────────────────────────────────────────────
def train_model(name, train_gen, val_gen):
    print(f"\n{'='*55}")
    print(f"  Entrenando: {name}")
    print(f"{'='*55}")

    model, base = build_model(name)
    model_path  = RESULTS_DIR / f"{name}_best.keras"

    # Pesos de clase para compensar el desbalance (normal >> benigno/maligno)
    class_indices = train_gen.class_indices  # {"benigno":0, "maligno":1, "normal":2}
    y_train       = train_gen.classes
    weights_arr   = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = dict(zip(np.unique(y_train), weights_arr))
    print(f"  Class weights: {class_weights}")

    cb = [
        callbacks.ModelCheckpoint(str(model_path), save_best_only=True,
                                  monitor="val_loss", verbose=1),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3,
                                    monitor="val_loss", verbose=1),
    ]

    # ── Etapa 1: solo cabeza ──────────────────────────────────────────────────
    print(f"\n  [Etapa 1] Entrenando cabeza ({EPOCHS_HEAD} épocas máx)")
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS_HEAD, callbacks=cb, verbose=1,
                   class_weight=class_weights)

    # ── Etapa 2: fine-tuning capas superiores ─────────────────────────────────
    # Descongelar último 30% de capas del backbone
    n_layers     = len(base.layers)
    unfreeze_from = int(n_layers * 0.85)   # último 15% (antes 30%, muy agresivo)
    for layer in base.layers[unfreeze_from:]:
        # Las capas BatchNorm deben permanecer en modo inferencia:
        # descongelarlas en datasets pequeños rompe sus estadísticas acumuladas
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    print(f"\n  [Etapa 2] Fine-tuning ({n_layers - unfreeze_from} capas desbloqueadas, {EPOCHS_FINE} épocas máx)")
    model.compile(
        optimizer=optimizers.Adam(1e-5),   # antes 1e-4, muy alto para dataset chico
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS_FINE, callbacks=cb, verbose=1,
                   class_weight=class_weights)

    # Combinar historiales
    history = {}
    for k in h1.history:
        history[k] = h1.history[k] + h2.history[k]

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / f"{name}_history.csv", index=False)

    return model, history

# ─── EVALUACIÓN ───────────────────────────────────────────────────────────────
def evaluate_model(model, test_gen, name):
    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = test_gen.classes

    # Métricas
    report = classification_report(y_true, y_pred,
                                   target_names=CLASSES, output_dict=True)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # AUC-ROC multiclase (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc = roc_auc_score(y_true_bin, y_pred_prob,
                        multi_class="ovr", average="macro")

    acc = report["accuracy"]

    print(f"\n  [{name}] Test accuracy: {acc:.4f} | F1 macro: {f1_macro:.4f} | AUC-ROC: {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASSES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión - {name}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()

    # Curvas de entrenamiento
    hist_path = RESULTS_DIR / f"{name}_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(hist["accuracy"], label="train")
        axes[0].plot(hist["val_accuracy"], label="val")
        axes[0].set_title(f"{name} - Accuracy"); axes[0].legend()
        axes[1].plot(hist["loss"], label="train")
        axes[1].plot(hist["val_loss"], label="val")
        axes[1].set_title(f"{name} - Loss"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{name}_training_curves.png", dpi=150)
        plt.close()

    return {"modelo": name, "accuracy": acc,
            "f1_macro": f1_macro, "auc_roc": auc}

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    model_names = ["MobileNetV2", "ResNet50", "EfficientNetB3"]
    resultados  = []

    for name in model_names:
        # Generadores propios por modelo: cada backbone requiere su preprocess_input
        print(f"\nCargando generadores de datos para {name}...")
        train_gen, val_gen, test_gen = make_generators(name)
        print(f"  Train batches : {len(train_gen)}  ({train_gen.samples} imágenes)")
        print(f"  Val batches   : {len(val_gen)}  ({val_gen.samples} imágenes)")
        print(f"  Test batches  : {len(test_gen)}  ({test_gen.samples} imágenes)")

        model, history = train_model(name, train_gen, val_gen)
        metrics = evaluate_model(model, test_gen, name)
        resultados.append(metrics)

        # Liberar memoria GPU/CPU entre modelos
        del model
        tf.keras.backend.clear_session()

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    comp_df = pd.DataFrame(resultados)
    comp_df.to_csv(RESULTS_DIR / "comparacion_modelos.csv", index=False)

    print(f"\n{'='*55}")
    print("  COMPARACIÓN FINAL (sin segmentación)")
    print(f"{'='*55}")
    print(comp_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n  Resultados guardados en: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
