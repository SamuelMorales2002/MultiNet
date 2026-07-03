"""
Split estratificado + Augmentation - MIAS Database
Requiere haber ejecutado prepare_mias.py primero.

Estructura de salida:
    practicas/
    └── data/
        ├── mias_labels.csv
        ├── splits/
        │   ├── train.csv
        │   ├── val.csv
        │   └── test.csv
        └── images_augmented/
            ├── normal/
            ├── benigno/
            └── maligno/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import random
import shutil

# ─── RUTAS ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("C:/Users/Samuel/Desktop/Practicas")   # <-- AJUSTA SI ES NECESARIO
DATA_DIR     = BASE_DIR / "data"
CSV_PATH     = DATA_DIR / "cbis_labels.csv"
IMG_DIR      = DATA_DIR / "images"
SPLITS_DIR   = DATA_DIR / "splits"
AUG_DIR      = DATA_DIR / "images_augmented"

SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO  = 0.15 (el resto)

# Cuántas imágenes aumentadas generar por imagen de la clase minoritaria
# Con CBIS-DDSM el desbalance es menor que en MIAS, pero se deja el mecanismo
AUG_TARGET   = 500

random.seed(SEED)
np.random.seed(SEED)

# ─── AUGMENTATION ─────────────────────────────────────────────────────────────
def augment_image(img: Image.Image) -> Image.Image:
    """Aplica transformaciones aleatorias a una imagen."""
    ops = []

    # Flip horizontal
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # Flip vertical
    if random.random() > 0.5:
        img = ImageOps.flip(img)

    # Rotación aleatoria ±20°
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, fillcolor=(0, 0, 0))

    # Brillo aleatorio
    factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contraste aleatorio
    factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Zoom aleatorio (crop + resize)
    if random.random() > 0.5:
        w, h = img.size
        margin = int(min(w, h) * 0.1)
        left   = random.randint(0, margin)
        top    = random.randint(0, margin)
        right  = w - random.randint(0, margin)
        bottom = h - random.randint(0, margin)
        img    = img.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)

    return img

# ─── SPLIT ESTRATIFICADO POR PACIENTE ─────────────────────────────────────────
def stratified_split(df, train_ratio, val_ratio, seed):
    """
    Divide por patient_id (no por imagen): un mismo paciente puede tener
    varias vistas (CC/MLO, izq/der) y todas deben quedar en el mismo split,
    o el modelo "ve" el caso en train y lo evalúa en test.
    """
    # Un paciente puede tener imágenes de ambas clases en casos raros;
    # se asigna la clase mayoritaria del paciente para estratificar.
    patient_label = (df.groupby("patient_id")["label"]
                        .agg(lambda x: x.value_counts().idxmax()))

    train_ids, val_ids, test_ids = [], [], []

    for label, group in patient_label.groupby(patient_label):
        ids = group.index.to_series().sample(frac=1, random_state=seed).tolist()
        n       = len(ids)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_ids += ids[:n_train]
        val_ids   += ids[n_train:n_train + n_val]
        test_ids  += ids[n_train + n_val:]

    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df

# ─── COPIAR IMÁGENES A CARPETA POR CLASE ──────────────────────────────────────
def copy_split_images(df, split_name, aug_dir):
    for _, row in df.iterrows():
        src  = IMG_DIR / row["png_filename"]
        dest = aug_dir / row["label"] / row["png_filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Total imágenes: {len(df)}")

    # ── 1. Split estratificado ────────────────────────────────────────────────
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = stratified_split(df, TRAIN_RATIO, VAL_RATIO, SEED)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR  / "val.csv",   index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv",  index=False)

    print(f"\nSplit estratificado:")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split["label"].value_counts().to_dict()
        print(f"  {name:5s} ({len(split):3d}): {dist}")

    # ── 2. Copiar imágenes originales de train a carpetas por clase ───────────
    TRAIN_AUG_DIR = AUG_DIR / "train"
    VAL_DIR       = AUG_DIR / "val"
    TEST_DIR      = AUG_DIR / "test"

    for d in [TRAIN_AUG_DIR, VAL_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    copy_split_images(train_df, "train", TRAIN_AUG_DIR)
    copy_split_images(val_df,   "val",   VAL_DIR)
    copy_split_images(test_df,  "test",  TEST_DIR)

    # ── 3. Augmentation solo en train para clases minoritarias ────────────────
    aug_log = []
    class_counts = train_df["label"].value_counts()
    majority_n   = class_counts.max()
    minority_labels = [lbl for lbl in class_counts.index if class_counts[lbl] < majority_n]

    for label in minority_labels:
        label_df    = train_df[train_df["label"] == label]
        label_dir   = TRAIN_AUG_DIR / label
        current_n   = len(label_df)
        needed      = max(0, AUG_TARGET - current_n)

        print(f"\n  Augmentation '{label}': {current_n} originales → generando {needed} adicionales")

        sources = list(label_df["png_filename"])
        generated = 0

        while generated < needed:
            src_name = random.choice(sources)
            src_path = IMG_DIR / src_name

            img = Image.open(src_path).convert("RGB")
            aug = augment_image(img)

            aug_name = f"aug_{generated:04d}_{src_name}"
            aug.save(label_dir / aug_name)

            aug_log.append({
                "png_filename": aug_name,
                "label":        label,
                "source":       src_name,
                "augmented":    True,
            })
            generated += 1

    # ── 4. Resumen final ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Distribución final en TRAIN (orig + aug):")
    for label in class_counts.index:
        orig = len(train_df[train_df["label"] == label])
        aug  = sum(1 for r in aug_log if r["label"] == label)
        print(f"    {label:10s}: {orig:3d} orig + {aug:3d} aug = {orig+aug:3d}")

    print(f"\n  Carpetas generadas en: {AUG_DIR}")
    print(f"  CSVs de splits en    : {SPLITS_DIR}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
