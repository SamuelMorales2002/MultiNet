"""
Preparación de datos - CBIS-DDSM
Clasificación binaria: benigno vs maligno (CBIS-DDSM no incluye casos "normal"
con ROI; los normales de DDSM son una colección aparte y quedan fuera).

Estructura esperada (tras descargar con NBIA Data Retriever):
    practicas/
    ├── CBIS-DDSM/
    │   ├── calc_case_description_train_set.csv
    │   ├── calc_case_description_test_set.csv
    │   ├── mass_case_description_train_set.csv
    │   ├── mass_case_description_test_set.csv
    │   └── <carpetas con los .dcm, una por serie>
    └── prepare_cbis.py     ← este script

Salida:
    practicas/data/
    ├── images/             ← PNGs 224x224 (imagen completa, no el crop/ROI)
    └── cbis_labels.csv     ← patient_id, label, png_filename, view, etc.
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
import pydicom
import numpy as np

# ─── RUTAS ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("C:/Users/Samuel/Desktop/Practicas")   # <-- AJUSTA SI ES NECESARIO
CBIS_DIR    = BASE_DIR / "CBIS-DDSM"
OUTPUT_DIR  = BASE_DIR / "data"
IMG_OUT_DIR = OUTPUT_DIR / "images"
CSV_PATH    = OUTPUT_DIR / "cbis_labels.csv"

IMG_SIZE = (224, 224)

CSV_FILES = [
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
]

# ─── ETIQUETA FINAL ────────────────────────────────────────────────────────────
def get_label(pathology: str):
    pathology = pathology.strip().upper()
    if pathology in ("BENIGN", "BENIGN_WITHOUT_CALLBACK"):
        return "benigno"
    elif pathology == "MALIGNANT":
        return "maligno"
    return None  # descarta valores inesperados

# ─── DICOM → PNG ───────────────────────────────────────────────────────────────
def dicom_to_png(dcm_path: Path, out_path: Path):
    ds  = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # Normalizar a 0-255 (los DICOM de CBIS-DDSM vienen en distintos rangos de bits)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img.save(out_path)

# ─── ENCONTRAR EL .dcm DE LA IMAGEN COMPLETA ──────────────────────────────────
def find_full_image_dicom(image_file_path: str, cbis_dir: Path):
    """
    'image file path' en el CSV apunta a la carpeta de la serie DICOM de la
    imagen completa (no el crop ni la máscara). Buscamos el .dcm dentro.
    """
    series_dir = cbis_dir / Path(image_file_path).parent
    if not series_dir.exists():
        return None
    dcm_files = list(series_dir.glob("*.dcm"))
    return dcm_files[0] if dcm_files else None

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    for fname in CSV_FILES:
        path = CBIS_DIR / fname
        if not path.exists():
            print(f"[WARN] No encontrado: {fname}")
            continue
        df = pd.read_csv(path)
        df["source_csv"] = fname
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total de casos en CSVs: {len(full_df)}")

    rows = []
    skipped = []

    for idx, row in full_df.iterrows():
        label = get_label(str(row["pathology"]))
        if label is None:
            skipped.append((row.get("patient_id", "?"), "pathology desconocida"))
            continue

        dcm_path = find_full_image_dicom(row["image file path"], CBIS_DIR)
        if dcm_path is None:
            skipped.append((row["patient_id"], "dicom no encontrado"))
            continue

        png_name = f"{row['patient_id']}_{row['left or right breast']}_{row['image view']}_{idx}.png"
        png_path = IMG_OUT_DIR / png_name

        try:
            dicom_to_png(dcm_path, png_path)
        except Exception as e:
            skipped.append((row["patient_id"], f"error: {e}"))
            continue

        rows.append({
            "patient_id":     row["patient_id"],   # clave para split sin leakage
            "png_filename":   png_name,
            "label":          label,
            "breast":         row["left or right breast"],
            "view":           row["image view"],
            "abnormality":    row.get("abnormality type", ""),
            "pathology_raw":  row["pathology"],
            "source_csv":     row["source_csv"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(CSV_PATH, index=False)

    print(f"\n{'='*45}")
    print(f"  Imágenes procesadas : {len(rows)}")
    print(f"  Omitidas            : {len(skipped)}")
    print(f"  Pacientes únicos    : {out_df['patient_id'].nunique()}")
    print(f"  CSV guardado en     : {CSV_PATH}")
    print(f"{'='*45}")

    dist = out_df["label"].value_counts()
    print(f"\n  Distribución de clases:")
    for k, v in dist.items():
        print(f"    {k:10s}: {v:4d} ({v/len(out_df)*100:.1f}%)")

    if skipped:
        print(f"\n  Primeros omitidos: {skipped[:5]}")

if __name__ == "__main__":
    main()
