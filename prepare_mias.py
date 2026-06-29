"""
Preparación de datos - MIAS Database v1.21
Estructura esperada:
    practicas/
    ├── MIASDBV1.21/        ← 322 archivos .pgm
    └── prepare_mias.py     ← este script

Salida generada:
    practicas/
    ├── data/
    │   ├── images/         ← PNGs redimensionados (224x224)
    │   └── mias_labels.csv ← etiquetas y metadatos
"""

import os
import re
import csv
import shutil
from pathlib import Path
from PIL import Image

# ─── RUTAS ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("C:/Users/Samuel/Desktop/Practicas")   # <-- AJUSTA ESTA RUTA
PGM_DIR     = BASE_DIR / "MIASDBV1.21"
OUTPUT_DIR  = BASE_DIR / "data"
IMG_OUT_DIR = OUTPUT_DIR / "images"
CSV_PATH    = OUTPUT_DIR / "mias_labels.csv"
README_PATH = BASE_DIR / "00README.txt"               # si tienes el .txt original

IMG_SIZE = (224, 224)

# ─── DATOS DEL README (extraídos manualmente del PDF) ─────────────────────────
# Formato: "filename": (tissue, abnormality, severity, x, y, radius)
# None = campo ausente / no aplica
README_DATA = {
    "mdb001": ("G","CIRC","B",1815,1116,790),
    "mdb002": ("G","CIRC","B",3091,1262,277),
    "mdb003": ("D","NORM",None,None,None,None),
    "mdb004": ("D","NORM",None,None,None,None),
    "mdb005": ("F","CIRC","B",647,1163,122),    # múltiple: se toma la primera
    "mdb006": ("F","NORM",None,None,None,None),
    "mdb007": ("G","NORM",None,None,None,None),
    "mdb008": ("G","NORM",None,None,None,None),
    "mdb009": ("F","NORM",None,None,None,None),
    "mdb010": ("F","CIRC","B",2509,975,135),
    "mdb011": ("F","NORM",None,None,None,None),
    "mdb012": ("F","CIRC","B",2378,1467,162),
    "mdb013": ("G","MISC","B",1574,1923,127),
    "mdb014": ("G","NORM",None,None,None,None),
    "mdb015": ("G","CIRC","B",3571,1359,275),
    "mdb016": ("G","NORM",None,None,None,None),
    "mdb017": ("G","CIRC","B",2407,943,192),
    "mdb018": ("G","NORM",None,None,None,None),
    "mdb019": ("G","CIRC","B",2021,1864,197),
    "mdb020": ("G","NORM",None,None,None,None),
    "mdb021": ("G","CIRC","B",612,1224,197),
    "mdb022": ("G","NORM",None,None,None,None),
    "mdb023": ("G","CIRC","M",2837,1405,117),
    "mdb024": ("G","NORM",None,None,None,None),
    "mdb025": ("F","CIRC","B",1886,1948,318),
    "mdb026": ("F","NORM",None,None,None,None),
    "mdb027": ("F","NORM",None,None,None,None),
    "mdb028": ("F","CIRC","M",2953,1999,224),
    "mdb029": ("G","NORM",None,None,None,None),
    "mdb030": ("G","MISC","B",1505,1785,174),
    "mdb031": ("G","NORM",None,None,None,None),
    "mdb032": ("G","MISC","B",1243,1798,267),
    "mdb033": ("D","NORM",None,None,None,None),
    "mdb034": ("D","NORM",None,None,None,None),
    "mdb035": ("D","NORM",None,None,None,None),
    "mdb036": ("D","NORM",None,None,None,None),
    "mdb037": ("D","NORM",None,None,None,None),
    "mdb038": ("D","NORM",None,None,None,None),
    "mdb039": ("D","NORM",None,None,None,None),
    "mdb040": ("D","NORM",None,None,None,None),
    "mdb041": ("G","NORM",None,None,None,None),
    "mdb042": ("G","NORM",None,None,None,None),
    "mdb043": ("G","NORM",None,None,None,None),
    "mdb044": ("G","NORM",None,None,None,None),
    "mdb045": ("G","NORM",None,None,None,None),
    "mdb046": ("G","NORM",None,None,None,None),
    "mdb047": ("G","NORM",None,None,None,None),
    "mdb048": ("G","NORM",None,None,None,None),
    "mdb049": ("G","NORM",None,None,None,None),
    "mdb050": ("G","NORM",None,None,None,None),
    "mdb051": ("G","NORM",None,None,None,None),
    "mdb052": ("G","NORM",None,None,None,None),
    "mdb053": ("D","NORM",None,None,None,None),
    "mdb054": ("D","NORM",None,None,None,None),
    "mdb055": ("G","NORM",None,None,None,None),
    "mdb056": ("G","NORM",None,None,None,None),
    "mdb057": ("D","NORM",None,None,None,None),
    "mdb058": ("D","MISC","M",2774,2079,110),
    "mdb059": ("F","CIRC","B",None,None,None),  # sin coordenadas
    "mdb060": ("F","NORM",None,None,None,None),
    "mdb061": ("D","NORM",None,None,None,None),
    "mdb062": ("D","NORM",None,None,None,None),
    "mdb063": ("D","MISC","B",1967,1163,133),
    "mdb064": ("D","NORM",None,None,None,None),
    "mdb065": ("D","NORM",None,None,None,None),
    "mdb066": ("D","NORM",None,None,None,None),
    "mdb067": ("D","NORM",None,None,None,None),
    "mdb068": ("D","NORM",None,None,None,None),
    "mdb069": ("F","CIRC","B",1739,1101,177),
    "mdb070": ("F","NORM",None,None,None,None),
    "mdb071": ("G","NORM",None,None,None,None),
    "mdb072": ("G","ASYM","M",2140,2011,115),
    "mdb073": ("G","NORM",None,None,None,None),
    "mdb074": ("G","NORM",None,None,None,None),
    "mdb075": ("F","ASYM","M",2982,850,92),
    "mdb076": ("F","NORM",None,None,None,None),
    "mdb077": ("F","NORM",None,None,None,None),
    "mdb078": ("F","NORM",None,None,None,None),
    "mdb079": ("F","NORM",None,None,None,None),
    "mdb080": ("F","CIRC","B",3615,1344,81),
    "mdb081": ("G","ASYM","B",2007,1220,525),
    "mdb082": ("G","NORM",None,None,None,None),
    "mdb083": ("G","ASYM","B",891,1428,152),
    "mdb084": ("G","NORM",None,None,None,None),
    "mdb085": ("G","NORM",None,None,None,None),
    "mdb086": ("G","NORM",None,None,None,None),
    "mdb087": ("F","NORM",None,None,None,None),
    "mdb088": ("F","NORM",None,None,None,None),
    "mdb089": ("G","NORM",None,None,None,None),
    "mdb090": ("G","ASYM","M",2021,1035,198),
    "mdb091": ("F","CIRC","B",2090,1696,82),
    "mdb092": ("F","ASYM","M",1562,1382,175),
    "mdb093": ("G","NORM",None,None,None,None),
    "mdb094": ("G","NORM",None,None,None,None),
    "mdb095": ("F","ASYM","M",2181,1118,116),
    "mdb096": ("F","NORM",None,None,None,None),
    "mdb097": ("F","ASYM","B",1302,1702,137),
    "mdb098": ("F","NORM",None,None,None,None),
    "mdb099": ("D","ASYM","B",1473,1834,93),
    "mdb100": ("D","NORM",None,None,None,None),
    "mdb101": ("D","NORM",None,None,None,None),
    "mdb102": ("D","ASYM","M",2369,1412,152),
    "mdb103": ("D","NORM",None,None,None,None),
    "mdb104": ("D","ASYM","B",2751,1645,203),
    "mdb105": ("D","ASYM","M",1229,1318,392),
    "mdb106": ("D","NORM",None,None,None,None),
    "mdb107": ("D","ASYM","B",2597,1653,446),
    "mdb108": ("D","NORM",None,None,None,None),
    "mdb109": ("D","NORM",None,None,None,None),
    "mdb110": ("D","ASYM","M",2502,2590,205),
    "mdb111": ("D","ASYM","M",2414,1275,428),
    "mdb112": ("D","NORM",None,None,None,None),
    "mdb113": ("G","NORM",None,None,None,None),
    "mdb114": ("G","NORM",None,None,None,None),
    "mdb115": ("G","ARCH","M",2240,1096,468),
    "mdb116": ("G","NORM",None,None,None,None),
    "mdb117": ("G","ARCH","M",2417,1175,337),
    "mdb118": ("G","NORM",None,None,None,None),
    "mdb119": ("G","NORM",None,None,None,None),
    "mdb120": ("G","ARCH","M",3162,1659,319),
    "mdb121": ("G","ARCH","B",1849,1221,348),
    "mdb122": ("G","NORM",None,None,None,None),
    "mdb123": ("G","NORM",None,None,None,None),
    "mdb124": ("G","ARCH","M",1729,1609,135),
    "mdb125": ("D","ARCH","M",2322,2054,242),
    "mdb126": ("D","ARCH","B",2015,2585,93),
    "mdb127": ("G","ARCH","B",2317,1069,194),
    "mdb128": ("G","NORM",None,None,None,None),
    "mdb129": ("D","NORM",None,None,None,None),
    "mdb130": ("D","ARCH","M",2002,2469,112),
    "mdb131": ("F","NORM",None,None,None,None),
    "mdb132": ("F","CIRC","B",1499,3043,211),   # múltiple: se toma la primera
    "mdb133": ("F","NORM",None,None,None,None),
    "mdb134": ("F","MISC","M",1736,2173,199),
    "mdb135": ("F","NORM",None,None,None,None),
    "mdb136": ("F","NORM",None,None,None,None),
    "mdb137": ("D","NORM",None,None,None,None),
    "mdb138": ("D","NORM",None,None,None,None),
    "mdb139": ("F","NORM",None,None,None,None),
    "mdb140": ("F","NORM",None,None,None,None),
    "mdb141": ("F","CIRC","M",3591,1832,117),
    "mdb142": ("F","CIRC","B",2104,2662,104),
    "mdb143": ("F","NORM",None,None,None,None),
    "mdb144": ("F","MISC","M",2491,2799,108),   # múltiple: se toma la maligna
    "mdb145": ("D","SPIC","B",2726,2631,197),
    "mdb146": ("D","NORM",None,None,None,None),
    "mdb147": ("F","NORM",None,None,None,None),
    "mdb148": ("F","SPIC","M",2220,2745,699),
    "mdb149": ("F","NORM",None,None,None,None),
    "mdb150": ("F","ARCH","B",2005,2647,249),
    "mdb151": ("F","NORM",None,None,None,None),
    "mdb152": ("F","ARCH","B",2704,1349,195),
    "mdb153": ("F","NORM",None,None,None,None),
    "mdb154": ("F","NORM",None,None,None,None),
    "mdb155": ("F","ARCH","M",2032,1046,380),
    "mdb156": ("F","NORM",None,None,None,None),
    "mdb157": ("F","NORM",None,None,None,None),
    "mdb158": ("F","ARCH","M",1951,915,353),
    "mdb159": ("F","NORM",None,None,None,None),
    "mdb160": ("F","ARCH","B",2133,1206,245),
    "mdb161": ("D","NORM",None,None,None,None),
    "mdb162": ("D","NORM",None,None,None,None),
    "mdb163": ("D","ARCH","B",1574,817,202),
    "mdb164": ("D","NORM",None,None,None,None),
    "mdb165": ("D","ARCH","B",2073,903,168),
    "mdb166": ("D","NORM",None,None,None,None),
    "mdb167": ("F","ARCH","B",2740,1550,141),
    "mdb168": ("F","NORM",None,None,None,None),
    "mdb169": ("D","NORM",None,None,None,None),
    "mdb170": ("D","ARCH","M",2288,1118,331),
    "mdb171": ("D","ARCH","M",2622,1102,248),
    "mdb172": ("D","NORM",None,None,None,None),
    "mdb173": ("F","NORM",None,None,None,None),
    "mdb174": ("F","NORM",None,None,None,None),
    "mdb175": ("G","SPIC","B",2795,1344,132),
    "mdb176": ("G","NORM",None,None,None,None),
    "mdb177": ("G","NORM",None,None,None,None),
    "mdb178": ("G","SPIC","M",1810,880,280),
    "mdb179": ("D","SPIC","M",2168,1152,268),
    "mdb180": ("D","NORM",None,None,None,None),
    "mdb181": ("G","SPIC","M",1563,1052,217),
    "mdb182": ("G","NORM",None,None,None,None),
    "mdb183": ("F","NORM",None,None,None,None),
    "mdb184": ("F","SPIC","M",1712,1943,458),
    "mdb185": ("G","NORM",None,None,None,None),
    "mdb186": ("G","SPIC","M",2114,1237,191),
    "mdb187": ("G","NORM",None,None,None,None),
    "mdb188": ("G","SPIC","B",1741,1448,247),
    "mdb189": ("G","NORM",None,None,None,None),
    "mdb190": ("G","SPIC","B",1724,1302,127),
    "mdb191": ("G","SPIC","B",2177,1128,165),
    "mdb192": ("G","NORM",None,None,None,None),
    "mdb193": ("D","SPIC","B",2364,850,528),
    "mdb194": ("D","NORM",None,None,None,None),
    "mdb195": ("F","SPIC","B",631,2155,107),
    "mdb196": ("F","NORM",None,None,None,None),
    "mdb197": ("D","NORM",None,None,None,None),
    "mdb198": ("D","SPIC","B",1761,800,373),
    "mdb199": ("D","SPIC","B",820,1543,125),
    "mdb200": ("D","NORM",None,None,None,None),
    "mdb201": ("D","NORM",None,None,None,None),
    "mdb202": ("D","SPIC","M",1122,1123,149),
    "mdb203": ("F","NORM",None,None,None,None),
    "mdb204": ("F","SPIC","B",2614,2005,84),
    "mdb205": ("F","NORM",None,None,None,None),
    "mdb206": ("F","SPIC","M",3410,1876,71),
    "mdb207": ("D","SPIC","B",2370,1262,76),
    "mdb208": ("D","NORM",None,None,None,None),
    "mdb209": ("G","CALC","M",2126,1842,348),
    "mdb210": ("G","NORM",None,None,None,None),
    "mdb211": ("G","CALC","M",1423,1698,53),
    "mdb212": ("G","CALC","B",None,None,None),  # sin coordenadas
    "mdb213": ("G","CALC","M",2193,940,183),
    "mdb214": ("G","CALC","B",None,None,None),  # sin coordenadas
    "mdb215": ("D","NORM",None,None,None,None),
    "mdb216": ("D","CALC","M",None,None,None),  # nota especial
    "mdb217": ("G","NORM",None,None,None,None),
    "mdb218": ("G","CALC","B",1694,1275,35),
    "mdb219": ("G","CALC","B",3136,1439,119),
    "mdb220": ("G","NORM",None,None,None,None),
    "mdb221": ("D","NORM",None,None,None,None),
    "mdb222": ("D","CALC","B",2502,1482,70),
    "mdb223": ("D","CALC","B",2043,846,116),    # múltiple: se toma la primera
    "mdb224": ("D","NORM",None,None,None,None),
    "mdb225": ("D","NORM",None,None,None,None),
    "mdb226": ("D","CALC","B",1770,1927,31),    # múltiple: se toma la primera
    "mdb227": ("G","CALC","B",1981,993,36),
    "mdb228": ("G","NORM",None,None,None,None),
    "mdb229": ("F","NORM",None,None,None,None),
    "mdb230": ("F","NORM",None,None,None,None),
    "mdb231": ("F","CALC","M",2265,1665,179),
    "mdb232": ("F","NORM",None,None,None,None),
    "mdb233": ("G","CALC","M",None,None,None),  # nota especial
    "mdb234": ("G","NORM",None,None,None,None),
    "mdb235": ("D","NORM",None,None,None,None),
    "mdb236": ("D","CALC","B",912,2247,58),
    "mdb237": ("F","NORM",None,None,None,None),
    "mdb238": ("F","CALC","M",1998,986,70),
    "mdb239": ("D","CALC","M",3133,1833,160),   # múltiple: se toma la primera
    "mdb240": ("D","CALC","B",1752,776,95),
    "mdb241": ("D","CALC","M",2827,565,155),
    "mdb242": ("D","NORM",None,None,None,None),
    "mdb243": ("D","NORM",None,None,None,None),
    "mdb244": ("D","CIRC","B",1940,1209,209),
    "mdb245": ("F","CALC","M",None,None,None),  # nota especial
    "mdb246": ("F","NORM",None,None,None,None),
    "mdb247": ("F","NORM",None,None,None,None),
    "mdb248": ("F","CALC","B",1805,1836,42),
    "mdb249": ("D","CALC","M",2146,1154,194),   # múltiple: se toma la primera
    "mdb250": ("D","NORM",None,None,None,None),
    "mdb251": ("F","NORM",None,None,None,None),
    "mdb252": ("F","CALC","B",2743,1318,94),
    "mdb253": ("D","CALC","M",2368,2185,112),
    "mdb254": ("D","NORM",None,None,None,None),
    "mdb255": ("F","NORM",None,None,None,None),
    "mdb256": ("F","CALC","M",2272,1750,149),
    "mdb257": ("D","NORM",None,None,None,None),
    "mdb258": ("D","NORM",None,None,None,None),
    "mdb259": ("D","NORM",None,None,None,None),
    "mdb260": ("D","NORM",None,None,None,None),
    "mdb261": ("D","NORM",None,None,None,None),
    "mdb262": ("D","NORM",None,None,None,None),
    "mdb263": ("G","NORM",None,None,None,None),
    "mdb264": ("G","MISC","M",2487,691,147),
    "mdb265": ("G","MISC","M",2104,1351,242),
    "mdb266": ("G","NORM",None,None,None,None),
    "mdb267": ("F","MISC","M",2036,2427,227),
    "mdb268": ("F","NORM",None,None,None,None),
    "mdb269": ("G","NORM",None,None,None,None),
    "mdb270": ("G","CIRC","M",430,1649,291),
    "mdb271": ("F","MISC","M",1193,2391,274),
    "mdb272": ("F","NORM",None,None,None,None),
    "mdb273": ("F","NORM",None,None,None,None),
    "mdb274": ("F","MISC","M",2630,3542,495),
    "mdb275": ("G","NORM",None,None,None,None),
    "mdb276": ("G","NORM",None,None,None,None),
    "mdb277": ("G","NORM",None,None,None,None),
    "mdb278": ("G","NORM",None,None,None,None),
    "mdb279": ("G","NORM",None,None,None,None),
    "mdb280": ("G","NORM",None,None,None,None),
    "mdb281": ("D","NORM",None,None,None,None),
    "mdb282": ("D","NORM",None,None,None,None),
    "mdb283": ("D","NORM",None,None,None,None),
    "mdb284": ("D","NORM",None,None,None,None),
    "mdb285": ("D","NORM",None,None,None,None),
    "mdb286": ("D","NORM",None,None,None,None),
    "mdb287": ("D","NORM",None,None,None,None),
    "mdb288": ("D","NORM",None,None,None,None),
    "mdb289": ("D","NORM",None,None,None,None),
    "mdb290": ("D","CIRC","B",2799,1502,181),
    "mdb291": ("G","NORM",None,None,None,None),
    "mdb292": ("G","NORM",None,None,None,None),
    "mdb293": ("F","NORM",None,None,None,None),
    "mdb294": ("F","NORM",None,None,None,None),
    "mdb295": ("D","NORM",None,None,None,None),
    "mdb296": ("D","NORM",None,None,None,None),
    "mdb297": ("F","NORM",None,None,None,None),
    "mdb298": ("F","NORM",None,None,None,None),
    "mdb299": ("F","NORM",None,None,None,None),
    "mdb300": ("F","NORM",None,None,None,None),
    "mdb301": ("F","NORM",None,None,None,None),
    "mdb302": ("F","NORM",None,None,None,None),
    "mdb303": ("F","NORM",None,None,None,None),
    "mdb304": ("F","NORM",None,None,None,None),
    "mdb305": ("F","NORM",None,None,None,None),
    "mdb306": ("F","NORM",None,None,None,None),
    "mdb307": ("F","NORM",None,None,None,None),
    "mdb308": ("F","NORM",None,None,None,None),
    "mdb309": ("F","NORM",None,None,None,None),
    "mdb310": ("F","NORM",None,None,None,None),
    "mdb311": ("F","NORM",None,None,None,None),
    "mdb312": ("F","MISC","B",3158,2389,81),
    "mdb313": ("F","NORM",None,None,None,None),
    "mdb314": ("F","MISC","B",3447,1277,158),
    "mdb315": ("D","CIRC","B",1900,1317,372),
    "mdb316": ("D","NORM",None,None,None,None),
    "mdb317": ("D","NORM",None,None,None,None),
    "mdb318": ("D","NORM",None,None,None,None),
    "mdb319": ("D","NORM",None,None,None,None),
    "mdb320": ("D","NORM",None,None,None,None),
    "mdb321": ("D","NORM",None,None,None,None),
    "mdb322": ("D","NORM",None,None,None,None),
}

# ─── FUNCIÓN: derivar etiqueta final ──────────────────────────────────────────
def get_label(abnormality, severity):
    if abnormality == "NORM":
        return "normal"
    elif severity == "B":
        return "benigno"
    elif severity == "M":
        return "maligno"
    else:
        return "desconocido"  # casos sin severidad (mdb059, etc.)

# ─── FUNCIÓN: encontrar el archivo .pgm correspondiente ───────────────────────
def find_pgm(mdbid, pgm_dir):
    """El nombre real incluye sufijos como ll, rl, lm, rm, ls, rs, lx, rx."""
    for f in pgm_dir.glob(f"{mdbid}*.pgm"):
        return f
    return None

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []

    for mdbid, (tissue, abnormality, severity, x, y, radius) in README_DATA.items():
        pgm_path = find_pgm(mdbid, PGM_DIR)

        if pgm_path is None:
            print(f"[WARN] No encontrado: {mdbid}")
            skipped.append(mdbid)
            continue

        label = get_label(abnormality, severity)

        # Convertir PGM → PNG y redimensionar
        png_name = pgm_path.stem + ".png"
        png_path = IMG_OUT_DIR / png_name

        try:
            img = Image.open(pgm_path).convert("RGB")
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            img.save(png_path)
        except Exception as e:
            print(f"[ERROR] {mdbid}: {e}")
            skipped.append(mdbid)
            continue

        rows.append({
            "id":           mdbid,
            "filename":     pgm_path.name,
            "png_filename": png_name,
            "tissue":       tissue,
            "abnormality":  abnormality,
            "severity":     severity if severity else "",
            "x":            x if x is not None else "",
            "y":            y if y is not None else "",
            "radius":       radius if radius is not None else "",
            "label":        label,
        })

    # Guardar CSV
    fieldnames = ["id","filename","png_filename","tissue","abnormality","severity","x","y","radius","label"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Resumen
    print(f"\n{'='*45}")
    print(f"  Imágenes procesadas : {len(rows)}")
    print(f"  Omitidas            : {len(skipped)}")
    print(f"  CSV guardado en     : {CSV_PATH}")
    print(f"{'='*45}")

    from collections import Counter
    dist = Counter(r["label"] for r in rows)
    print(f"\n  Distribución de clases:")
    for k, v in sorted(dist.items()):
        print(f"    {k:10s}: {v:3d} ({v/len(rows)*100:.1f}%)")
    print()

if __name__ == "__main__":
    main()
