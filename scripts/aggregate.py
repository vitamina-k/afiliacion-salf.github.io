import csv, json, os, glob, unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

BASE = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE / "input"
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

GAZETTEER_PATH = DATA_DIR / "municipios_es.csv"  # opcional pero recomendado

# --------- Normalización básica ---------
def norm_basic(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # normalizar unicode, espacios múltiples y apóstrofes “raros”
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("´", "'").replace("’", "'").replace("`", "'")
    s = " ".join(s.split())
    return s

def title_soft(s: str) -> str:
    # Title case suave (sin tocar “d', del, de, la…” demasiado)
    if not s: return s
    parts = s.split()
    out = []
    lowers = {"de","del","la","las","los","y","e","o","u","da","do","das","dos"}
    for w in parts:
        wl = w.lower()
        if wl in lowers:
            out.append(wl)
        elif "'" in wl:
            # p.ej. d'uixó → D'Uixó
            sub = []
            for t in wl.split("'"):
                sub.append(t.capitalize() if t else t)
            out.append("'".join(sub))
        else:
            out.append(wl.capitalize())
    return " ".join(out)

def normalize_city(s: str) -> str:
    s = norm_basic(s)
    # corrige rápido cosas típicas sin gazetteer:
    quick = {
        "valència": "Valencia",
        "torrente": "Torrent",
        "vila-real": "Villarreal",
        "vilareal": "Villarreal",
        "vinaros": "Vinaròs",
        "vinarós": "Vinaròs",
        "almazora": "Almassora",
        "la vall de uxó": "La Vall D'uixó",
        "la vall d´uixó": "La Vall D'uixó",
        "la vall d’uixó": "La Vall D'uixó",
    }
    low = s.lower()
    if low in quick: return quick[low]
    return title_soft(s)

def normalize_prov(s: str) -> str:
    s = norm_basic(s)
    return title_soft(s)

# --------- Carga Gazetteer ---------
def load_gazetteer():
    if not GAZETTEER_PATH.exists():
        return None, None
    g = pd.read_csv(GAZETTEER_PATH, dtype=str).fillna("")
    # columnas esperadas: provincia, municipio [, lat, lon]
    g["provincia_n"] = g["provincia"].map(normalize_prov)
    g["municipio_n"] = g["municipio"].map(normalize_city)
    # map provinc->list of city names
    prov_to_muns = {}
    for prov, sub in g.groupby("provincia_n"):
        prov_to_muns[prov] = sorted(sub["municipio_n"].unique().tolist())
    # set global cities list
    all_cities = sorted(g["municipio_n"].unique().tolist())
    return prov_to_muns, all_cities

PROV_TO_MUNS, ALL_CITIES = load_gazetteer()

# --------- Fuzzy matching ---------
# Umbrales: >=90 corrige, 80-89 sugiere, <80 pendiente
THRESH_FIX = 90
THRESH_SUGGEST = 80

def best_match(city: str, provincia: str):
    """
    Devuelve (match_text, score, scope), donde:
    - match_text: texto sugerido/corregido o None
    - score: 0-100
    - scope: 'provincia' si matcheó dentro de la provincia, 'nacional' si global, 'none'
    """
    if not city:
        return None, 0, "none"
    # primero dentro de la misma provincia si hay gazetteer
    if PROV_TO_MUNS and provincia in PROV_TO_MUNS:
        choices = PROV_TO_MUNS[provincia]
        match = process.extractOne(
            city, choices, scorer=fuzz.WRatio
        )
        if match:
            return match[0], match[1], "provincia"
    # luego nacional (si hay gazetteer)
    if ALL_CITIES:
        match = process.extractOne(
            city, ALL_CITIES, scorer=fuzz.WRatio
        )
        if match:
            return match[0], match[1], "nacional"
    # si no hay gazetteer, no podemos sugerir
    return None, 0, "none"

# --------- Lectura flexible de CSVs ---------
def parse_csv_line(line: str):
    out, cur, q = [], "", False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            if q and i + 1 < len(line) and line[i+1] == '"':
                cur += '"'; i += 1
            else:
                q = not q
        elif ch == ',' and not q:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
        i += 1
    out.append(cur.strip())
    return out

def iter_input_rows():
    for path in INPUT_DIR.rglob("*.csv"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            continue
        header = [h.lower() for h in parse_csv_line(lines[0])]
        start = 1 if ("ciudad" in header and "provincia" in header) else 0
        for ln in lines[start:]:
            cols = parse_csv_line(ln)
            # esperamos como mínimo: ..., ciudad, provincia (últimos dos campos)
            if len(cols) < 2:
                continue
            ciudad = cols[-2]
            provincia = cols[-1]
            yield provincia, ciudad

# --------- Main aggregation ---------
def main():
    agg = defaultdict(int)
    sugerencias = []  # filas dudosas/sugeridas
    pendientes = []   # no sugerible con mínima confianza

    for prov_raw, city_raw in iter_input_rows():
        prov_n = normalize_prov(prov_raw)
        city_n = normalize_city(city_raw)

        # Si tenemos gazetteer, intentamos ajustar con fuzzy
        match_text, score, scope = best_match(city_n, prov_n)

        if match_text and score >= THRESH_FIX:
            # corregir automáticamente
            city_final = match_text
            corrected = True
            suggest = ""
        elif match_text and THRESH_SUGGEST <= score < THRESH_FIX:
            # no corregimos, pero sugerimos
            city_final = city_n
            corrected = False
            suggest = match_text
            sugerencias.append({
                "provincia": prov_n,
                "ciudad_original": city_n,
                "sugerencia": suggest,
                "confianza": score,
                "ambito": scope
            })
        else:
            # sin sugerencia útil
            city_final = city_n
            corrected = False
            suggest = ""
            pendientes.append({
                "provincia": prov_n,
                "ciudad_original": city_n
            })

        agg[(prov_n, city_final)] += 1

    # Salidas
    rows = sorted(
        [(prov, city, n) for (prov, city), n in agg.items()],
        key=lambda x: (-x[2], x[0], x[1])
    )

    # CSV agregado
    out_csv = DATA_DIR / "agregado.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["provincia", "ciudad", "personas"])
        for prov, city, n in rows:
            w.writerow([prov, city, n])

    # JSON agregado
    out_json = DATA_DIR / "agregado.json"
    with open(out_json, "w", encoding="utf-8") as out:
        json.dump(
            [{"provincia": prov, "ciudad": city, "personas": n} for prov, city, n in rows],
            out, ensure_ascii=False, indent=2
        )

    # Sugerencias (si las hay)
    if sugerencias:
        sug_csv = DATA_DIR / "sugerencias.csv"
        pd.DataFrame(sugerencias).to_csv(sug_csv, index=False, encoding="utf-8")

    # Pendientes (si los hay)
    if pendientes:
        pen_csv = DATA_DIR / "pendientes.csv"
        pd.DataFrame(pendientes).to_csv(pen_csv, index=False, encoding="utf-8")

    print(f"✔ Agregado: {out_csv} y {out_json}")
    if sugerencias:
        print(f"ℹ Sugerencias: {len(sugerencias)} → data/sugerencias.csv")
    if pendientes:
        print(f"ℹ Pendientes: {len(pendientes)} → data/pendientes.csv")

if __name__ == "__main__":
    main()
