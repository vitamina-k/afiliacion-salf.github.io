import csv, json, os, unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

BASE = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE / "input"
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

GAZETTEER_PATH = DATA_DIR / "municipios_es.csv"  # columnas: provincia,municipio_es,municipio_local,lat,lon,aliases(opc)

# ---------- utilidades ----------
def norm_basic(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("´", "'").replace("’", "'").replace("`", "'")
    s = " ".join(s.split())
    return s

def norm_key(s: str) -> str:
    # clave robusta para diccionarios: sin tildes, minúsculas, sin dobles espacios
    s = norm_basic(s)
    s = unidecode(s).lower()
    return " ".join(s.split())

def title_soft(s: str) -> str:
    if not s: return s
    parts = s.split()
    lowers = {"de","del","la","las","los","y","e","o","u","da","do","das","dos"}
    out=[]
    for w in parts:
        wl = w.lower()
        if wl in lowers: out.append(wl)
        elif "'" in wl:
            out.append("'".join([t.capitalize() if t else t for t in wl.split("'")]))
        else:
            out.append(wl.capitalize())
    return " ".join(out)

def normalize_prov(s: str) -> str:
    return title_soft(norm_basic(s))

# ---------- cargar gazetteer ----------
def load_gazetteer():
    if not GAZETTEER_PATH.exists():
        return None
    g = pd.read_csv(GAZETTEER_PATH, dtype=str).fillna("")
    # esperado: provincia, municipio_es, municipio_local, lat, lon, aliases(opcional)
    g["provincia_n"] = g["provincia"].map(normalize_prov)
    g["mun_es"]      = g["municipio_es"].map(title_soft)
    g["mun_local"]   = g["municipio_local"].map(title_soft)

    # construir diccionario: (provincia, alias_norm) -> mun_es
    alias_map = {}
    by_prov = defaultdict(set)

    for _, r in g.iterrows():
        prov = r["provincia_n"]
        mun_es = r["mun_es"]
        mun_local = r["mun_local"]
        if not prov or not mun_es: 
            continue

        # alias base: mun_es, mun_local
        aliases = {mun_es, mun_local}
        # aliases extras
        extra = [a.strip() for a in (r.get("aliases") or "").split("|") if a.strip()]
        aliases.update(extra)

        for a in aliases:
            if not a: continue
            alias_map[(prov, norm_key(a))] = mun_es
            by_prov[prov].add(mun_es)

    # lista nacional para fuzzy de respaldo
    all_muns = sorted({m for s in by_prov.values() for m in s})
    return alias_map, by_prov, all_muns

GAZ = load_gazetteer()

# fuzzy thresholds
THRESH_FIX = 90
THRESH_SUGGEST = 80

def canonical_city(prov: str, raw_city: str):
    """Devuelve (city_out, suggestion or '', score, scope) normalizando al castellano si es posible."""
    prov_n = normalize_prov(prov)
    city_raw = norm_basic(raw_city)
    if not city_raw:
        return "", "", 0, "none"

    # 1) diccionario exacto por alias (preferido)
    if GAZ:
        alias_map, by_prov, all_muns = GAZ
        key = norm_key(city_raw)
        hit = alias_map.get((prov_n, key))
        if hit:
            return hit, "", 100, "alias"

        # 2) fuzzy por provincia
        choices = sorted(by_prov.get(prov_n, []))
        if choices:
            match = process.extractOne(city_raw, choices, scorer=fuzz.WRatio)
            if match:
                mname, score, _ = match
                if score >= THRESH_FIX:
                    return mname, "", score, "provincia"
                elif score >= THRESH_SUGGEST:
                    return city_raw, mname, score, "provincia"

        # 3) fuzzy nacional
        if all_muns:
            match = process.extractOne(city_raw, all_muns, scorer=fuzz.WRatio)
            if match:
                mname, score, _ = match
                if score >= THRESH_FIX:
                    return mname, "", score, "nacional"
                elif score >= THRESH_SUGGEST:
                    return city_raw, mname, score, "nacional"

    # 4) sin gazetteer: solo formateo suave
    return title_soft(city_raw), "", 0, "none"

# ---------- lectura flexible input ----------
def detect_columns(df: pd.DataFrame):
    cols = [c.strip().lower() for c in df.columns]
    # ciudad / localidad
    city_col = "ciudad" if "ciudad" in cols else ("localidad" if "localidad" in cols else None)
    prov_col = "provincia" if "provincia" in cols else None
    people_col = "personas" if "personas" in cols else None
    if not city_col or not prov_col:
        raise ValueError("Faltan columnas requeridas: 'provincia' y 'ciudad' (o 'localidad').")
    return city_col, prov_col, people_col

def iter_input_rows():
    for path in INPUT_DIR.rglob("*.csv"):
        try:
            df = pd.read_csv(path, dtype=str).dropna(how="all")
        except Exception:
            # fallback muy permisivo
            df = pd.read_csv(path, dtype=str, sep=",", engine="python").dropna(how="all")
        if df.empty: 
            continue
        city_col, prov_col, people_col = detect_columns(df)
        df[prov_col] = df[prov_col].astype(str)
        df[city_col] = df[city_col].astype(str)

        if people_col and people_col in df:
            df["__personas__"] = pd.to_numeric(df[people_col], errors="coerce").fillna(0).astype(int)
        else:
            df["__personas__"] = 1

        for _, r in df.iterrows():
            yield r[prov_col], r[city_col], int(r["__personas__"])

# ---------- main ----------
def main():
    agg = defaultdict(int)
    sugerencias = []
    pendientes = []

    any_row = False
    for prov_raw, city_raw, n in iter_input_rows():
        any_row = True
        prov_n = normalize_prov(prov_raw)
        city_out, suggest, score, scope = canonical_city(prov_n, city_raw)

        if suggest and THRESH_SUGGEST <= score < THRESH_FIX:
            sugerencias.append({
                "provincia": prov_n,
                "ciudad_original": norm_basic(city_raw),
                "sugerencia": suggest,
                "confianza": score,
                "ambito": scope
            })
        elif not city_out:
            pendientes.append({
                "provincia": prov_n,
                "ciudad_original": norm_basic(city_raw)
            })

        final_city = city_out if city_out else title_soft(norm_basic(city_raw))
        agg[(prov_n, final_city)] += max(n, 0)

    if not any_row:
        print("⚠️ No se cargaron datos en input/**")
        return

    rows = sorted(
        [(prov, city, num) for (prov, city), num in agg.items()],
        key=lambda x: (-x[2], x[0], x[1])
    )

    # salida CSV
    out_csv = DATA_DIR / "agregado.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["provincia", "ciudad", "personas"])
        for prov, city, num in rows:
            w.writerow([prov, city, num])

    # salida JSON
    out_json = DATA_DIR / "agregado.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            [{"provincia": prov, "ciudad": city, "personas": num} for prov, city, num in rows],
            f, ensure_ascii=False, indent=2
        )

    # sugerencias y pendientes
    if sugerencias:
        pd.DataFrame(sugerencias).to_csv(DATA_DIR / "sugerencias.csv", index=False, encoding="utf-8")
    if pendientes:
        pd.DataFrame(pendientes).to_csv(DATA_DIR / "pendientes.csv", index=False, encoding="utf-8")

    print(f"✔ Agregado: {out_csv} y {out_json}")
    if sugerencias:
        print(f"ℹ Sugerencias: {len(sugerencias)} → data/sugerencias.csv")
    if pendientes:
        print(f"ℹ Pendientes: {len(pendientes)} → data/pendientes.csv")

if __name__ == "__main__":
    main()
