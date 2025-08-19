import csv, json, unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

BASE = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE / "input"
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

# Gazetteer con dos columnas clave por municipio:
# provincia,municipio_es,municipio_local[,lat,lon]
GAZETTEER_PATH = DATA_DIR / "municipios_es.csv"

# ---------- utilidades ----------
def norm_basic(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    # unificar apostrofes raros
    s = s.replace("´", "'").replace("’", "'").replace("`", "'").replace("‘", "'")
    # espacios
    s = " ".join(s.split())
    return s

def norm_key(s: str) -> str:
    # clave robusta: sin tildes, minúsculas, sin dobles espacios
    s = norm_basic(s)
    s = unidecode(s).lower()
    return " ".join(s.split())

def title_soft(s: str) -> str:
    if not s: return s
    parts = s.split()
    lowers = {"de","del","la","las","los","y","e","o","u","da","do","das","dos","d"}
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

# ---------- generador de alias a partir de (castellano, local) ----------
def gen_aliases(name: str) -> set:
    """
    Dado un nombre, genera variantes:
    - sin tildes/diacríticos
    - l·l -> ll (ela geminada) y variante "l·l"→"ll"
    - apostrofes normalizados y eliminados
    - guiones↔espacios y sin separadores
    - abreviaturas Sant/San/Sta.
    - artículos L'/La, D'/De
    """
    base = norm_basic(name)
    variants = set([base])

    # 1) sin diacríticos
    variants.add(unidecode(base))

    # 2) ela geminada
    v_e = base.replace("l·l", "ll").replace("L·L", "LL").replace("·", "")
    variants.add(v_e); variants.add(unidecode(v_e))

    # 3) apostrofes variantes y sin apóstrofe
    ap_vars = set()
    for v in list(variants):
        ap_vars.add(v.replace("'", ""))            # sin apostrofe
        ap_vars.add(v.replace(" d'", " de ").replace(" l'", " la ").replace(" s'", " sa "))
        ap_vars.add(v.replace(" D'", " De ").replace(" L'", " La ").replace(" S'", " Sa "))
    variants |= ap_vars

    # 4) guiones/espacios y sin separadores
    sep_vars = set()
    for v in list(variants):
        sep_vars.add(v.replace("-", " "))
        sep_vars.add(v.replace(" ", "-"))
        sep_vars.add(v.replace("-", "").replace(" ", ""))
    variants |= sep_vars

    # 5) Sant/San/Sta abreviaturas usuales
    repl_map = {
        "St. ": "San ",
        "Sta. ": "Santa ",
        "St ": "San ",
        "Sta ": "Santa ",
        "Sant ": "San ",   # castellano usa "San"
        "Sants ": "San ",  # aproximación para errores comunes
    }
    abbr_vars = set()
    for v in list(variants):
        vv = v
        for a,b in repl_map.items():
            vv = vv.replace(a, b)
        abbr_vars.add(vv)
    variants |= abbr_vars

    # 6) normalizar capitalización suave
    variants = { title_soft(v) for v in variants }

    return { v for v in variants if v }

def build_alias_dict(gdf: pd.DataFrame):
    """
    Construye:
      alias_map[(provincia, alias_norm)] -> municipio_es (castellano)
      by_prov[provincia] -> set de municipios_es
      all_muns -> lista nacional
    """
    alias_map = {}
    by_prov = defaultdict(set)

    for _, r in gdf.iterrows():
        prov = normalize_prov(r["provincia"])
        mun_es  = title_soft(norm_basic(r["municipio_es"]))
        mun_loc = title_soft(norm_basic(r.get("municipio_local", "") or ""))

        if not prov or not mun_es: 
            continue

        aliases = set()
        aliases |= gen_aliases(mun_es)
        if mun_loc:
            aliases |= gen_aliases(mun_loc)

        # guardar alias -> canónico en castellano
        for a in aliases:
            alias_map[(prov, norm_key(a))] = mun_es

        by_prov[prov].add(mun_es)

    all_muns = sorted({m for s in by_prov.values() for m in s})
    return alias_map, by_prov, all_muns

def load_gazetteer():
    if not GAZETTEER_PATH.exists():
        return None
    g = pd.read_csv(GAZETTEER_PATH, dtype=str).fillna("")
    return build_alias_dict(g)

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

    if GAZ:
        alias_map, by_prov, all_muns = GAZ

        # 1) alias exacto (con variantes generadas)
        hit = alias_map.get((prov_n, norm_key(city_raw)))
        if hit:
            return hit, "", 100, "alias"

        # 2) fuzzy por provincia
        choices = sorted(by_prov.get(prov_n, []))
        if choices:
            m = process.extractOne(city_raw, choices, scorer=fuzz.WRatio)
            if m:
                name, score, _ = m
                if score >= THRESH_FIX:
                    return name, "", score, "provincia"
                elif score >= THRESH_SUGGEST:
                    return city_raw, name, score, "provincia"

        # 3) fuzzy nacional
        if all_muns:
            m = process.extractOne(city_raw, all_muns, scorer=fuzz.WRatio)
            if m:
                name, score, _ = m
                if score >= THRESH_FIX:
                    return name, "", score, "nacional"
                elif score >= THRESH_SUGGEST:
                    return city_raw, name, score, "nacional"

    # 4) sin gazetteer: formateo suave
    return title_soft(city_raw), "", 0, "none"

# ---------- lectura flexible input ----------
def detect_columns(df: pd.DataFrame):
    # nombres originales y normalizados (quita BOM \ufeff, espacios, minúsculas)
    raw_cols = list(df.columns)
    cols = [str(c).replace("\ufeff","").strip().lower() for c in raw_cols]

    # sinónimos admitidos
    city_syns   = {"ciudad","localidad","municipio","poblacion","población","villa","pueblo","municipi"}
    prov_syns   = {"provincia","province","prov"}
    people_syns = {"personas","n","num","numero","número","count","cantidad","total"}

    city_col = prov_col = people_col = None
    for i, cl in enumerate(cols):
        if prov_col   is None and cl in prov_syns:   prov_col   = raw_cols[i]
        if city_col   is None and cl in city_syns:   city_col   = raw_cols[i]
        if people_col is None and cl in people_syns: people_col = raw_cols[i]

    # Si no detectamos cabecera, asumir por POSICIÓN si hay 2-3 columnas
    if city_col is None or prov_col is None:
        if len(df.columns) in (2,3):
            df.columns = ["provincia","ciudad"] + (["personas"] if len(df.columns)==3 else [])
            return "ciudad", "provincia", ("personas" if len(df.columns)==3 else None)

    if city_col and prov_col:
        return city_col, prov_col, people_col

    raise ValueError("Faltan columnas requeridas: 'provincia' y 'ciudad' (o 'localidad').")

def iter_input_rows():
    for path in INPUT_DIR.rglob("*.csv"):
        # Leer con autodetección de separador y quitando BOM
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                sep=None,              # auto-sniff: coma, punto y coma, tab, etc.
                engine="python",
                encoding="utf-8-sig",  # elimina BOM si lo hay
            ).dropna(how="all")
        except Exception:
            # Reintentos conservadores con separadores comunes
            df = pd.DataFrame()
            for sep in [",",";","\t","|"]:
                try:
                    df = pd.read_csv(path, dtype=str, sep=sep, engine="python", encoding="utf-8-sig").dropna(how="all")
                    if not df.empty: break
                except Exception:
                    pass

        if df.empty:
            print(f"⚠️ No se pudo leer {path} o está vacío")
            continue

        # Detectar columnas (tolerante a BOM/sinónimos/sin cabecera)
        city_col, prov_col, people_col = detect_columns(df)

        # Normalizar valores
        df[prov_col] = df[prov_col].astype(str).str.replace("\ufeff","", regex=False)
        df[city_col] = df[city_col].astype(str).str.replace("\ufeff","", regex=False)

        # personas: si no hay, cuenta 1 por fila
        if people_col and people_col in df.columns:
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
