import csv, json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE / "input"
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- utilidades ---
CITY_SYNS   = {"ciudad","localidad","municipio","poblacion","población","villa","pueblo","municipi"}
PROV_SYNS   = {"provincia","province","prov"}
PEOPLE_SYNS = {"personas","n","num","numero","número","count","cantidad","total"}

def _norm_cols(cols):
    return [str(c).replace("\ufeff","").strip().lower() for c in cols]

def detect_columns(df: pd.DataFrame):
    raw = list(df.columns)
    low = _norm_cols(raw)

    city_col = prov_col = people_col = None
    for i,cl in enumerate(low):
        if prov_col   is None and cl in PROV_SYNS:   prov_col   = raw[i]
        if city_col   is None and cl in CITY_SYNS:   city_col   = raw[i]
        if people_col is None and cl in PEOPLE_SYNS: people_col = raw[i]

    if city_col and prov_col:
        return city_col, prov_col, people_col

    # Si no hay cabecera reconocible pero hay 2-3 columnas, asumimos orden:
    if len(df.columns) in (2,3):
        cols = ["provincia","ciudad"] + (["personas"] if len(df.columns)==3 else [])
        df.columns = cols
        return "ciudad","provincia", ("personas" if len(df.columns)==3 else None)

    raise ValueError("Faltan columnas requeridas: 'provincia' y 'ciudad' (o 'localidad').")

def read_csv_flex(path: Path) -> pd.DataFrame:
    # 1) autodetectar separador y quitar BOM
    try:
        df = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame()
        for sep in [",",";","\t","|"]:
            try:
                df = pd.read_csv(path, dtype=str, sep=sep, engine="python", encoding="utf-8-sig")
                if not df.empty: break
            except Exception:
                pass
    return df.dropna(how="all")

def iter_rows():
    for p in INPUT_DIR.rglob("*.csv"):
        df = read_csv_flex(p)
        if df.empty:
            print(f"⚠️ {p}: vacío o ilegible"); continue
        try:
            city_col, prov_col, people_col = detect_columns(df)
        except Exception as e:
            print(f"⚠️ {p}: {e} (columnas={list(df.columns)})"); continue

        # limpiar posibles BOM en valores
        df[prov_col] = df[prov_col].astype(str).str.replace("\ufeff","", regex=False).str.strip()
        df[city_col] = df[city_col].astype(str).str.replace("\ufeff","", regex=False).str.strip()

        if people_col and people_col in df.columns:
            df["__personas__"] = pd.to_numeric(df[people_col], errors="coerce").fillna(0).astype(int)
        else:
            df["__personas__"] = 1

        for _, r in df.iterrows():
            prov = str(r[prov_col]).strip()
            city = str(r[city_col]).strip()
            n    = int(r["__personas__"])
            if not prov or not city: 
                continue
            yield prov, city, n

def main():
    from collections import defaultdict
    agg = defaultdict(int)
    any_row = False

    for prov, city, n in iter_rows():
        any_row = True
        # formateo suave
        prov = prov.title()
        city = " ".join(city.split())
        city = city[:1].upper() + city[1:] if city else city
        agg[(prov, city)] += max(n, 0)

    if not any_row:
        # generar salidas vacías pero válidas
        (DATA_DIR / "agregado.csv").write_text("provincia,ciudad,personas\n", encoding="utf-8")
        (DATA_DIR / "agregado.json").write_text("[]\n", encoding="utf-8")
        print("⚠️ No se cargaron datos en input/** — generado agregado vacío.")
        return

    rows = sorted([(p,c,n) for (p,c),n in agg.items()],
                  key=lambda x: (-x[2], x[0], x[1]))

    # CSV
    with open(DATA_DIR/"agregado.csv","w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["provincia","ciudad","personas"])
        for p,c,n in rows: w.writerow([p,c,n])

    # JSON
    with open(DATA_DIR/"agregado.json","w", encoding="utf-8") as f:
        json.dump([{"provincia":p,"ciudad":c,"personas":n} for p,c,n in rows],
                  f, ensure_ascii=False, indent=2)

    print(f"✅ Generados data/agregado.csv y data/agregado.json ({len(rows)} filas)")

if __name__ == "__main__":
    main()
