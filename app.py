import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import mlflow

# Configure page
st.set_page_config(page_title="Truck Price Model", layout="wide")

# ----------------------------
# Model location (MLflow artifact)
# ----------------------------
MODEL_URI = os.getenv("MODEL_URI", "./model")

# ----------------------------
# Helpers (must match training)
# ----------------------------
def engine_family(s):
    t = (s or "").lower()
    if any(k in t for k in ["dd13","dd15","dd16","detroit"]): return "Detroit"
    if any(k in t for k in ["x15","isx","cummins"]): return "Cummins"
    if any(k in t for k in ["mx13","mx-13","paccar"]): return "PACCAR"
    if any(k in t for k in ["d13","volvo"]): return "Volvo"
    if any(k in t for k in ["mp7","mp8","mack"]): return "Mack"
    if "cat" in t or re.search(r"\bc(7|9|10|12|13|15)\b", t): return "Caterpillar"
    return "Other/Unknown"

def model_series_and_variant(x):
    if not isinstance(x, str) or not x.strip():
        return "Unknown", np.nan
    m = re.match(r"([A-Za-z]+)\s*(\d{2,3})?", x.strip())
    if not m:
        return x.strip().upper(), np.nan
    base = m.group(1).upper()
    num = float(m.group(2)) if m.group(2) else np.nan
    return base, num

def transmission_simple_from(text_a, text_b):
    t = (text_a or "") + " " + (text_b or "")
    t = t.lower()
    if re.search(r"automatic|allison", t): return "Automatic"
    if re.search(r"automated|auto[- ]?shift|ultrashift|i[- ]?shift|m[- ]?drive|dt12", t): return "Automated Manual"
    return "Manual"

def sleeper_norm(s):
    s = (s or "").lower()
    if re.search(r"raised|hi[- ]?rise|high\s*roof|hi\s*rise", s): return "raised_roof"
    if re.search(r"mid[- ]?roof|midroof|\bmr\b", s): return "mid_roof"
    if re.search(r"flat[- ]?top|flat\s*top|flattop", s): return "flat_top"
    if "sleeper" in s: return "other"
    return ""

SELECTED_FEATURES = [
    # numeric engineered
    "age_years","mileage","log_mileage","miles_per_year","horsepower","bunk_count","model_variant_num",
    # categoricals
    "manufacturer","model_series","sleeper_type","transmission_simple","engine_family","transmission_make","state",
    # bool-ish numeric
    "has_apu"
]

def _ensure_feature_frame(row_dict_or_df) -> pd.DataFrame:
    if isinstance(row_dict_or_df, dict):
        df = pd.DataFrame([row_dict_or_df])
    else:
        df = row_dict_or_df.copy()
    for k in SELECTED_FEATURES:
        if k not in df.columns:
            df[k] = (
                "" if k in {"manufacturer","model_series","sleeper_type",
                            "transmission_simple","engine_family",
                            "transmission_make","state"}
                else np.nan
            )
    return df.reindex(columns=SELECTED_FEATURES, copy=False)

def featurize_single(
    year:int, mileage:float, horsepower:float, bunk_count:int,
    manufacturer:str, model_text:str, sleeper_type_raw:str,
    transmission:str, transmission_type:str, transmission_make:str,
    engine_make:str, engine_model:str, state:str, has_apu_bool:bool
) -> pd.DataFrame:
    year = int(year) if year else np.nan
    mileage = float(mileage) if mileage is not None else np.nan
    horsepower = float(horsepower) if horsepower is not None else np.nan
    bunk_count = float(bunk_count) if bunk_count is not None else np.nan
    has_apu = 1.0 if has_apu_bool else 0.0

    from datetime import date
    THIS_YEAR = date.today().year
    age_years = max(0, THIS_YEAR - year) if year == year else np.nan
    log_mileage = np.log1p(mileage) if mileage == mileage else np.nan
    miles_per_year = (mileage / max(age_years, 0.5)) if (mileage == mileage and age_years == age_years) else np.nan

    model_series, model_variant_num = model_series_and_variant(model_text)
    sleeper_type = sleeper_norm(sleeper_type_raw)
    engine_fam = engine_family(engine_make if (engine_make or "").strip() else engine_model)
    trans_simple = transmission_simple_from(transmission, transmission_type)

    row = {
        "age_years": age_years,
        "mileage": mileage,
        "log_mileage": log_mileage,
        "miles_per_year": miles_per_year,
        "horsepower": horsepower,
        "bunk_count": bunk_count,
        "model_variant_num": model_variant_num,

        "manufacturer": (manufacturer or "").strip().upper(),
        "model_series": model_series,
        "sleeper_type": sleeper_type,
        "transmission_simple": trans_simple,
        "engine_family": engine_fam,
        "transmission_make": (transmission_make or "").strip().upper(),
        "state": (state or "").strip().upper(),

        "has_apu": has_apu
    }
    return _ensure_feature_frame(row)

def featurize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for c in ["year","mileage","horsepower","bunk_count","manufacturer","model","sleeper_type",
              "transmission","transmission_type","transmission_make","engine_make","engine_model",
              "state","has_apu"]:
        if c not in df.columns:
            df[c] = np.nan

    from datetime import date
    THIS_YEAR = date.today().year
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df["bunk_count"] = pd.to_numeric(df["bunk_count"], errors="coerce")
    df["has_apu"] = df["has_apu"].map({True:1, False:0, "true":1, "false":0, 1:1, 0:0}).fillna(0)

    df["age_years"] = (THIS_YEAR - df["year"]).clip(lower=0)
    df["log_mileage"] = np.log1p(df["mileage"].fillna(0))
    df["miles_per_year"] = df["mileage"] / (df["age_years"].replace(0, 0.5))

    df["manufacturer"] = df["manufacturer"].astype(str).str.upper()
    df["transmission_make"] = df["transmission_make"].astype(str).str.upper()
    df["state"] = df["state"].astype(str).str.upper()

    ser, var = [], []
    for m in df["model"].astype(str).tolist():
        s, v = model_series_and_variant(m)
        ser.append(s); var.append(v)
    df["model_series"] = ser
    df["model_variant_num"] = var

    df["sleeper_type"] = df["sleeper_type"].apply(sleeper_norm)

    df["engine_family"] = np.where(
        df["engine_make"].astype(str).str.len() > 0,
        df["engine_make"].apply(engine_family),
        df["engine_model"].apply(engine_family)
    )

    df["transmission_simple"] = [
        transmission_simple_from(a, b) for a, b in zip(df["transmission"], df["transmission_type"])
    ]

    out = _ensure_feature_frame(df)
    out["bunk_count"] = pd.to_numeric(out["bunk_count"], errors="coerce")
    return out

# ----------------------------
# Load MLflow model (cache) — use a distinct name
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model(uri: str):
    return mlflow.pyfunc.load_model(uri)

ml_model = load_model(MODEL_URI)

def predict_logprice(df_features: pd.DataFrame) -> np.ndarray:
    preds = ml_model.predict(df_features)
    if isinstance(preds, (list, tuple)): preds = np.array(preds)
    if isinstance(preds, pd.Series): preds = preds.values
    return np.asarray(preds).reshape(-1)

# ----------------------------
# UI
# ----------------------------
st.title("🚛 Truck Price Prediction")

st.sidebar.header("Model")
st.sidebar.write(f"Model URI: `{MODEL_URI}`")
mape_pct = st.sidebar.slider("Display ±MAPE band (%)", min_value=2.0, max_value=20.0, value=6.7, step=0.1)
st.sidebar.markdown("""
**Tips**
- Unknown categories are handled (One-Hot ignores unknowns).
- Inputs match your training features.
- Use *Batch Score* for CSVs.
""")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Compare Two", "Batch Score CSV"])

# ----------------------------
# Tab 1: Single Prediction
# ----------------------------
with tab1:
    st.subheader("Single Prediction")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        year = st.number_input("Year", min_value=1990, max_value=2030, value=2017, step=1)
        mileage = st.number_input("Mileage (mi)", min_value=0, value=750000, step=1000)
        horsepower = st.number_input("Horsepower", min_value=200, max_value=800, value=450, step=10)
        bunk_count = st.selectbox("Bunk count", [0,1,2], index=1)
    with colB:
        manufacturer = st.text_input("Manufacturer", "FREIGHTLINER")
        truck_model = st.text_input("Model", "CASCADIA 125")  # renamed
        sleeper_type_raw = st.selectbox("Sleeper type", ["Raised Roof Sleeper","Mid Roof Sleeper","Flat Top Sleeper","Other"], index=1)
        has_apu_bool = st.checkbox("Has APU", value=False)
    with colC:
        transmission = st.text_input("Transmission", "Eaton Fuller")
        transmission_type = st.text_input("Transmission Type", "Manual")
        transmission_make = st.text_input("Transmission Make", "EATON-FULLER")
    with colD:
        engine_make = st.text_input("Engine Make", "DETROIT")
        engine_model = st.text_input("Engine Model", "DD15")
        state = st.text_input("State (2-letter)", "MO")

    if st.button("Predict Price"):
        X = featurize_single(year, mileage, horsepower, bunk_count,
                             manufacturer, truck_model, sleeper_type_raw,
                             transmission, transmission_type, transmission_make,
                             engine_make, engine_model, state, has_apu_bool)
        try:
            pred_log = predict_logprice(X)      # model predicts log(price)
            y_pred = np.exp(pred_log)           # convert back to dollars
            price = float(y_pred[0])
            st.success(f"Predicted price: **${price:,.0f}**")

            lo, hi = price*(1 - mape_pct/100.0), price*(1 + mape_pct/100.0)
            st.caption(f"Approx. ±MAPE band: ${lo:,.0f} – ${hi:,.0f} (display only)")
            st.write("Features sent to model:")
            st.dataframe(X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------------------
# Tab 2: Compare Two
# ----------------------------
with tab2:
    st.subheader("Compare Two Trucks")

    def input_block(prefix, container):
        with container:
            year = st.number_input(f"{prefix} Year", 1990, 2030, 2017, key=prefix+"y")
            mileage = st.number_input(f"{prefix} Mileage", 0, 2_000_000, 750000, step=1000, key=prefix+"m")
            hp = st.number_input(f"{prefix} HP", 200, 800, 450, step=10, key=prefix+"h")
            bunks = st.selectbox(f"{prefix} Bunks", [0,1,2], index=1, key=prefix+"b")
            man = st.text_input(f"{prefix} Manufacturer", "FREIGHTLINER", key=prefix+"man")
            mod = st.text_input(f"{prefix} Model", "CASCADIA 125", key=prefix+"mod")
            slp = st.selectbox(f"{prefix} Sleeper", ["Raised Roof Sleeper","Mid Roof Sleeper","Flat Top Sleeper","Other"], index=1, key=prefix+"sl")
            apu = st.checkbox(f"{prefix} Has APU", value=False, key=prefix+"apu")
            tr = st.text_input(f"{prefix} Transmission", "Eaton Fuller", key=prefix+"tr")
            trt = st.text_input(f"{prefix} Transmission Type", "Manual", key=prefix+"trt")
            trm = st.text_input(f"{prefix} Transmission Make", "EATON-FULLER", key=prefix+"trm")
            engm = st.text_input(f"{prefix} Engine Make", "DETROIT", key=prefix+"em")
            engmd = st.text_input(f"{prefix} Engine Model", "DD15", key=prefix+"emd")
            stt = st.text_input(f"{prefix} State", "MO", key=prefix+"st")
            return featurize_single(year, mileage, hp, bunks, man, mod, slp, tr, trt, trm, engm, engmd, stt, apu)

    c1, c2 = st.columns(2)
    X1 = input_block("A", c1)
    X2 = input_block("B", c2)

    if st.button("Compare"):
        try:
            p1 = float(np.exp(predict_logprice(X1))[0])
            p2 = float(np.exp(predict_logprice(X2))[0])
            st.success(f"A: **${p1:,.0f}**    |    B: **${p2:,.0f}**    →    Δ = **${(p2-p1):,.0f}** ({(p2/p1-1)*100:+.1f}%)")
            both = pd.concat([X1.assign(_which="A", _pred=p1), X2.assign(_which="B", _pred=p2)], ignore_index=True)
            st.dataframe(both)
        except Exception as e:
            st.error(f"Compare failed: {e}")

# ----------------------------
# Tab 3: Batch Score
# ----------------------------
with tab3:
    st.subheader("Batch Score a CSV")
    st.caption("Upload a CSV with columns like: year, mileage, horsepower, bunk_count, manufacturer, model, sleeper_type, transmission, transmission_type, transmission_make, engine_make, engine_model, state, has_apu.")

    template = pd.DataFrame({
        "year":[2017], "mileage":[750000], "horsepower":[450], "bunk_count":[2],
        "manufacturer":["FREIGHTLINER"], "model":["CASCADIA 125"], "sleeper_type":["Mid Roof Sleeper"],
        "transmission":["Eaton Fuller"], "transmission_type":["Manual"], "transmission_make":["EATON-FULLER"],
        "engine_make":["DETROIT"], "engine_model":["DD15"], "state":["MO"], "has_apu":[False]
    })
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="truckprice_template.csv",
        mime="text/csv"
    )

    up = st.file_uploader("Choose CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        st.write("Raw preview:")
        st.dataframe(raw.head(10))

        MAX_ROWS = 10000
        if len(raw) > MAX_ROWS:
            st.warning(f"Trimming to first {MAX_ROWS} rows to keep things snappy on Streamlit Cloud.")
            raw = raw.head(MAX_ROWS)

        Xb = featurize_dataframe(raw)
        try:
            preds_log = predict_logprice(Xb)
            preds = np.exp(preds_log)
            out = raw.copy()
            out["predicted_price"] = preds
            st.success("Scored!")
            st.dataframe(out.head(20))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv, file_name="scored_trucks.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")
