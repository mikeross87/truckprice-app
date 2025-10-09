import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import mlflow

# ---------------------------------------
# Page config
# ---------------------------------------
st.set_page_config(page_title="Sleeper Truck Retail Price Prediction", layout="wide")
st.title("Sleeper Truck Retail Price Prediction")

# ---------------------------------------
# Model location (MLflow artifact)
# ---------------------------------------
MODEL_URI = os.getenv("MODEL_URI", "./model")  # path to MLflow model folder in repo or models:/... URI

# ---------------------------------------
# Helpers (must match training logic)
# ---------------------------------------
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
    # text_a = "Transmission" (we now pass ""), text_b = "Transmission Type"
    t = (text_a or "") + " " + (text_b or "")
    t = t.lower()
    if re.search(r"automatic|allison", t): return "Automatic"
    if re.search(r"automated|auto[- ]?shift|ultrashift|i[- ]?shift|m[- ]?drive|dt12", t): return "Automated Manual"
    return "Manual"  # default used during training as well

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

def _to_float_or_nan(x):
    try:
        if x is None: return np.nan
        s = str(x).strip()
        if s == "": return np.nan
        return float(s.replace(",", ""))
    except Exception:
        return np.nan

def _to_int_or_nan(x):
    try:
        if x is None: return np.nan
        s = str(x).strip()
        if s == "": return np.nan
        return int(float(s.replace(",", "")))
    except Exception:
        return np.nan

def featurize_single(
    year, mileage, horsepower, bunk_count,
    manufacturer, model_text, sleeper_type_raw,
    transmission, transmission_type, transmission_make,
    engine_make, engine_model, state, has_apu_bool
) -> pd.DataFrame:
    # Convert flexible/blank inputs to numbers or NaN
    year = _to_int_or_nan(year)
    mileage = _to_float_or_nan(mileage)
    horsepower = _to_float_or_nan(horsepower)
    bunk_count = _to_float_or_nan(bunk_count)
    has_apu = 1.0 if bool(has_apu_bool) else 0.0

    from datetime import date
    THIS_YEAR = date.today().year
    age_years = max(0, THIS_YEAR - year) if year == year else np.nan
    log_mileage = np.log1p(mileage) if mileage == mileage else np.nan
    miles_per_year = (mileage / max(age_years, 0.5)) if (mileage == mileage and age_years == age_years) else np.nan

    model_series, model_variant_num = model_series_and_variant(model_text or "")
    sleeper_type = sleeper_norm(sleeper_type_raw)
    engine_fam = engine_family((engine_make or "").strip() or (engine_model or ""))
    trans_simple = transmission_simple_from(transmission or "", transmission_type or "")

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

# (Batch featurizer kept for completeness; not used in this trimmed app view)
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

# ---------------------------------------
# Load MLflow model (cached)
# ---------------------------------------
@st.cache_resource(show_spinner=True)
def load_model(uri: str):
    return mlflow.pyfunc.load_model(uri)

ml_model = load_model(MODEL_URI)

def predict_logprice(df_features: pd.DataFrame) -> np.ndarray:
    preds = ml_model.predict(df_features)
    if isinstance(preds, (list, tuple)): preds = np.array(preds)
    if isinstance(preds, pd.Series): preds = preds.values
    return np.asarray(preds).reshape(-1)

# ---------------------------------------
# Session-state defaults (blank-friendly)
# ---------------------------------------
DEFAULTS = {
    "sp_year": "", "sp_mileage": "", "sp_hp": "", "sp_bunks": "",
    "sp_manufacturer": "", "sp_model": "",
    "sp_sleeper": "", "sp_apu": False,
    # we removed "Transmission" free text; we keep type + make:
    "sp_tr_type": "", "sp_tr_make": "",
    "sp_engine_make": "", "sp_engine_model": "", "sp_state": ""
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Fixed display band (formerly slider on sidebar)
MAPE_DISPLAY_PCT = 6.7

# ---------------------------------------
# Single Prediction (form: no rerun on each change)
# ---------------------------------------
with st.form("single_form", clear_on_submit=False):
    st.markdown("Fill in what you know. Any field may be left blank.")

    colA, colB, colC = st.columns(3)

    with colA:
        st.text_input("Year", key="sp_year", placeholder="e.g., 2017")
        st.text_input("Mileage (mi)", key="sp_mileage", placeholder="e.g., 750000")
        st.text_input("Horsepower", key="sp_hp", placeholder="e.g., 450")
        st.text_input("Bunk count", key="sp_bunks", placeholder="0, 1, or 2")

    with colB:
        st.text_input("Manufacturer", key="sp_manufacturer", placeholder="e.g., FREIGHTLINER")
        st.text_input("Model", key="sp_model", placeholder="e.g., CASCADIA 125")
        sleeper_opts = ["", "Raised Roof Sleeper", "Mid Roof Sleeper", "Flat Top Sleeper", "Other"]
        # use selectbox for sleeper with a blank option
        idx = sleeper_opts.index(st.session_state.sp_sleeper) if st.session_state.sp_sleeper in sleeper_opts else 0
        st.selectbox("Sleeper type", sleeper_opts, index=idx, key="sp_sleeper",
                     help="If unknown, leave blank. Examples: Mid Roof Sleeper, Raised Roof Sleeper, Flat Top Sleeper.")
        st.checkbox("Has APU", key="sp_apu")

    with colC:
        # Removed "Transmission" free text (duplicative). Keep Type + Make:
        tr_type_opts = ["", "Manual", "Automated Manual", "Automatic"]
        idx = tr_type_opts.index(st.session_state.sp_tr_type) if st.session_state.sp_tr_type in tr_type_opts else 0
        st.selectbox(
            "Transmission Type",
            tr_type_opts,
            index=idx,
            key="sp_tr_type",
            help="Manual (driver shifts); Automated Manual (AMT: UltraShift, DT12, I-Shift, mDRIVE); Automatic (torque-converter, e.g., Allison)."
        )
        st.text_input(
            "Transmission Make",
            key="sp_tr_make",
            placeholder="e.g., EATON-FULLER, ALLISON, DETROIT, VOLVO, MACK",
            help="Brand/manufacturer. Examples: EATON-FULLER (many manuals/AMTs), ALLISON (automatics), DETROIT (DT12), VOLVO (I-SHIFT), MACK (mDRIVE)."
        )
        st.text_input("Engine Make", key="sp_engine_make", placeholder="e.g., DETROIT")
        st.text_input("Engine Model", key="sp_engine_model", placeholder="e.g., DD15")
        st.text_input("State (2-letter)", key="sp_state", placeholder="e.g., MO")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    X = featurize_single(
        st.session_state.sp_year,
        st.session_state.sp_mileage,
        st.session_state.sp_hp,
        st.session_state.sp_bunks,
        st.session_state.sp_manufacturer,
        st.session_state.sp_model,
        st.session_state.sp_sleeper,
        "",  # transmission free-text (removed from UI)
        st.session_state.sp_tr_type,
        st.session_state.sp_tr_make,
        st.session_state.sp_engine_make,
        st.session_state.sp_engine_model,
        st.session_state.sp_state,
        st.session_state.sp_apu,
    )
    try:
        pred_log = predict_logprice(X)
        y_pred = np.exp(pred_log)
        price = float(y_pred[0])
        st.success(f"Predicted price: **${price:,.0f}**")
        lo, hi = price*(1 - MAPE_DISPLAY_PCT/100.0), price*(1 + MAPE_DISPLAY_PCT/100.0)
        st.caption(f"Approx. ±MAPE band: ${lo:,.0f} – ${hi:,.0f} (display only)")
        with st.expander("View engineered features sent to the model"):
            st.dataframe(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
