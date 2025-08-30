# app.py
# ============================================================
# 🏆 MLB Home Run Predictor — "Max Power" Edition (Tuned, No-Tuner UI)
#
# - Robust feature pruning & pinning
# - Target encoding with time-embargoed OOF
# - Tree ensemble (XGB/LGB/CB) + meta stacker (LR)
# - Isotonic calibration + top-K temperature tuning
# - Context overlays (weather + weak/slumping pitcher + short hot streak)
# - Day-wise ranker head (LambdaRank)
# - NEW: Uncertainty-aware overlay shrinkage (always on)
# - NEW: Reciprocal Rank Fusion auxiliary rank (always on)
# - NEW: Model-disagreement penalty (always on)
# - Tuned blend weights baked in (NO Blend Tuner; no plots)
# - Adds proxy outputs: prob_2tb (2+ Total Bases) and prob_rbi
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gc, time, psutil, pickle
from datetime import timedelta
from collections import defaultdict

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from scipy.special import logit, expit

# ===== Tuned blend weights (locked) =====
DEFAULT_WEIGHTS = dict(
    w_prob=0.3879,     # calibrated/base prob term
    w_overlay=0.0161,  # contextual overlay term
    w_ranker=0.4461,   # LambdaRank/learned ranker term
    w_rrf=0.1416,      # Reciprocal Rank Fusion auxiliary term
    w_penalty=0.0083   # model-disagreement penalty (subtracted)
)

# ===================== UI =====================
st.set_page_config(page_title="🏆 MLB Home Run Predictor — Max Power", layout="wide")
st.title("🏆 MLB Home Run Predictor — Max Power")

# ===================== Helpers / Utilities =====================
@st.cache_data(show_spinner=False, max_entries=2)
def safe_read_cached(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1', low_memory=False)

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def find_duplicate_columns(df):
    return [col for col in df.columns if list(df.columns).count(col) > 1]

def fix_types(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        if pd.api.types.is_float_dtype(df[col]):
            try:
                s = df[col].dropna()
                if len(s) and (s % 1 == 0).all():
                    df[col] = df[col].astype(pd.Int64Dtype())
            except Exception:
                pass
    return df

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
    df = fix_types(df)
    allowed_obj = {
        'wind_dir_string','condition','player_name','city','park','roof_status',
        'team_code','time','batter_hand','pitcher_team_code','pitcher_hand','stand'
    }
    drop_cols = [c for c in df.select_dtypes('O').columns if c not in allowed_obj]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.fillna(-1)
    if train_cols is not None:
        for c in train_cols:
            if c not in df.columns:
                df[c] = -1
        df = df[list(train_cols)]
    return df

def get_valid_feature_cols(df, drop=None):
    base_drop = set(['game_date','batter_id','mlb_id','player_name','pitcher_id','city','park',
                     'roof_status','team_code','time'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def nan_inf_check(X, name):
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include=[np.number])
        nans = X_num.isna().sum().sum()
        infs = np.isinf(X_num.to_numpy(dtype=np.float64, copy=False)).sum()
    else:
        nans = np.isnan(X).sum()
        infs = np.isinf(X).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

def feature_debug(X):
    X.columns = X.columns.astype(str)
    st.write("🛡️ Feature Debugging:")
    st.write("Data types:", X.dtypes.value_counts())
    object_cols = X.select_dtypes(include="object").columns.tolist()
    st.write("Columns with object dtype:", object_cols if object_cols else [])
    st.write("Missing values per column (top 10):", X.isna().sum().sort_values(ascending=False).head(10))

def drift_check(train, today, n=5):
    drifted = []
    for c in train.columns:
        if c not in today.columns: continue
        tmean = np.nanmean(train[c]); tstd = np.nanstd(train[c])
        dmean = np.nanmean(today[c])
        if tstd > 0 and abs(tmean - dmean) / tstd > n:
            drifted.append(c)
    return drifted

def winsorize_clip(X, limits=(0.01, 0.99)):
    X = X.astype(float)
    for col in X.columns:
        lower = X[col].quantile(limits[0])
        upper = X[col].quantile(limits[1])
        X[col] = X[col].clip(lower=lower, upper=upper)
    return X

def remove_outliers(
    X, y, method="iforest", contamination=0.012,
    n_estimators=150, max_samples='auto', n_neighbors=20, scale=True
):
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    if method == "iforest":
        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42
        )
        mask = clf.fit_predict(X_scaled) == 1
    elif method == "lof":
        clf = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
        mask = clf.fit_predict(X_scaled) == 1
    else:
        raise ValueError("Unknown method: choose 'iforest' or 'lof'")
    return X[mask], y[mask]

def smooth_labels(y, smoothing=0.02):
    y = np.asarray(y)
    y_smooth = y.copy().astype(float)
    y_smooth[y == 1] = 1 - smoothing
    y_smooth[y == 0] = smoothing
    return y_smooth

# ---------------- Target Encoding (time-safe OOF) ----------------
def oof_target_encode(values, y, dates, folds, global_prior=None, smoothing=50.0):
    y = np.asarray(y).astype(float)
    if global_prior is None:
        global_prior = y.mean() if len(y) else 0.0

    full_ct = values.value_counts(dropna=False)
    grp_sum = defaultdict(float); grp_cnt = defaultdict(float)
    for v, target in zip(values.fillna("__NA__"), y):
        grp_sum[v] += target; grp_cnt[v] += 1.0

    oof = np.zeros(len(values), dtype=np.float32)
    for tr_idx, va_idx in folds:
        tr_vals = values.iloc[tr_idx].fillna("__NA__")
        tr_y    = y[tr_idx]
        sum_d = defaultdict(float); cnt_d = defaultdict(float)
        for v, t in zip(tr_vals, tr_y):
            sum_d[v] += t; cnt_d[v] += 1.0
        te_map = {}
        for v in set(tr_vals):
            c = cnt_d.get(v, 0.0)
            m = sum_d.get(v, 0.0) / max(1.0, c)
            m_smooth = (c * m + smoothing * global_prior) / (c + smoothing)
            te_map[v] = m_smooth
        va_vals = values.iloc[va_idx].fillna("__NA__")
        oof[va_idx] = np.array([te_map.get(v, global_prior) for v in va_vals], dtype=np.float32)

    final_map = {}
    for v in full_ct.index:
        c = grp_cnt.get(v, 0.0)
        m = (grp_sum.get(v, 0.0) / max(1.0, c)) if c > 0 else global_prior
        final_map[v] = (c * m + smoothing * global_prior) / (c + smoothing)

    return oof, final_map, float(global_prior)

# ---------------- Embargoed time splits ----------------
def embargo_time_splits(dates_series, n_splits=2, embargo_days=1):
    dates = pd.to_datetime(dates_series).reset_index(drop=True)
    u_days = pd.Series(dates.dt.floor("D")).dropna().unique()
    u_days = pd.to_datetime(u_days)
    day_folds = np.array_split(np.arange(len(u_days)), n_splits)
    folds = []
    for k in range(n_splits):
        va_days_idx = day_folds[k]
        va_days = set(u_days[va_days_idx])
        if len(va_days):
            min_va = min(va_days)
            embargo_mask = (dates.dt.floor("D") >= (min_va - pd.Timedelta(days=embargo_days))) & (dates.dt.floor("D") < min_va)
        else:
            embargo_mask = pd.Series(False, index=dates.index)
        va_mask = dates.dt.floor("D").isin(va_days)
        tr_mask = ~va_mask & ~embargo_mask
        tr_idx = np.where(tr_mask.values)[0]
        va_idx = np.where(va_mask.values)[0]
        if len(tr_idx) and len(va_idx):
            folds.append((tr_idx, va_idx))
    return folds

# ---------------- Misc tools ----------------
def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

def tune_temperature_for_topk(p_oof, y, K=20, T_grid=np.linspace(0.8, 1.6, 17)):
    y = np.asarray(y).astype(int)
    best_T, best_hits = 1.0, -1
    logits = logit(np.clip(p_oof, 1e-6, 1-1e-6))
    for T in T_grid:
        p_adj = expit(logits * T)
        order = np.argsort(-p_adj)
        hits = int(y[order][:K].sum())
        if hits > best_hits:
            best_hits, best_T = hits, float(T)
    return best_T

# ===================== Overlays & Ratings =====================
def _getv(row, keys, default=np.nan):
    for k in keys:
        if k in row and pd.notnull(row[k]):
            return row[k]
    return default

def _hand(row, batter=True):
    return str(_getv(row, ["stand","batter_hand"] if batter else ["pitcher_hand","p_throws"], "R")).upper() or "R"

def first_present(row, base, windows):
    # expects full column names already (not a prefix)
    # kept for compatibility with original usage where full names are passed
    for w in windows:
        col = base  # user passes full names; we ignore w here
        if col in row and pd.notnull(row[col]):
            return row[col]
    return np.nan

def overlay_multiplier(row):
    EDGE_MIN, EDGE_MAX = 0.68, 1.44
    PULL_HI, PULL_LO   = 0.35, 0.28
    FB_HI_P, FB_HI_B   = 0.25, 0.22
    BARREL_HI, BARREL_MID = 0.12, 0.08
    HOT_HR_HI, HOT_HR_LO = 0.09, 0.025

    edge = 1.0
    b_hand = _hand(row, True)
    p_hand = _hand(row, False)

    temp     = _getv(row, ["temp"])
    humidity = _getv(row, ["humidity"])
    wind     = _getv(row, ["wind_mph"])
    wind_dir = str(_getv(row, ["wind_dir_string"], "")).lower().strip()
    roof     = str(_getv(row, ["roof_status"], "")).lower().strip()
    altitude = _getv(row, ["park_altitude"])

    pf_base  = _getv(row, ["park_hr_rate", "park_hand_hr_rate", "park_hr_pct_hand"])
    pf_rhb   = _getv(row, ["park_hr_pct_rhb"])
    pf_lhb   = _getv(row, ["park_hr_pct_lhb"])
    pf_hand  = pf_rhb if b_hand == "R" else pf_lhb if b_hand == "L" else np.nan

    def _cap_pf(x):
        try: return max(0.80, min(1.22, float(x)))
        except Exception: return np.nan

    pfs = [pf_hand, pf_base]
    pfs = [_cap_pf(x) for x in pfs if pd.notnull(x)]
    if pfs:
        edge *= pfs[0]

    # Pick from best-available short windows (user passes full names in their data)
    b_pull = _getv(row, ["b_pull_rate_7","b_pull_rate_14","b_pull_rate_5","b_pull_rate_3"])
    b_fb   = _getv(row, ["b_fb_rate_7","b_fb_rate_14","b_fb_rate_5","b_fb_rate_3"])
    b_brl  = _getv(row, ["b_barrel_rate_7","b_barrel_rate_14","b_barrel_rate_5","b_barrel_rate_3"])
    b_hot  = _getv(row, ["b_hr_per_pa_7","b_hr_per_pa_5","b_hr_per_pa_3"])

    if pd.notnull(b_brl):
        if b_brl >= BARREL_HI: edge *= 1.04
        elif b_brl >= BARREL_MID: edge *= 1.02

    if pd.notnull(b_hot):
        if b_hot > HOT_HR_HI: edge *= 1.04
        elif b_hot < HOT_HR_LO: edge *= 0.97

    p_fb = _getv(row, ["p_fb_rate_14","p_fb_rate_7","p_fb_rate"])

    roof_closed = ("closed" in roof) or ("indoor" in roof) or ("domed" in roof)

    if pd.notnull(p_fb) and float(p_fb) >= 0.40:
        if (pd.notnull(temp) and temp >= 75) and (pd.notnull(wind) and wind >= 7 and "out" in wind_dir) and not roof_closed:
            edge *= 1.02  # controlled nudge

    if pd.notnull(altitude):
        if altitude >= 5000: edge *= 1.05
        elif altitude >= 3000: edge *= 1.02

    if pd.notnull(temp):
        edge *= 1.035 ** ((temp - 70) / 10.0)
    if pd.notnull(humidity):
        if humidity >= 65: edge *= 1.02
        elif humidity <= 35: edge *= 0.98

    pulled_field = "lf" if b_hand == "R" else "rf"

    wind_factor = 1.0
    if pd.notnull(wind) and wind >= 6 and wind_dir:
        out = ("out" in wind_dir)
        inn = ("in"  in wind_dir)
        has_lf = "lf" in wind_dir
        has_rf = "rf" in wind_dir
        has_cf = ("cf" in wind_dir) or ("center" in wind_dir)

        hi_pull = pd.notnull(b_pull) and (b_pull >= PULL_HI)
        lo_pull = pd.notnull(b_pull) and (b_pull <= PULL_LO)
        hi_bfb  = pd.notnull(b_fb)   and (b_fb   >= FB_HI_B)
        hi_pfb  = pd.notnull(p_fb)   and (p_fb   >= FB_HI_P)

        OUT_CF_BOOST   = 1.11
        OUT_PULL_BOOST = 1.20
        OPPO_TINY      = 1.05
        IN_CF_FADE     = 0.92
        IN_PULL_FADE   = 0.85

        if has_cf and hi_bfb:
            wind_factor *= OUT_CF_BOOST if out else IN_CF_FADE if inn else 1.0

        if has_lf and pulled_field == "lf" and hi_pull:
            wind_factor *= OUT_PULL_BOOST if out else IN_PULL_FADE if inn else 1.0
        if has_rf and pulled_field == "rf" and hi_pull:
            wind_factor *= OUT_PULL_BOOST if out else IN_PULL_FADE if inn else 1.0

        if out and lo_pull:
            if has_lf and pulled_field == "rf": wind_factor *= OPPO_TINY
            if has_rf and pulled_field == "lf": wind_factor *= OPPO_TINY

        if hi_pfb and (out or inn):
            wind_factor *= 1.05 if out else 0.97

        if roof_closed:
            wind_factor = 1.0 + (wind_factor - 1.0) * 0.35

        if out or inn:
            extra = max(0.0, (wind - 8.0) / 3.0)
            wind_factor *= min(1.08, 1.0 + 0.01 * extra) if out else max(0.92, 1.0 - 0.01 * extra)

    edge *= wind_factor

    if pd.notnull(temp) and pd.notnull(wind):
        if (temp >= 75 and wind >= 7 and "out" in wind_dir) and not roof_closed:
            edge *= 1.05
        elif temp >= 65 and wind >= 5 and not roof_closed:
            edge *= 1.02
        else:
            edge *= 0.985

    if b_hand != p_hand: edge *= 1.01
    else: edge *= 0.995

    return float(np.clip(edge, EDGE_MIN, EDGE_MAX))

def rate_weather(row):
    ratings = {}
    temp = _getv(row, ["temp"])
    if pd.isna(temp):
        ratings["temp_rating"] = "?"
    elif 75 <= temp <= 85:
        ratings["temp_rating"] = "Excellent"
    elif 68 <= temp < 75 or 85 < temp <= 90:
        ratings["temp_rating"] = "Good"
    elif 60 <= temp < 68 or 90 < temp <= 95:
        ratings["temp_rating"] = "Fair"
    else:
        ratings["temp_rating"] = "Poor"

    hum = _getv(row, ["humidity"])
    if pd.isna(hum):
        ratings["humidity_rating"] = "?"
    elif hum >= 60:
        ratings["humidity_rating"] = "Excellent"
    elif 45 <= hum < 60:
        ratings["humidity_rating"] = "Good"
    elif 30 <= hum < 45:
        ratings["humidity_rating"] = "Fair"
    else:
        ratings["humidity_rating"] = "Poor"

    wind = _getv(row, ["wind_mph"])
    wdir = str(_getv(row, ["wind_dir_string"], "")).lower()
    if pd.isna(wind):
        ratings["wind_rating"] = "?"
    elif wind < 6:
        ratings["wind_rating"] = "Excellent"
    elif 6 <= wind < 12:
        ratings["wind_rating"] = "Good"
    elif 12 <= wind < 18:
        ratings["wind_rating"] = "Fair" if "in" in wdir else "Good"
    else:
        ratings["wind_rating"] = "Poor" if "in" in wdir else "Fair"

    cond = str(_getv(row, ["condition"], "")).lower()
    if not cond or cond in ("unknown","none","na"):
        ratings["condition_rating"] = "?"
    elif "clear" in cond or "sun" in cond or "outdoor" in cond:
        ratings["condition_rating"] = "Excellent"
    elif "cloud" in cond or "partly" in cond:
        ratings["condition_rating"] = "Good"
    elif "rain" in cond or "fog" in cond:
        ratings["condition_rating"] = "Poor"
    else:
        ratings["condition_rating"] = "Fair"
    return pd.Series(ratings)

# -------- Weak / Slumping Pitcher Overlay --------
def _first_non_null(row, *cands, default=np.nan):
    for c in cands:
        if c in row and pd.notnull(row[c]):
            return row[c]
    return default

def weak_pitcher_factor(row):
    def _get(*names, default=np.nan):
        for n in names:
            if n in row and pd.notnull(row[n]):
                return row[n]
        return default

    factor = 1.0

    # Recent HR vulnerability with sample-size shrink
    hr3   = _get("p_rolling_hr_3", "p_hr_count_3")
    pa3   = _get("p_rolling_pa_3")
    if pd.notnull(hr3) and pd.notnull(pa3) and pa3 > 0:
        hr_rate_short = float(hr3) / float(pa3)
        ss_shrink = min(1.0, pa3 / 30.0)
        if hr_rate_short >= 0.10:
            factor *= (1.12 * (0.5 + 0.5 * ss_shrink))
        elif hr_rate_short >= 0.07:
            factor *= (1.06 * (0.5 + 0.5 * ss_shrink))

    # Quality of contact allowed (use 14g/30g if present)
    brl14 = _get("p_fs_barrel_rate_14", "p_barrel_rate_14", "p_hard_hit_rate_14")
    brl30 = _get("p_fs_barrel_rate_30", "p_barrel_rate_30", "p_hard_hit_rate_30")
    qoc = np.nanmax([brl14, brl30]) if any(pd.notnull(x) for x in [brl14, brl30]) else np.nan
    if pd.notnull(qoc):
        v = float(qoc)
        if v >= 0.11: factor *= 1.07
        elif v >= 0.09: factor *= 1.04

    # Fly/ground profile
    fb14 = _get("p_fb_rate_14", "p_fb_rate_7", "p_fb_rate", "p_fb_pct")
    gb14 = _get("p_gb_rate_14", "p_gb_rate_7", "p_gb_rate", "p_gb_pct")
    if pd.notnull(fb14):
        if float(fb14) >= 0.42: factor *= 1.04
        elif float(fb14) >= 0.38: factor *= 1.02
    if pd.notnull(gb14) and float(gb14) <= 0.40:
        factor *= 1.02

    # Walks / damage proxies
    bb_rate = _get("p_bb_rate_14", "p_bb_rate_30", "p_bb_rate")
    xwoba_con = _get("p_xwoba_con_14", "p_xwoba_con_30", "p_xwoba_con")
    if pd.notnull(bb_rate) and float(bb_rate) >= 0.09:
        factor *= 1.02
    if pd.notnull(xwoba_con):
        if float(xwoba_con) >= 0.40: factor *= 1.05
        elif float(xwoba_con) >= 0.36: factor *= 1.03

    # EV allowed
    ev_allowed = _get("p_avg_exit_velo_14", "p_avg_exit_velo_7", "p_avg_exit_velo_30",
                      "p_exit_velocity_avg", "p_avg_exit_velo")
    if pd.notnull(ev_allowed) and float(ev_allowed) >= 90.0:
        factor *= 1.03

    # Platoon
    b_hand = str(_get("stand", "batter_hand", default="R")).upper()
    p_hand = str(_get("pitcher_hand", "p_throws", default="R")).upper()
    if b_hand == "L":
        p_platoon_hr = _get("p_hr_pa_vl_30", "p_hr_pa_vl_14", "p_hr_pa_vl")
    else:
        p_platoon_hr = _get("p_hr_pa_vr_30", "p_hr_pa_vr_14", "p_hr_pa_vr")
    if pd.notnull(p_platoon_hr):
        v = float(p_platoon_hr)
        if v >= 0.06: factor *= 1.05
        elif v >= 0.04: factor *= 1.03

    if (b_hand == "L" and p_hand == "R") or (b_hand == "R" and p_hand == "L"):
        factor *= 1.015

    return float(np.clip(factor, 0.90, 1.18))

def short_term_hot_factor(row):
    ev = _first_non_null(row, "b_avg_exit_velo_5", "b_avg_exit_velo_3", default=np.nan)
    la = _first_non_null(row, "b_la_mean_5", "b_la_mean_3", default=np.nan)
    br = _first_non_null(row, "b_barrel_rate_5", "b_barrel_rate_3", default=np.nan)
    factor = 1.0
    try:
        if pd.notnull(ev) and float(ev) >= 91: factor *= 1.03
        if pd.notnull(la) and 12 <= float(la) <= 24: factor *= 1.02
        if pd.notnull(br):
            if float(br) >= 0.12: factor *= 1.05
            elif float(br) >= 0.08: factor *= 1.02
    except Exception:
        pass
    return float(np.clip(factor, 0.96, 1.10))

# ===================== APP START =====================

# Allow uploading optional learning ranker up front
lr_file = st.file_uploader("Optional: upload learning_ranker.pkl (can upload now)", type=["pkl"], key="lrpk")

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read_cached(event_file)
        today_df = safe_read_cached(today_file)

        # Basic cleaning
        event_df = event_df.dropna(axis=1, how='all')
        today_df = today_df.dropna(axis=1, how='all')
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = event_df.reset_index(drop=True)
        today_df = today_df.reset_index(drop=True)

        if find_duplicate_columns(event_df):
            st.error("Duplicate columns in event file")
            st.stop()
        if find_duplicate_columns(today_df):
            st.error("Duplicate columns in today file")
            st.stop()

        # Type fixes
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

        st.write(f"event_df shape: {event_df.shape}, today_df shape: {today_df.shape}")
        st.write(f"event_df memory usage (MB): {event_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        st.write(f"today_df memory usage (MB): {today_df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")

    # ---- Leak-prone drop ----
    LEAK = {
        "post_away_score","post_home_score","post_bat_score","post_fld_score",
        "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
        "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
        "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
        "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
        "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
    }
    event_df = event_df.drop(columns=[c for c in event_df.columns if c in LEAK], errors="ignore")

    # ---- Feature list intersection ----
    feature_cols = sorted(list(set(get_valid_feature_cols(event_df)) & set(get_valid_feature_cols(today_df))))
    st.write(f"Feature count before filtering: {len(feature_cols)}")

    X = clean_X(event_df[feature_cols])
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    feature_debug(X)

    # ---- Prune: NaN heavy, near-constant, dup-correlated ----
    nan_thresh = 0.30
    nan_pct = X.isna().mean()
    drop_cols = nan_pct[nan_pct > nan_thresh].index.tolist()
    if drop_cols:
        st.warning(f"Dropping {len(drop_cols)} features with >30% NaNs (showing up to 20): {drop_cols[:20]}")
        X = X.drop(columns=drop_cols)
        X_today = X_today.drop(columns=drop_cols, errors='ignore')

    nzv_cols = X.loc[:, X.nunique() <= 2].columns.tolist()
    if nzv_cols:
        st.warning(f"Dropping {len(nzv_cols)} near-constant features.")
        X = X.drop(columns=nzv_cols)
        X_today = X_today.drop(columns=nzv_cols, errors='ignore')

    corrs = X.corr().abs()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    if to_drop:
        st.warning(f"Dropping {len(to_drop)} highly correlated features.")
        X = X.drop(columns=to_drop)
        X_today = X_today.drop(columns=to_drop, errors='ignore')

    X = winsorize_clip(X)
    X_today = winsorize_clip(X_today)

    # --- Optional crosses (off by default; retained but no plots) ---
    use_crosses = st.checkbox("Enable polynomial interaction crosses (slow, memory heavy)", value=False)
    if use_crosses:
        try:
            gc.collect()
            mem_before = psutil.virtual_memory().available / (1024 ** 3)
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            Xt = X.astype(np.float32)
            X_cross = pd.DataFrame(poly.fit_transform(Xt), columns=poly.get_feature_names_out(X.columns))
            X_cross = dedup_columns(X_cross)
            var_scores = X_cross.var().sort_values(ascending=False)
            keep_cross = list(var_scores.head(min(300, max(50, X_cross.shape[1]//10))).index)
            X_cross = X_cross[keep_cross]
            X = pd.concat([X, X_cross], axis=1)

            Xt_today = X_today.astype(np.float32)
            X_today_cross = pd.DataFrame(poly.transform(Xt_today), columns=poly.get_feature_names_out(X_today.columns))
            X_today_cross = X_today_cross[keep_cross]
            X_today = pd.concat([X_today, X_today_cross], axis=1)
            gc.collect()
            mem_after = psutil.virtual_memory().available / (1024 ** 3)
            st.write(f"🧠 Memory: {mem_before:.1f}GB → {mem_after:.1f}GB after crosses")
        except Exception as e:
            st.error(f"❌ Cross-feature generation failed: {str(e)}")

    # --- Cheap synergy feats ---
    def add_synergy_feats(df_ref, df_today_ref):
        for df in (df_ref, df_today_ref):
            if "b_fb_rate_7" in df.columns and "park_hr_rate" in df.columns:
                df["feat_fb_park7"] = df["b_fb_rate_7"].astype(np.float32) * df["park_hr_rate"].astype(np.float32)
            elif "b_fb_rate_14" in df.columns and "park_hr_rate" in df.columns:
                df["feat_fb_park7"] = df["b_fb_rate_14"].astype(np.float32) * df["park_hr_rate"].astype(np.float32)
            else:
                df["feat_fb_park7"] = 0.0
            temp_col = "temp" if "temp" in df.columns else None
            pfb_col = "p_fb_rate_14" if "p_fb_rate_14" in df.columns else ("p_fb_rate_7" if "p_fb_rate_7" in df.columns else None)
            if temp_col and pfb_col:
                df["feat_pfb_temp"] = df[pfb_col].astype(np.float32) * np.maximum(0, df[temp_col].astype(np.float32) - 70.0)
            else:
                df["feat_pfb_temp"] = 0.0
        return df_ref, df_today_ref

    X, X_today = add_synergy_feats(X, X_today)

    def add_micro_feats(df):
        if "b_barrel_rate_7" in df.columns and "park_hr_rate" in df.columns:
            df["feat_brl_park7"] = (df["b_barrel_rate_7"].astype(np.float32) * df["park_hr_rate"].astype(np.float32))
        if "b_pull_rate_7" in df.columns:
            df["feat_pull_wind"] = 0.0  # train stays 0 to avoid leakage (filled for today below)
        if "p_fb_rate_14" in df.columns and "temp" in df.columns:
            df["feat_pfb_temp14"] = (df["p_fb_rate_14"].astype(np.float32) * np.maximum(0, df["temp"].astype(np.float32) - 70.0))
        return df

    X = add_micro_feats(X)
    X_today = add_micro_feats(X_today)

    # ---- Chronological ordering BEFORE outlier removal ----
    y = event_df[target_col].astype(int)

    order_idx = None
    if "game_date" in event_df.columns:
        dates = pd.to_datetime(event_df["game_date"], errors="coerce")
        min_date = dates.min()
        dates_filled = dates if pd.isna(min_date) else dates.fillna(min_date)
        order_idx = dates_filled.sort_values(kind="mergesort").index
        X = X.loc[order_idx].reset_index(drop=True)
        y = y.loc[order_idx].reset_index(drop=True)

    # Keep aligned dates for folds and ranker grouping
    if "game_date" in event_df.columns:
        dates_aligned = pd.to_datetime(event_df["game_date"], errors="coerce")
        dates_aligned = dates_aligned.fillna(dates_aligned.min())
        dates_aligned = dates_aligned.loc[X.index].reset_index(drop=True)
    else:
        dates_aligned = pd.Series(pd.Timestamp("2000-01-01"), index=X.index)

    # --- Outlier removal ---
    st.write("🚦 Starting outlier removal...")
    X_filtered, y_filtered = remove_outliers(X, y, method="iforest", contamination=0.012)
    dates_aligned = dates_aligned.loc[X_filtered.index].reset_index(drop=True)
    X = X_filtered.reset_index(drop=True).copy()
    y = pd.Series(y_filtered).reset_index(drop=True)
    st.write(f"✅ Outlier removal complete. Rows remaining: {X.shape[0]}")

    # --- Fill missing, cast ---
    st.write("🩹 Filling missing values...")
    X = X.fillna(-1)
    X_today = X_today.fillna(-1)
    try:
        X = X.astype(np.float32)
        X_today = X_today.astype(np.float32)
        st.success("✅ Converted feature matrices to float32")
    except Exception as e:
        st.error(f"❌ Conversion to float32 failed: {e}")

    # ---------- Target Encoding setup ----------
    cat_cols_available = [c for c in ["park","team_code","batter_hand","pitcher_team_code"] if c in event_df.columns]
    cats_full = event_df[cat_cols_available].copy()
    if order_idx is not None:
        cats_full = cats_full.loc[order_idx].reset_index(drop=True)
    cats_full = cats_full.loc[X.index].reset_index(drop=True)

    def _combo(a, b):
        a = a.fillna("__NA__").astype(str)
        b = b.fillna("__NA__").astype(str)
        return (a + "×" + b).replace("nan", "__NA__")

    te_specs = []
    if "park" in cats_full.columns: te_specs.append(("te_park", cats_full["park"]))
    if "team_code" in cats_full.columns: te_specs.append(("te_team", cats_full["team_code"]))
    if set(["park","batter_hand"]).issubset(cats_full.columns):
        te_specs.append(("te_park_hand", _combo(cats_full["park"], cats_full["batter_hand"])))
    if set(["pitcher_team_code","batter_hand"]).issubset(cats_full.columns):
        te_specs.append(("te_pteam_hand", _combo(cats_full["pitcher_team_code"], cats_full["batter_hand"])))

    n_splits = st.slider("Number of time-embargoed CV splits", 4, 7, 5, 1)
    folds = embargo_time_splits(dates_aligned, n_splits=n_splits, embargo_days=1)

    te_maps = {}
    global_means = {}
    for name, ser in te_specs:
        oof_vals, fmap, gmean = oof_target_encode(ser, y.values, dates_aligned, folds, smoothing=50.0)
        X[name] = oof_vals.astype(np.float32)
        te_maps[name] = fmap
        global_means[name] = gmean

    # Map TE to today
    def _map_series_to_te(series, fmap, gmean):
        s = series.fillna("__NA__").astype(str)
        return s.map(lambda v: fmap.get(v, gmean)).astype(np.float32)

    if "park" in today_df.columns:
        X_today["te_park"] = _map_series_to_te(today_df["park"], te_maps.get("te_park", {}), global_means.get("te_park", y.mean()))
    if "team_code" in today_df.columns:
        X_today["te_team"] = _map_series_to_te(today_df["team_code"], te_maps.get("te_team", {}), global_means.get("te_team", y.mean()))
    if set(["park","batter_hand"]).issubset(today_df.columns):
        X_today["te_park_hand"] = _map_series_to_te(_combo(today_df["park"], today_df["batter_hand"]),
                                                    te_maps.get("te_park_hand", {}), global_means.get("te_park_hand", y.mean()))
    if set(["pitcher_team_code","batter_hand"]).issubset(today_df.columns):
        X_today["te_pteam_hand"] = _map_series_to_te(_combo(today_df["pitcher_team_code"], today_df["batter_hand"]),
                                                     te_maps.get("te_pteam_hand", {}), global_means.get("te_pteam_hand", y.mean()))

    # Fill learnable wind-to-pull flag for today (train was 0.0 to avoid leakage)
    if "wind_dir_string" in today_df.columns and ("stand" in today_df.columns or "batter_hand" in today_df.columns):
        wdir = today_df["wind_dir_string"].astype(str).str.lower().fillna("")
        hand_src = "stand" if "stand" in today_df.columns else "batter_hand"
        hand = today_df[hand_src].astype(str).str.upper().fillna("R")
        pulled_field = np.where(hand == "R", "lf", "rf")
        out_to_pull = wdir.str.contains("out") & (
            ((pulled_field == "lf") & wdir.str.contains("lf")) |
            ((pulled_field == "rf") & wdir.str.contains("rf"))
        )
        X_today["feat_pull_wind"] = out_to_pull.astype(np.float32)

    # === FINAL COLUMN PINNING ===
    X = dedup_columns(X); X_today = dedup_columns(X_today)
    X.columns = X.columns.astype(str); X_today.columns = X_today.columns.astype(str)
    extra_today = sorted(set(X_today.columns) - set(X.columns))
    if extra_today:
        st.warning(f"Dropping {len(extra_today)} extra today cols (e.g. {extra_today[:8]})")
        X_today = X_today.drop(columns=extra_today, errors='ignore')
    missing_today = sorted(set(X.columns) - set(X_today.columns))
    for c in missing_today: X_today[c] = -1
    X_today = X_today[X.columns]
    X_today = X_today.fillna(-1)
    nan_inf_check(X_today, "X_today (pinned)")
    st.write(f"✅ Columns pinned. Train cols: {X.shape[1]} | Today cols: {X_today.shape[1]}")
    st.dataframe(X_today.head(300))

    # ========== Train/Validation via Embargoed Time Splits ==========
    seeds = [42, 101, 202, 404]  # extra bags to steady p_base and reduce disagreement noise
    P_xgb_oof = np.zeros(len(y), dtype=np.float32)
    P_lgb_oof = np.zeros(len(y), dtype=np.float32)
    P_cat_oof = np.zeros(len(y), dtype=np.float32)
    P_xgb_today, P_lgb_today, P_cat_today = [], [], []
    fold_times = []

    for fold, (tr_idx, va_idx) in enumerate(folds):
        t_fold_start = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        spw_fold = max(1.0, (len(y_tr)-y_tr.sum())/max(1.0, y_tr.sum()))

        preds_xgb_va, preds_lgb_va, preds_cat_va = [], [], []
        preds_xgb_td, preds_lgb_td, preds_cat_td = [], [], []

        for sd in seeds:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=650, max_depth=6, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
                eval_metric="logloss", tree_method="hist",
                scale_pos_weight=spw_fold, early_stopping_rounds=50,
                n_jobs=1, verbosity=0, random_state=sd
            )
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=63,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                reg_lambda=2.0, n_jobs=1, is_unbalance=True, random_state=sd
            )
            cat_clf = cb.CatBoostClassifier(
                iterations=1500, depth=7, learning_rate=0.03, l2_leaf_reg=6.0,
                loss_function="Logloss", eval_metric="Logloss",
                class_weights=[1.0, spw_fold], od_type="Iter", od_wait=50,
                verbose=0, thread_count=1, random_seed=sd
            )

            xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            cat_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            preds_xgb_va.append(xgb_clf.predict_proba(X_va)[:,1])
            preds_lgb_va.append(lgb_clf.predict_proba(X_va)[:,1])
            preds_cat_va.append(cat_clf.predict_proba(X_va)[:,1])

            preds_xgb_td.append(xgb_clf.predict_proba(X_today)[:,1])
            preds_lgb_td.append(lgb_clf.predict_proba(X_today)[:,1])
            preds_cat_td.append(cat_clf.predict_proba(X_today)[:,1])

        P_xgb_oof[va_idx] = np.mean(preds_xgb_va, axis=0)
        P_lgb_oof[va_idx] = np.mean(preds_lgb_va, axis=0)
        P_cat_oof[va_idx] = np.mean(preds_cat_va, axis=0)

        P_xgb_today.append(np.mean(preds_xgb_td, axis=0))
        P_lgb_today.append(np.mean(preds_lgb_td, axis=0))
        P_cat_today.append(np.mean(preds_cat_td, axis=0))

        fold_time = time.time() - t_fold_start
        fold_times.append(fold_time)
        avg_time = np.mean(fold_times)
        est_time_left = avg_time * (len(folds) - (fold + 1))
        st.write(f"Fold {fold + 1}/{len(folds)} finished in {timedelta(seconds=int(fold_time))}. Est. {timedelta(seconds=int(est_time_left))} left.")

    # ---------- Day-wise LGBMRanker head ----------
    days = pd.to_datetime(dates_aligned).dt.floor("D")
    ranker_oof = np.zeros(len(y), dtype=np.float32)
    ranker_today_parts = []

    def _groups_from_days(d):
        return d.groupby(d.values).size().values.tolist()

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr = days.iloc[tr_idx]; d_va = days.iloc[va_idx]
        g_tr = _groups_from_days(d_tr); g_va = _groups_from_days(d_va)

        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=fold
        )
        rk.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        ranker_oof[va_idx] = rk.predict(X_va)
        ranker_today_parts.append(rk.predict(X_today))

    ranker_today = np.mean(ranker_today_parts, axis=0)

    # ---------- Meta stacker ----------
    X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)
    scaler_meta = StandardScaler()
    X_meta_s = scaler_meta.fit_transform(X_meta)
    meta = LogisticRegression(max_iter=1000, solver="lbfgs")
    meta.fit(X_meta_s, y.values)

    P_today_base = np.column_stack([
        np.mean(P_xgb_today, axis=0),
        np.mean(P_lgb_today, axis=0),
        np.mean(P_cat_today, axis=0)
    ]).astype(np.float32)
    P_today_meta = meta.predict_proba(scaler_meta.transform(P_today_base))[:,1]

    # ---------- Calibration ----------
    st.markdown("### 📊 Calibration (Isotonic on OOF meta)")
    oof_pred_meta = meta.predict_proba(X_meta_s)[:,1]
    auc_oof = roc_auc_score(y, oof_pred_meta)
    ll_oof  = log_loss(y, oof_pred_meta)
    st.success(f"OOF Meta AUC: {auc_oof:.4f} | OOF Meta LogLoss: {ll_oof:.4f}")

    ir = IsotonicRegression(out_of_bounds="clip")
    y_oof_iso = ir.fit_transform(oof_pred_meta, y.values)
    today_iso = ir.transform(P_today_meta)

    best_T = tune_temperature_for_topk(y_oof_iso, y.values, K=20, T_grid=np.linspace(0.8, 1.6, 17))
    logits_today = logit(np.clip(today_iso, 1e-6, 1-1e-6))
    today_iso_t = expit(logits_today * best_T)

    # ---------- Apply overlays ----------
    today_df = today_df.copy()
    ratings_df = today_df.apply(rate_weather, axis=1)
    for col in ratings_df.columns:
        today_df[col] = ratings_df[col]

    today_df["overlay_multiplier"] = today_df.apply(overlay_multiplier, axis=1)
    today_df["weak_pitcher_factor"] = today_df.apply(weak_pitcher_factor, axis=1).astype(np.float32)
    today_df["hot_streak_factor"]   = today_df.apply(short_term_hot_factor, axis=1).astype(np.float32)

    # Raw multiplier
    today_df["final_multiplier_raw"] = (
        today_df["overlay_multiplier"].astype(float).clip(0.68, 1.44)
        * today_df["weak_pitcher_factor"].astype(float)
        * today_df["hot_streak_factor"].astype(float)
    ).clip(0.60, 1.65).astype(np.float32)

    # ======= Uncertainty-aware overlay shrinkage (always on) =======
    roof = today_df.get("roof_status", pd.Series("", index=today_df.index)).astype(str).str.lower()
    roof_closed = roof.str.contains("closed|indoor|domed", regex=True)
    has_temp = today_df["temp"].notna() if "temp" in today_df.columns else pd.Series(False, index=today_df.index)
    has_hum  = today_df["humidity"].notna() if "humidity" in today_df.columns else pd.Series(False, index=today_df.index)
    has_wind = today_df["wind_mph"].notna() if "wind_mph" in today_df.columns else pd.Series(False, index=today_df.index)

    conf_base = (has_temp.astype(float) + has_hum.astype(float) + has_wind.astype(float)) / 3.0
    conf_roof = np.where(roof_closed, 0.35, 1.0)
    confidence = np.clip(conf_base * conf_roof, 0.0, 1.0)

    alpha = 0.5 + 0.5 * confidence
    today_df["final_multiplier"] = (1.0 + alpha * (today_df["final_multiplier_raw"].values - 1.0)).astype(np.float32)

    # ========= USE LEARNING RANKER IF PROVIDED =========
    learned_rank_score = None
    if lr_file is not None:
        try:
            bundle = pickle.load(lr_file)
            # Prefer single model under key "model"; else fallback to bundle["models"]["lgb"] or any available
            lbr = bundle.get("model")
            if lbr is None and "models" in bundle:
                models = bundle["models"]
                lbr = models.get("lgb") or next((m for m in models.values() if m is not None), None)
            rk_feats = bundle.get("features", ["base_prob","logit_p","log_overlay","ranker_z","overlay_multiplier","final_multiplier"])

            if lbr is not None:
                feat_map = {
                    "base_prob":   np.asarray(today_iso_t, dtype=float),
                    "logit_p":     logit(np.clip(today_iso_t, 1e-6, 1 - 1e-6)),
                    "log_overlay": np.log(today_df["final_multiplier"].values + 1e-9),
                    "ranker_z":    zscore(ranker_today),
                    "overlay_multiplier": today_df.get("overlay_multiplier", pd.Series(1.0, index=today_df.index)).to_numpy(dtype=float),
                    "final_multiplier":   today_df.get("final_multiplier",   pd.Series(1.0, index=today_df.index)).to_numpy(dtype=float),
                }
                Xrk_today = np.column_stack([feat_map[f] for f in rk_feats if f in feat_map]).astype(np.float32)
                learned_rank_score = lbr.predict(Xrk_today)
                st.info("Applied learning ranker. Replacing original ranker_z for blending.")
                ranker_z = zscore(learned_rank_score)
            else:
                st.warning("learning_ranker.pkl did not contain a usable model; falling back to LGB ranker head.")
                ranker_z = zscore(ranker_today)
        except Exception as e:
            st.warning(f"Could not load/apply learning ranker: {e}")
            ranker_z = zscore(ranker_today)
    else:
        ranker_z = zscore(ranker_today)

    # ---- Base prob terms (use temp-tuned isotonic if available)
    try:
        p_base = today_iso_t
    except NameError:
        try:
            p_base = today_iso
        except NameError:
            p_base = P_today_meta

    logit_p = logit(np.clip(p_base, 1e-6, 1 - 1e-6))
    log_overlay = np.log(today_df["final_multiplier"].values + 1e-9)

    # ======= Reciprocal Rank Fusion auxiliary rank (always on) =======
    def _rank_desc(x):
        x = np.asarray(x)
        return pd.Series(-x).rank(method="min").astype(int).values

    r_prob    = _rank_desc(p_base)
    r_ranker  = _rank_desc(ranker_z)
    r_overlay = _rank_desc(today_df["final_multiplier"].values)
    k_rrf = 60.0
    rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
    rrf_z = zscore(rrf)

    # ======= Model-disagreement penalty (always on) =======
    p_xgb = np.mean(P_xgb_today, axis=0)
    p_lgb = np.mean(P_lgb_today, axis=0)
    p_cat = np.mean(P_cat_today, axis=0)
    disagree_std = np.std(np.vstack([p_xgb, p_lgb, p_cat]), axis=0)
    dis_z = zscore(disagree_std)
    dis_penalty = np.clip(dis_z, 0, 3)  # only penalize above-average disagreement

    # Bayesian shrink of p_base based on base-model disagreement
    model_std = disagree_std.astype(np.float32)
    s0, s1 = 0.02, 0.08
    alpha_shrink = np.clip((model_std - s0) / max(1e-9, (s1 - s0)), 0.0, 1.0) * 0.40
    base_rate = float(np.clip(y.mean() if len(y) else 0.03, 1e-6, 1-1e-6))
    p_base = (1.0 - alpha_shrink) * np.clip(p_base, 1e-6, 1-1e-6) + alpha_shrink * base_rate
    logit_p = logit(np.clip(p_base, 1e-6, 1-1e-6))

    # --- Final blended score using tuned weights (NO tuner UI) ---
    W = DEFAULT_WEIGHTS.copy()
    ranked_score = expit(
        W["w_prob"]    * logit_p
      + W["w_overlay"] * log_overlay
      + W["w_ranker"]  * zscore(ranker_z)
      + W["w_rrf"]     * rrf_z
      - W["w_penalty"] * dis_penalty
    )

    # ================= Auxiliary props: 2+ Total Bases & RBI (proxies) =================
    # Robust getters for best-available stat window
    def pick_best_col(df, base, windows=(14, 30, 7, 20, 60, 5, 3)):
        for w in windows:
            c = f"{base}_{w}"
            if c in df.columns:
                return df[c].astype(float)
        return pd.Series(np.nan, index=df.index, dtype="float32")

    b_slg = pick_best_col(today_df, "b_slg")
    b_hh  = pick_best_col(today_df, "b_hard_hit_rate")
    b_hc  = pick_best_col(today_df, "b_hard_contact_rate")  # sometimes present
    b_fb  = pick_best_col(today_df, "b_fb_rate")
    b_brl = pick_best_col(today_df, "b_barrel_rate")

    # z-safe helper
    def zsafe(s):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        mu = np.nanmean(s.values); sd = np.nanstd(s.values) + 1e-9
        return pd.Series((s.values - mu)/sd, index=s.index)

    z_slg = zsafe(b_slg.fillna(b_slg.median()))
    z_hh  = zsafe(b_hh.fillna(b_hh.median()))
    z_hc  = zsafe(b_hc.fillna(b_hc.median()))
    z_fb  = zsafe(b_fb.fillna(b_fb.median()))
    z_brl = zsafe(b_brl.fillna(b_brl.median()))

    # Proxy: probability of 2+ total bases
    logits_2tb = (
        1.40 * logit_p
      + 0.70 * z_slg.values
      + 0.45 * z_hh.values
      + 0.35 * z_brl.values
      + 0.20 * np.log(today_df["final_multiplier"].values + 1e-9)
    )
    prob_2tb = expit(logits_2tb)

    # Proxy: RBI probability
    logits_rbi = (
        1.20 * logit_p
      + 0.50 * z_hc.values
      + 0.35 * z_hh.values
      + 0.20 * z_fb.values
      + 0.15 * np.log(today_df["final_multiplier"].values + 1e-9)
    )
    prob_rbi = expit(logits_rbi)

    # ================= Leaderboard Build & Outputs =================
    def build_leaderboard(df, calibrated_probs, final_score, prob_2tb, prob_rbi, label="hr_probability_iso_T"):
        df = df.copy()
        df[label] = np.asarray(calibrated_probs)
        df["ranked_probability"] = np.asarray(final_score)
        df["prob_2tb"] = np.asarray(prob_2tb)
        df["prob_rbi"] = np.asarray(prob_rbi)

        df = df.sort_values("ranked_probability", ascending=False).reset_index(drop=True)
        df["hr_base_rank"] = df[label].rank(method="min", ascending=False)

        # Include MLB ID (batter_id or mlb_id if present)
        mlb_id_col = None
        for c in ["batter_id", "mlb_id"]:
            if c in df.columns:
                mlb_id_col = c
                break

        cols = []
        if mlb_id_col:
            cols.append(mlb_id_col)
        for c in ["player_name", "team_code", "time"]:
            if c in df.columns: cols.append(c)

        cols += [
            label, "ranked_probability",
            "prob_2tb", "prob_rbi",
            "overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
            "final_multiplier_raw", "final_multiplier",
            "temp", "temp_rating", "humidity", "humidity_rating",
            "wind_mph", "wind_rating", "wind_dir_string",
            "condition", "condition_rating",
            # diagnostics
            "rrf_aux","model_disagreement",
            "hr_outcome",
        ]
        cols = [c for c in cols if c in df.columns]
        out = df[cols].copy()

        # rounding for readability
        for c in [label, "ranked_probability", "prob_2tb", "prob_rbi"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float).round(4)
        for c in ["overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
                  "final_multiplier_raw", "final_multiplier", "rrf_aux", "model_disagreement"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float).round(3)
        return out

    # attach diagnostics to today_df for display
    today_df["rrf_aux"] = rrf
    today_df["model_disagreement"] = disagree_std

    leaderboard = build_leaderboard(today_df, p_base, ranked_score, prob_2tb, prob_rbi, label="hr_probability_iso_T")

    # ===== Render current-day leaderboard (no charts) =====
    top_n = st.sidebar.number_input("Top-N to display", min_value=10, max_value=100, value=30, step=5)
    st.markdown(f"### 🏆 **Top {int(top_n)} HR Leaderboard (Blended + Overlays + Ranker)**")
    leaderboard_top = leaderboard.head(int(top_n))
    st.dataframe(leaderboard_top, use_container_width=True)

    st.download_button(
        label=f"⬇️ Download Top {int(top_n)} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{int(top_n)}_leaderboard_blended.csv",
        mime="text/csv",
    )
    st.download_button(
        label="⬇️ Download Full Prediction CSV (Blended)",
        data=leaderboard.to_csv(index=False),
        file_name="today_hr_predictions_full_blended.csv",
        mime="text/csv",
    )

    # Drift diagnostics (safe, no plots)
    try:
        drifted = drift_check(X, X_today, n=6)
        if drifted:
            st.markdown("#### ⚡ **Feature Drift Diagnostics**")
            st.write("These features show unusual mean/std changes:", drifted)
    except Exception:
        pass

    gc.collect()
    st.success("✅ HR Prediction pipeline complete. Leaderboard generated and ready.")
    st.caption("Meta-ensemble + calibrated probs + contextual overlays + RRF + disagreement control + (optional) learning ranker. 2+TB and RBI proxies included.")
