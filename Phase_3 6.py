# app.py
# ============================================================
# 🏆 MLB Home Run Predictor — "Max Power" Edition (Upgraded)
#
# Always-on upgrades:
# - Adaptive top-K temperature tuning
# - Learner fail-closed parity check (no half-applied ranker)
# - Tie-breaking with 2TB/RBI
# - Micro retune of w_prob / w_ranker on OOF (tiny grid)
# - Park/hand Bayesian prior blend into p_base
# - Handedness-segmented models + small TT-Aug (B=3)
# - Polynomial crosses removed (no toggles)
# - Vectorized overlays + de-fragmentation & concat-based TE to avoid slow inserts
# - TT-Aug std-vector aligned to X_today columns to prevent shape mismatches
# - Safer rounding in leaderboard (Series-only)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gc, time, psutil, pickle
from datetime import timedelta
from collections import defaultdict

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from scipy.special import logit, expit

# ===== Tuned blend weights (baseline) =====
DEFAULT_WEIGHTS = dict(
    w_prob=0.3879,
    w_overlay=0.0161,
    w_ranker=0.4461,
    w_rrf=0.1416,
    w_penalty=0.0083
)

# ===================== UI =====================
st.set_page_config(page_title="🏆 MLB Home Run Predictor — Max Power", layout="wide")
st.title("🏆 MLB Home Run Predictor — Max Power")

# ===================== Helpers =====================
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
            try: df[col] = pd.to_numeric(df[col], errors='ignore')
            except: pass
        if pd.api.types.is_float_dtype(df[col]):
            try:
                s = df[col].dropna()
                if len(s) and (s % 1 == 0).all():
                    df[col] = df[col].astype(pd.Int64Dtype())
            except: pass
    return df

def clean_X(df, train_cols=None):
    df = dedup_columns(df); df = fix_types(df)
    allowed_obj = {'wind_dir_string','condition','player_name','city','park','roof_status',
                   'team_code','time','batter_hand','pitcher_team_code','pitcher_hand','stand'}
    drop_cols = [c for c in df.select_dtypes('O').columns if c not in allowed_obj]
    df = df.drop(columns=drop_cols, errors='ignore').fillna(-1)
    if train_cols is not None:
        for c in train_cols:
            if c not in df.columns: df[c] = -1
        df = df[list(train_cols)]
    return df

def get_valid_feature_cols(df, drop=None):
    base_drop = {'game_date','batter_id','mlb_id','player_name','pitcher_id','city',
                 'park','roof_status','team_code','time'}
    if drop: base_drop |= set(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def nan_inf_check(X, name):
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include=[np.number])
        nans = X_num.isna().sum().sum()
        infs = np.isinf(X_num.to_numpy(dtype=np.float64, copy=False)).sum()
    else:
        nans = np.isnan(X).sum(); infs = np.isinf(X).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix."); st.stop()

def winsorize_clip(X, limits=(0.01, 0.99)):
    X = X.astype(float)
    for col in X.columns:
        lower = X[col].quantile(limits[0]); upper = X[col].quantile(limits[1])
        X[col] = X[col].clip(lower=lower, upper=upper)
    return X

def remove_outliers(X, y, method="iforest", contamination=0.012,
                    n_estimators=150, max_samples='auto', n_neighbors=20, scale=True):
    if scale:
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    else: X_scaled = X
    if method=="iforest":
        clf = IsolationForest(contamination=contamination,n_estimators=n_estimators,
                              max_samples=max_samples,random_state=42)
        mask = clf.fit_predict(X_scaled)==1
    elif method=="lof":
        clf = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
        mask = clf.fit_predict(X_scaled)==1
    else:
        raise ValueError("Unknown method")
    return X[mask], y[mask]

def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

def debug_mem(tag, df):
    try:
        mb = df.memory_usage(deep=True).sum() / 1024**2
        st.write(f"🧠 {tag}: {mb:.2f} MB, shape={getattr(df,'shape',None)}")
    except Exception:
        st.write(f"🧠 {tag}: (memory calc failed)")

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

# ---------- Adaptive top-K temperature tuning ----------
def choose_adaptive_K(n_rows):
    # Smooth adaptive K that scales with slate size
    return int(np.clip(round(0.14 * n_rows), 12, 25))

def tune_temperature_for_topk_adaptive(p_oof, y, K=None, T_grid=np.linspace(0.8, 1.6, 17)):
    y = np.asarray(y).astype(int)
    K = choose_adaptive_K(len(y)) if K is None else int(K)
    logits = logit(np.clip(p_oof, 1e-6, 1-1e-6))
    best_T, best_hits = 1.0, -1
    for T in T_grid:
        p_adj = expit(logits * T)
        order = np.argsort(-p_adj)
        hits = int(y[order][:K].sum())
        if hits > best_hits:
            best_hits, best_T = hits, float(T)
    return best_T, K, int(best_hits)

# ===================== Vectorized Overlays & Ratings =====================
def _series(df, name, default=np.nan):
    return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)), errors="coerce")

def _strs(df, name):
    return df.get(name, pd.Series("", index=df.index)).astype(str).fillna("")

def rate_weather_vectorized(df):
    temp = _series(df, "temp")
    hum  = _series(df, "humidity")
    wind = _series(df, "wind_mph")
    wdir = _strs(df, "wind_dir_string").str.lower()
    cond = _strs(df, "condition").str.lower()

    temp_rating = pd.Series("?", index=df.index)
    temp_rating[(temp>=75)&(temp<=85)] = "Excellent"
    temp_rating[((temp>=68)&(temp<75))|((temp>85)&(temp<=90))] = "Good"
    temp_rating[((temp>=60)&(temp<68))|((temp>90)&(temp<=95))] = "Fair"
    temp_rating[(temp<60)|(temp>95)] = "Poor"

    humidity_rating = pd.Series("?", index=df.index)
    humidity_rating[hum>=60] = "Excellent"
    humidity_rating[(hum>=45)&(hum<60)] = "Good"
    humidity_rating[(hum>=30)&(hum<45)] = "Fair"
    humidity_rating[hum<30] = "Poor"

    wind_rating = pd.Series("?", index=df.index)
    wind_rating[wind<6] = "Excellent"
    wind_rating[(wind>=6)&(wind<12)] = "Good"
    wind_rating[(wind>=12)&(wind<18)] = np.where(wdir.str.contains("in"), "Fair", "Good")
    wind_rating[wind>=18] = np.where(wdir.str.contains("in"), "Poor", "Fair")

    condition_rating = pd.Series("?", index=df.index)
    condition_rating[cond.str.contains("clear|sun|outdoor")] = "Excellent"
    condition_rating[cond.str.contains("cloud|partly")] = "Good"
    condition_rating[cond.str.contains("rain|fog")] = "Poor"
    condition_rating[(~cond.str.contains("clear|sun|outdoor|cloud|partly|rain|fog")) & (cond!="")] = "Fair"

    return pd.DataFrame({
        "temp_rating": temp_rating,
        "humidity_rating": humidity_rating,
        "wind_rating": wind_rating,
        "condition_rating": condition_rating
    }, index=df.index)

def overlay_multiplier_vectorized(df):
    # pull inputs
    temp     = _series(df, "temp")
    humidity = _series(df, "humidity")
    wind     = _series(df, "wind_mph")
    wdir     = _strs(df, "wind_dir_string").str.lower()
    roof     = _strs(df, "roof_status").str.lower()
    altitude = _series(df, "park_altitude")

    # park factors (prefer hand-specific then base)
    pf_base  = _series(df, "park_hr_rate")
    pf_hand  = pd.Series(np.nan, index=df.index)
    # choose batter hand
    hand = _strs(df, "batter_hand").str.upper()
    hand = np.where(hand=="", _strs(df, "stand").str.upper(), hand)
    pf_rhb = _series(df, "park_hr_pct_rhb")
    pf_lhb = _series(df, "park_hr_pct_lhb")
    pf_hand = np.where(hand=="R", pf_rhb, np.where(hand=="L", pf_lhb, np.nan))

    def cap_pf(x):
        return np.clip(x, 0.80, 1.22)

    edge = np.ones(len(df), dtype=np.float64)
    pfs = np.where(~np.isnan(pf_hand), cap_pf(pf_hand), np.where(~np.isnan(pf_base), cap_pf(pf_base), np.nan))
    edge *= np.where(np.isnan(pfs), 1.0, pfs)

    # batter windows
    b_pull = _series(df, "b_pull_rate_7")
    b_pull = np.where(np.isnan(b_pull), _series(df, "b_pull_rate_14"), b_pull)
    b_fb   = _series(df, "b_fb_rate_7")
    b_fb   = np.where(np.isnan(b_fb), _series(df, "b_fb_rate_14"), b_fb)
    b_brl  = _series(df, "b_barrel_rate_7")
    b_brl  = np.where(np.isnan(b_brl), _series(df, "b_barrel_rate_14"), b_brl)
    b_hot  = _series(df, "b_hr_per_pa_7")
    b_hot  = np.where(np.isnan(b_hot), _series(df, "b_hr_per_pa_5"), b_hot)

    # pitcher FB
    p_fb = _series(df, "p_fb_rate_14")
    p_fb = np.where(np.isnan(p_fb), _series(df, "p_fb_rate_7"), p_fb)
    roof_closed = roof.str.contains("closed|indoor|domed", regex=True)

    # barrel effect
    edge *= np.where(b_brl>=0.12, 1.04, np.where(b_brl>=0.08, 1.02, 1.0))
    # hot/cold HR rate
    edge *= np.where(b_hot>0.09, 1.04, np.where(b_hot<0.025, 0.97, 1.0))

    # altitude
    edge *= np.where(altitude>=5000, 1.05, np.where(altitude>=3000, 1.02, 1.0))

    # temp / humidity
    with np.errstate(invalid='ignore'):
        edge *= np.power(1.035, (np.nan_to_num(temp, nan=70.0) - 70.0) / 10.0)
    edge *= np.where(humidity>=65, 1.02, np.where(humidity<=35, 0.98, 1.0))

    # wind directional effects
    pulled_field = np.where(hand=="R", "lf", "rf")
    wind_factor = np.ones(len(df), dtype=np.float64)
    valid_wind = (wind>=6) & (wdir!="")
    out = wdir.str.contains("out")
    inn = wdir.str.contains("in")
    has_lf = wdir.str.contains("lf")
    has_rf = wdir.str.contains("rf")
    has_cf = wdir.str.contains("cf|center")

    hi_pull = (~np.isnan(b_pull)) & (b_pull>=0.35)
    lo_pull = (~np.isnan(b_pull)) & (b_pull<=0.28)
    hi_bfb  = (~np.isnan(b_fb)) & (b_fb>=0.22)
    hi_pfb  = (~np.isnan(p_fb)) & (p_fb>=0.25)

    OUT_CF_BOOST, OUT_PULL_BOOST, OPPO_TINY = 1.11, 1.20, 1.05
    IN_CF_FADE, IN_PULL_FADE = 0.92, 0.85

    # CF wind
    wind_factor *= np.where(valid_wind & has_cf & hi_bfb & out, OUT_CF_BOOST, 1.0)
    wind_factor *= np.where(valid_wind & has_cf & hi_bfb & inn, IN_CF_FADE, 1.0)

    # Pull-side wind
    wind_factor *= np.where(valid_wind & has_lf & (pulled_field=="lf") & hi_pull & out, OUT_PULL_BOOST, 1.0)
    wind_factor *= np.where(valid_wind & has_lf & (pulled_field=="lf") & hi_pull & inn, IN_PULL_FADE, 1.0)
    wind_factor *= np.where(valid_wind & has_rf & (pulled_field=="rf") & hi_pull & out, OUT_PULL_BOOST, 1.0)
    wind_factor *= np.where(valid_wind & has_rf & (pulled_field=="rf") & hi_pull & inn, IN_PULL_FADE, 1.0)

    # Oppo tiny boost
    wind_factor *= np.where(valid_wind & out & lo_pull & has_lf & (pulled_field=="rf"), OPPO_TINY, 1.0)
    wind_factor *= np.where(valid_wind & out & lo_pull & has_rf & (pulled_field=="lf"), OPPO_TINY, 1.0)

    # pitcher FB modulation
    wind_factor *= np.where(valid_wind & (out|inn) & hi_pfb, np.where(out, 1.05, 0.97), 1.0)

    # roof damp
    wind_factor = np.where(roof_closed.values, 1.0 + (wind_factor - 1.0) * 0.35, wind_factor)

    # magnitude extra
    extra = np.maximum(0.0, (np.nan_to_num(wind, nan=0.0) - 8.0) / 3.0)
    wind_factor *= np.where(valid_wind & out, np.minimum(1.08, 1.0 + 0.01 * extra), 1.0)
    wind_factor *= np.where(valid_wind & inn, np.maximum(0.92, 1.0 - 0.01 * extra), 1.0)

    edge *= wind_factor

    # combined rule on temp+wind to nudge
    cond_boost = ((temp>=75) & (wind>=7) & out & (~roof_closed))
    cond_small = ((temp>=65) & (wind>=5) & (~roof_closed))
    edge *= np.where(cond_boost, 1.05, np.where(cond_small, 1.02, 0.985))

    # platoon tiny
    p_hand = _strs(df, "pitcher_hand").str.upper()
    p_hand = np.where(p_hand=="", _strs(df, "p_throws").str.upper(), p_hand)
    same_hand = (hand == p_hand)
    edge *= np.where(same_hand, 0.995, 1.01)

    return np.clip(edge, 0.68, 1.44).astype(np.float32)

def weak_pitcher_factor_vectorized(df):
    def pick(*cols):
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        for c in cols:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                out = out.where(~out.isna(), v)
        return out

    factor = np.ones(len(df), dtype=np.float64)

    hr3 = pick("p_rolling_hr_3", "p_hr_count_3")
    pa3 = pick("p_rolling_pa_3")
    with np.errstate(invalid='ignore', divide='ignore'):
        hr_rate_short = hr3 / pa3
    ss_shrink = np.minimum(1.0, np.nan_to_num(pa3, nan=0.0) / 30.0)

    cond_hi  = (hr_rate_short >= 0.10) & (pa3 > 0)
    cond_mid = (hr_rate_short >= 0.07) & (hr_rate_short < 0.10) & (pa3 > 0)
    factor *= np.where(np.nan_to_num(cond_hi, nan=False),  (1.12 * (0.5 + 0.5 * ss_shrink)), 1.0)
    factor *= np.where(np.nan_to_num(cond_mid, nan=False), (1.06 * (0.5 + 0.5 * ss_shrink)), 1.0)

    brl14 = pick("p_fs_barrel_rate_14", "p_barrel_rate_14", "p_hard_hit_rate_14")
    brl30 = pick("p_fs_barrel_rate_30", "p_barrel_rate_30", "p_hard_hit_rate_30")
    qoc = np.nanmax(np.vstack([brl14.values, brl30.values]), axis=0)
    factor *= np.where(qoc>=0.11, 1.07, np.where(qoc>=0.09, 1.04, 1.0))

    fb14 = pick("p_fb_rate_14", "p_fb_rate_7", "p_fb_rate", "p_fb_pct")
    gb14 = pick("p_gb_rate_14", "p_gb_rate_7", "p_gb_rate", "p_gb_pct")
    factor *= np.where(fb14>=0.42, 1.04, np.where(fb14>=0.38, 1.02, 1.0))
    factor *= np.where((~np.isnan(gb14)) & (gb14<=0.40), 1.02, 1.0)

    bb_rate = pick("p_bb_rate_14", "p_bb_rate_30", "p_bb_rate")
    factor *= np.where(bb_rate>=0.09, 1.02, 1.0)

    xw = pick("p_xwoba_con_14", "p_xwoba_con_30", "p_xwoba_con")
    factor *= np.where(xw>=0.40, 1.05, np.where(xw>=0.36, 1.03, 1.0))

    ev_allowed = pick("p_avg_exit_velo_14", "p_avg_exit_velo_7", "p_avg_exit_velo_30",
                      "p_exit_velocity_avg", "p_avg_exit_velo")
    factor *= np.where(ev_allowed>=90.0, 1.03, 1.0)

    b_hand = _strs(df, "batter_hand").str.upper()
    b_hand = np.where(b_hand=="", _strs(df, "stand").str.upper(), b_hand)
    p_hand = _strs(df, "pitcher_hand").str.upper()
    p_hand = np.where(p_hand=="", _strs(df, "p_throws").str.upper(), p_hand)

    p_platoon_vl = pick("p_hr_pa_vl_30", "p_hr_pa_vl_14", "p_hr_pa_vl")
    p_platoon_vr = pick("p_hr_pa_vr_30", "p_hr_pa_vr_14", "p_hr_pa_vr")
    p_platoon = np.where(b_hand=="L", p_platoon_vl, p_platoon_vr)
    factor *= np.where(p_platoon>=0.06, 1.05, np.where(p_platoon>=0.04, 1.03, 1.0))

    opp = ((b_hand=="L") & (p_hand=="R")) | ((b_hand=="R") & (p_hand=="L"))
    factor *= np.where(opp, 1.015, 1.0)

    return np.clip(factor, 0.90, 1.18).astype(np.float32)

def short_term_hot_factor_vectorized(df):
    def pick(*cols):
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        for c in cols:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                out = out.where(~out.isna(), v)
        return out

    ev = pick("b_avg_exit_velo_5","b_avg_exit_velo_3")
    la = pick("b_la_mean_5","b_la_mean_3")
    br = pick("b_barrel_rate_5","b_barrel_rate_3")
    factor = np.ones(len(df), dtype=np.float64)
    factor *= np.where(ev>=91, 1.03, 1.0)
    factor *= np.where((la>=12)&(la<=24), 1.02, 1.0)
    factor *= np.where(br>=0.12, 1.05, np.where(br>=0.08, 1.02, 1.0))
    return np.clip(factor, 0.96, 1.10).astype(np.float32)

def compute_overlay_cols_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- ratings (expects a DataFrame aligned to df.index) ---
    ratings = rate_weather_vectorized(df)  # must return columns like temp_rating, humidity_rating, etc.

    # --- overlay components (vectorized helpers should return 1-D arrays/Series length == len(df)) ---
    overlay = overlay_multiplier_vectorized(df).astype(np.float32)
    weak_p  = weak_pitcher_factor_vectorized(df).astype(np.float32)
    hot_b   = short_term_hot_factor_vectorized(df).astype(np.float32)

    # ensure 1-D numpy arrays
    overlay = np.asarray(overlay, dtype=np.float32).reshape(-1)
    weak_p  = np.asarray(weak_p,  dtype=np.float32).reshape(-1)
    hot_b   = np.asarray(hot_b,   dtype=np.float32).reshape(-1)

    # raw multiplier with caps
    final_raw = np.clip(
        np.clip(overlay, 0.68, 1.44) *
        np.clip(weak_p,  0.90, 1.18) *
        np.clip(hot_b,   0.96, 1.10),
        0.60, 1.65
    ).astype(np.float32)

    # --- uncertainty-aware shrinkage ---
    roof = df.get("roof_status", pd.Series("", index=df.index)).astype(str).str.lower()
    roof_closed = roof.str.contains("closed|indoor|domed", regex=True)

    has_temp = df["temp"].notna()      if "temp"      in df.columns else pd.Series(False, index=df.index)
    has_hum  = df["humidity"].notna()  if "humidity"  in df.columns else pd.Series(False, index=df.index)
    has_wind = df["wind_mph"].notna()  if "wind_mph"  in df.columns else pd.Series(False, index=df.index)

    conf_base = (has_temp.astype(float) + has_hum.astype(float) + has_wind.astype(float)) / 3.0
    conf_roof = np.where(roof_closed.to_numpy(), 0.35, 1.0)

    confidence = np.clip(conf_base.to_numpy(dtype=np.float32) * conf_roof, 0.0, 1.0)
    alpha = 0.5 + 0.5 * confidence  # shape (n,)

    final_mult = (1.0 + alpha * (final_raw - 1.0)).astype(np.float32)

    # assemble all new cols in one shot to avoid fragmentation
    new_cols = pd.DataFrame({
        "overlay_multiplier": overlay,
        "weak_pitcher_factor": weak_p,
        "hot_streak_factor": hot_b,
        "final_multiplier_raw": final_raw,
        "final_multiplier": final_mult,
    }, index=df.index)

    if isinstance(ratings, pd.DataFrame):
        new_cols = pd.concat([new_cols, ratings], axis=1)

    return pd.concat([df, new_cols], axis=1)
# ===================== APP START =====================

# Optional learning ranker upload (fail-closed parity check)
lr_file = st.file_uploader("Optional: upload learning_ranker.pkl", type=["pkl"], key="lrpk")

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read_cached(event_file)
        today_df = safe_read_cached(today_file)

        # Basic cleaning & type fixes
        for dfname, df in [("event_df", event_df), ("today_df", today_df)]:
            if find_duplicate_columns(df):
                st.error(f"Duplicate columns in {dfname}"); st.stop()
        event_df = fix_types(dedup_columns(event_df.dropna(axis=1, how='all'))).reset_index(drop=True)
        today_df = fix_types(dedup_columns(today_df.dropna(axis=1, how='all'))).reset_index(drop=True)

        st.write(f"event_df shape: {event_df.shape}, today_df shape: {today_df.shape}")
        st.write(f"event_df memory (MB): {event_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        st.write(f"today_df memory (MB): {today_df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No 'hr_outcome' column in event data."); st.stop()
    st.success("✅ 'hr_outcome' found.")

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

    # ---- Feature intersection ----
    feature_cols = sorted(list(set(get_valid_feature_cols(event_df)) & set(get_valid_feature_cols(today_df))))
    st.write(f"Feature count before filtering: {len(feature_cols)}")

    X = clean_X(event_df[feature_cols])
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    # ---- Drop NaN-heavy / near-constant / dup-correlated ----
    nan_pct = X.isna().mean()
    drop_cols = nan_pct[nan_pct > 0.30].index.tolist()
    if drop_cols:
        X = X.drop(columns=drop_cols); X_today = X_today.drop(columns=drop_cols, errors='ignore')

    nzv_cols = X.loc[:, X.nunique() <= 2].columns.tolist()
    if nzv_cols:
        X = X.drop(columns=nzv_cols); X_today = X_today.drop(columns=nzv_cols, errors='ignore')

    corrs = X.corr().abs()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    if to_drop:
        X = X.drop(columns=to_drop); X_today = X_today.drop(columns=to_drop, errors='ignore')

    X = winsorize_clip(X); X_today = winsorize_clip(X_today)

    # ---- Chronological ordering ----
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

    # ---- Outlier removal (prevents tail explosions) ----
    st.write("🚦 Outlier removal...")
    X_filtered, y_filtered = remove_outliers(X, y, method="iforest", contamination=0.012)
    dates_aligned = dates_aligned.loc[X_filtered.index].reset_index(drop=True)
    X = X_filtered.reset_index(drop=True).copy()
    y = pd.Series(y_filtered).reset_index(drop=True)
    st.write(f"✅ Rows after outlier removal: {X.shape[0]}")

    # ---- Fill missing & cast ----
    X = X.fillna(-1).astype(np.float32)
    X_today = X_today.fillna(-1).astype(np.float32)
    nan_inf_check(X_today, "X_today (pinned)")

    # ---------- Target Encoding ----------
    cat_cols_available = [c for c in ["park","team_code","batter_hand","pitcher_team_code"] if c in event_df.columns]
    cats_full = event_df[cat_cols_available].copy()
    if order_idx is not None: cats_full = cats_full.loc[order_idx].reset_index(drop=True)
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

    n_splits = 2
    folds = embargo_time_splits(dates_aligned, n_splits=n_splits, embargo_days=1)

    te_maps = {}; global_means = {}
    for name, ser in te_specs:
        oof_vals, fmap, gmean = oof_target_encode(ser, y.values, dates_aligned, folds, smoothing=50.0)
        X[name] = oof_vals.astype(np.float32)
        te_maps[name] = fmap
        global_means[name] = gmean

    def _map_series_to_te(series, fmap, gmean):
        s = series.fillna("__NA__").astype(str)
        return s.map(lambda v: fmap.get(v, gmean)).astype(np.float32)

    # ---- Map TE to today via concat (avoid fragmented frame.insert) ----
    te_today_parts = {}
    if "park" in today_df.columns:
        te_today_parts["te_park"] = _map_series_to_te(today_df["park"], te_maps.get("te_park", {}), global_means.get("te_park", y.mean()))
    if "team_code" in today_df.columns:
        te_today_parts["te_team"] = _map_series_to_te(today_df["team_code"], te_maps.get("te_team", {}), global_means.get("te_team", y.mean()))
    if set(["park","batter_hand"]).issubset(today_df.columns):
        te_today_parts["te_park_hand"] = _map_series_to_te(
            _combo(today_df["park"], today_df["batter_hand"]),
            te_maps.get("te_park_hand", {}), global_means.get("te_park_hand", y.mean())
        )
    if set(["pitcher_team_code","batter_hand"]).issubset(today_df.columns):
        te_today_parts["te_pteam_hand"] = _map_series_to_te(
            _combo(today_df["pitcher_team_code"], today_df["batter_hand"]),
            te_maps.get("te_pteam_hand", {}), global_means.get("te_pteam_hand", y.mean())
        )

    if te_today_parts:
        X_today = pd.concat([X_today, pd.DataFrame(te_today_parts)], axis=1)
        X_today = X_today.copy()  # defragment

    # ---------- Build overlay features for TRAIN copy (for OOF retune diagnostics) ----------
    event_aligned = event_df.copy()
    if order_idx is not None: event_aligned = event_aligned.loc[order_idx].reset_index(drop=True)
    event_aligned = event_aligned.loc[X.index].reset_index(drop=True)

    debug_mem("Before overlay TRAIN", event_aligned)
    event_aligned = compute_overlay_cols_vectorized(event_aligned)
    event_aligned = event_aligned.copy()  # defragment
    debug_mem("After overlay TRAIN", event_aligned)
    # ========== Train/Validation via Embargoed Time Splits ==========
    seeds = [42, 101, 202, 404]

    # === TT-Aug std vector aligned to X_today ===
    # compute from train; reindex to today's columns; fill gaps with train median
    feat_std_train = pd.Series(np.asarray(X.std(axis=0), dtype=float), index=X.columns)
    feat_std_vec = feat_std_train.reindex(X_today.columns)
    fill_val = float(np.nanmedian(feat_std_train.values)) if np.isfinite(np.nanmedian(feat_std_train.values)) else 1e-3
    feat_std_vec = np.maximum(1e-6, feat_std_vec.fillna(fill_val).to_numpy(dtype=np.float32))

    def tt_aug_preds(clf, Xtd, B=3, noise_scale=0.003, kind="proba"):
        """Stochastic test-time ensembling with column-aligned noise."""
        # make sure we operate on a NumPy matrix with the same column order as feat_std_vec
        if isinstance(Xtd, pd.DataFrame):
            Xtd_mat = Xtd.to_numpy(dtype=np.float32)
        else:
            Xtd_mat = np.asarray(Xtd, dtype=np.float32)

        preds = []
        for _ in range(B):
            noise = np.random.normal(0.0, noise_scale, size=Xtd_mat.shape).astype(np.float32)
            # scale noise per-feature; broadcast as (1, n_features)
            noise *= feat_std_vec[np.newaxis, :]
            Xn = Xtd_mat + noise
            if kind == "ranker":
                preds.append(clf.predict(Xn))
            else:
                if hasattr(clf, "predict_proba"):
                    preds.append(clf.predict_proba(Xn)[:, 1])
                else:
                    preds.append(expit(clf.predict(Xn)).astype(np.float32))
        return np.mean(preds, axis=0)

    P_xgb_oof = np.zeros(len(y), dtype=np.float32)
    P_lgb_oof = np.zeros(len(y), dtype=np.float32)
    P_cat_oof = np.zeros(len(y), dtype=np.float32)
    P_xgb_today, P_lgb_today, P_cat_today = [], [], []

    fold_times = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        t_fold_start = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        spw_fold = max(1.0, (len(y_tr) - y_tr.sum()) / max(1.0, y_tr.sum()))

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

            preds_xgb_va.append(xgb_clf.predict_proba(X_va)[:, 1])
            preds_lgb_va.append(lgb_clf.predict_proba(X_va)[:, 1])
            preds_cat_va.append(cat_clf.predict_proba(X_va)[:, 1])

            # TT-Aug for today predictions (aligned noise)
            preds_xgb_td.append(tt_aug_preds(xgb_clf, X_today, B=3, noise_scale=0.003))
            preds_lgb_td.append(tt_aug_preds(lgb_clf, X_today, B=3, noise_scale=0.003))
            preds_cat_td.append(tt_aug_preds(cat_clf, X_today, B=3, noise_scale=0.003))

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
        st.write(
            f"Fold {fold + 1}/{len(folds)} finished in {timedelta(seconds=int(fold_time))}. "
            f"Est. {timedelta(seconds=int(est_time_left))} left."
        )

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
        # TT-Aug on ranker for today (aligned)
        ranker_today_parts.append(tt_aug_preds(rk, X_today, B=3, noise_scale=0.003, kind="ranker"))

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
    P_today_meta = meta.predict_proba(scaler_meta.transform(P_today_base))[:, 1]

    # ---------- Calibration (Isotonic on OOF meta) + Adaptive K temperature ----------
    st.markdown("### 📊 Calibration (Isotonic + Adaptive-K Temp)")
    oof_pred_meta = meta.predict_proba(X_meta_s)[:, 1]
    auc_oof = roc_auc_score(y, oof_pred_meta)
    ll_oof  = log_loss(y, oof_pred_meta)
    st.success(f"OOF Meta AUC: {auc_oof:.4f} | OOF Meta LogLoss: {ll_oof:.4f}")

    ir = IsotonicRegression(out_of_bounds="clip")
    y_oof_iso = ir.fit_transform(oof_pred_meta, y.values)
    today_iso = ir.transform(P_today_meta)

    # Adaptive K based on slate size
    K_adapt = choose_adaptive_K(len(y))
    best_T, K_used, hits_at_K = tune_temperature_for_topk_adaptive(y_oof_iso, y.values, K=K_adapt)
    st.write(f"Adaptive K used: {K_used} | Best T: {best_T:.3f} | OOF Hits@K: {hits_at_K}")
    logits_today = logit(np.clip(today_iso, 1e-6, 1-1e-6))
    today_iso_t = expit(logits_today * best_T)

    # ---------- Park/hand Bayesian prior blend into p_base ----------
    prior_today = X_today.get("te_park_hand", pd.Series(y.mean(), index=pd.RangeIndex(len(X_today)))).astype(float).values
    beta_prior = 0.06
    # initial calibrated base with prior
    p_base_calibrated = (1.0 - beta_prior) * today_iso_t + beta_prior * prior_today

    # Also prepare OOF p_base with same prior for micro weight retune
    prior_oof = X.get("te_park_hand", pd.Series(y.mean(), index=pd.RangeIndex(len(X)))).astype(float).values
    logits_oof = logit(np.clip(y_oof_iso, 1e-6, 1-1e-6))
    y_oof_iso_t = expit(logits_oof * best_T)
    p_oof_cal = (1.0 - beta_prior) * y_oof_iso_t + beta_prior * prior_oof
    # ---------- Handedness-segmented small models (blend into base preds) ----------
    def segment_indices(df_ref):
        hand = df_ref.get("batter_hand", df_ref.get("stand", pd.Series("R", index=df_ref.index))).astype(str).str.upper().fillna("R")
        seg_R = hand != "L"
        seg_L = hand == "L"
        return seg_L.values, seg_R.values

    # make sure we have an aligned event copy for OOF diagnostics/overlays (should exist from earlier)
    try:
        event_aligned
    except NameError:
        event_aligned = event_df.copy()
        if order_idx is not None:
            event_aligned = event_aligned.loc[order_idx].reset_index(drop=True)
        event_aligned = event_aligned.loc[X.index].reset_index(drop=True)

    segL_idx, segR_idx = segment_indices(event_aligned)
    segL_today, segR_today = segment_indices(today_df)

    def train_segmented_preds(mask_tr, mask_td):
        # train on subset using same folds filtered; fallback if too small
        idx = np.where(mask_tr)[0]
        if len(idx) < 200:
            return None, None, None  # too small, skip
        # map to local arrays
        X_loc = X.iloc[idx]; y_loc = y.iloc[idx]
        # simple CV: reuse global folds but intersect indices
        P_oof = np.zeros(len(y_loc), dtype=np.float32)
        P_td_parts = []
        for (tr_idx, va_idx) in folds:
            tr_m = np.intersect1d(idx, tr_idx, assume_unique=False)
            va_m = np.intersect1d(idx, va_idx, assume_unique=False)
            if len(tr_m) == 0 or len(va_m) == 0:
                continue
            # localize to subset positions
            loc_tr = np.searchsorted(idx, tr_m)
            loc_va = np.searchsorted(idx, va_m)
            X_tr, X_va = X_loc.iloc[loc_tr], X_loc.iloc[loc_va]
            y_tr, y_va = y_loc.iloc[loc_tr], y_loc.iloc[loc_va]
            spw_fold = max(1.0, (len(y_tr) - y_tr.sum()) / max(1.0, y_tr.sum()))
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=700, learning_rate=0.03, num_leaves=63,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                reg_lambda=2.0, n_jobs=1, is_unbalance=True, random_state=77
            )
            lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            P_oof[loc_va] = lgb_clf.predict_proba(X_va)[:, 1]
            # predict today subset
            X_td_sub = X_today[mask_td]
            if len(X_td_sub):
                P_td_parts.append(lgb_clf.predict_proba(X_td_sub)[:, 1])
        P_td = np.mean(P_td_parts, axis=0) if P_td_parts else None
        return P_oof, P_td, len(idx)

    P_segL_oof, P_segL_td, nL = train_segmented_preds(segL_idx, segL_today)
    P_segR_oof, P_segR_td, nR = train_segmented_preds(segR_idx, segR_today)

    # Blend segmented preds into base meta preds (gentle 50% where available)
    P_today_meta_seg = P_today_meta.copy()
    if P_segL_td is not None:
        P_today_meta_seg[segL_today] = 0.5 * P_today_meta_seg[segL_today] + 0.5 * np.asarray(P_segL_td)
    if P_segR_td is not None:
        P_today_meta_seg[segR_today] = 0.5 * P_today_meta_seg[segR_today] + 0.5 * np.asarray(P_segR_td)

    # Replace calibrated base with segmented-augmented calibrated+prior
    logits_today_seg = logit(np.clip(P_today_meta_seg, 1e-6, 1-1e-6))
    p_base = (1.0 - beta_prior) * expit(logits_today_seg * best_T) + beta_prior * prior_today

    # ---------- Build TODAY overlay columns ----------
    def compute_overlay_cols(df):
        df = df.copy()
        ratings_df = df.apply(rate_weather, axis=1)
        for col in ratings_df.columns:
            df[col] = ratings_df[col]
        df["overlay_multiplier"] = df.apply(overlay_multiplier, axis=1)
        df["weak_pitcher_factor"] = df.apply(weak_pitcher_factor, axis=1).astype(np.float32)
        df["hot_streak_factor"] = df.apply(short_term_hot_factor, axis=1).astype(np.float32)
        df["final_multiplier_raw"] = (
            df["overlay_multiplier"].astype(float).clip(0.68, 1.44)
            * df["weak_pitcher_factor"].astype(float)
            * df["hot_streak_factor"].astype(float)
        ).clip(0.60, 1.65).astype(np.float32)
        roof = df.get("roof_status", pd.Series("", index=df.index)).astype(str).str.lower()
        roof_closed = roof.str.contains("closed|indoor|domed", regex=True)
        has_temp = df["temp"].notna() if "temp" in df.columns else pd.Series(False, index=df.index)
        has_hum = df["humidity"].notna() if "humidity" in df.columns else pd.Series(False, index=df.index)
        has_wind = df["wind_mph"].notna() if "wind_mph" in df.columns else pd.Series(False, index=df.index)
        conf_base = (has_temp.astype(float) + has_hum.astype(float) + has_wind.astype(float)) / 3.0
        conf_roof = np.where(roof_closed, 0.35, 1.0)
        confidence = np.clip(conf_base * conf_roof, 0.0, 1.0)
        alpha = 0.5 + 0.5 * confidence
        df["final_multiplier"] = (1.0 + alpha * (df["final_multiplier_raw"].values - 1.0)).astype(np.float32)
        return df

    today_df = compute_overlay_cols(today_df)

    # ---------- OOF helpers for micro weight retune ----------
    # RRF on OOF
    def _rank_desc(x):
        x = np.asarray(x)
        return pd.Series(-x).rank(method="min").astype(int).values

    r_prob_oof = _rank_desc(p_oof_cal)
    r_ranker_oof = _rank_desc(ranker_oof)
    r_overlay_oof = _rank_desc(
        event_aligned["final_multiplier"].values
        if "final_multiplier" in event_aligned.columns else np.ones_like(r_prob_oof)
    )
    k_rrf = 60.0
    rrf_oof = 1.0/(k_rrf + r_prob_oof) + 1.0/(k_rrf + r_ranker_oof) + 1.0/(k_rrf + r_overlay_oof)
    rrf_oof_z = zscore(rrf_oof)

    # Disagreement penalty on OOF
    disagree_std_oof = np.std(np.vstack([P_xgb_oof, P_lgb_oof, P_cat_oof]), axis=0)
    dis_penalty_oof = np.clip(zscore(disagree_std_oof), 0, 3)

    # ---------- TODAY RRF + disagreement penalty ----------
    r_prob = _rank_desc(p_base)
    r_ranker = _rank_desc(ranker_today)
    r_overlay = _rank_desc(today_df["final_multiplier"].values)
    rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
    rrf_z = zscore(rrf)

    p_xgb = np.mean(P_xgb_today, axis=0)
    p_lgb = np.mean(P_lgb_today, axis=0)
    p_cat = np.mean(P_cat_today, axis=0)
    disagree_std = np.std(np.vstack([p_xgb, p_lgb, p_cat]), axis=0)
    dis_penalty = np.clip(zscore(disagree_std), 0, 3)

    # ---------- 2TB / RBI proxies (needed for tie-break & optional learner features) ----------
    def pick_best_col(df, base, windows=(14, 30, 7, 20, 60, 5, 3)):
        for w in windows:
            c = f"{base}_{w}"
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").astype(float)
        return pd.Series(np.nan, index=df.index, dtype="float32")

    def zsafe(s):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        mu = np.nanmean(s.values); sd = np.nanstd(s.values) + 1e-9
        return pd.Series((s.values - mu) / sd, index=s.index)

    logit_p = logit(np.clip(p_base, 1e-6, 1 - 1e-6))
    b_slg = pick_best_col(today_df, "b_slg")
    b_hh = pick_best_col(today_df, "b_hard_hit_rate")
    b_hc = pick_best_col(today_df, "b_hard_contact_rate")
    b_fb = pick_best_col(today_df, "b_fb_rate")
    b_brl = pick_best_col(today_df, "b_barrel_rate")
    z_slg = zsafe(b_slg.fillna(b_slg.median()))
    z_hh = zsafe(b_hh.fillna(b_hh.median()))
    z_hc = zsafe(b_hc.fillna(b_hc.median()))
    z_fb = zsafe(b_fb.fillna(b_fb.median()))
    z_brl = zsafe(b_brl.fillna(b_brl.median()))

    logits_2tb = (1.40 * logit_p
                  + 0.70 * z_slg.values
                  + 0.45 * z_hh.values
                  + 0.35 * z_brl.values
                  + 0.20 * np.log(today_df["final_multiplier"].values + 1e-9))
    prob_2tb = expit(logits_2tb)

    logits_rbi = (1.20 * logit_p
                  + 0.50 * z_hc.values
                  + 0.35 * z_hh.values
                  + 0.20 * z_fb.values
                  + 0.15 * np.log(today_df["final_multiplier"].values + 1e-9))
    prob_rbi = expit(logits_rbi)

    # ---------- Micro retune of w_prob / w_ranker on OOF (tiny grid, no leakage) ----------
    base_W = DEFAULT_WEIGHTS.copy()
    grid = [0.85, 1.0, 1.15]  # gentle
    best_loss, best_W = 1e9, base_W.copy()
    # Fixed terms (OOF)
    logit_p_oof = logit(np.clip(p_oof_cal, 1e-6, 1-1e-6))
    for m_prob in grid:
        for m_rank in grid:
            W = base_W.copy()
            W["w_prob"] *= m_prob
            W["w_ranker"] *= m_rank
            comb = (W["w_prob"] * logit_p_oof
                    + W["w_overlay"] * np.log(
                        (event_aligned["final_multiplier"].values
                         if "final_multiplier" in event_aligned.columns
                         else np.ones_like(logit_p_oof)) + 1e-9
                      )
                    + W["w_ranker"] * zscore(ranker_oof)
                    + W["w_rrf"] * rrf_oof_z
                    - W["w_penalty"] * dis_penalty_oof)
            p_hat = expit(comb)
            loss = log_loss(y, np.clip(p_hat, 1e-6, 1-1e-6))
            if loss < best_loss:
                best_loss, best_W = loss, W
    st.write(f"OOF micro-retune selected weights: w_prob={best_W['w_prob']:.4f}, w_ranker={best_W['w_ranker']:.4f} (OOF logloss {best_loss:.5f})")

    # ---------- Learning ranker (fail-closed parity check) ----------
    # Build candidate features available *before* final ranking
    feat_map_today = {
        "base_prob":           np.asarray(p_base, dtype=float),
        "logit_p":             logit_p,
        "log_overlay":         np.log(today_df["final_multiplier"].values + 1e-9),
        "ranker_z":            zscore(ranker_today),  # baseline ranker (always available)
        "overlay_multiplier":  today_df.get("overlay_multiplier", pd.Series(1.0, index=today_df.index)).to_numpy(dtype=float),
        "final_multiplier":    today_df.get("final_multiplier",   pd.Series(1.0, index=today_df.index)).to_numpy(dtype=float),
        "final_multiplier_raw":today_df.get("final_multiplier_raw", pd.Series(1.0, index=today_df.index)).to_numpy(dtype=float),
        "rrf_aux":             rrf,               # raw RRF sum (not z), model can learn scale
        "model_disagreement":  disagree_std,
        "prob_2tb":            prob_2tb,
        "prob_rbi":            prob_rbi,
        "temp":                today_df.get("temp", pd.Series(np.nan, index=today_df.index)).to_numpy(dtype=float),
        "humidity":            today_df.get("humidity", pd.Series(np.nan, index=today_df.index)).to_numpy(dtype=float),
        "wind_mph":            today_df.get("wind_mph", pd.Series(np.nan, index=today_df.index)).to_numpy(dtype=float),
        # NOTE: Intentionally *not* providing 'ranked_probability' or 'hr_probability_iso_T' to avoid circulars/leakage.
    }

    # Default: use baseline ranker signal
    ranker_z = zscore(ranker_today)

    if lr_file is not None:
        try:
            bundle = pickle.load(lr_file)
            lbr = bundle.get("model")
            if lbr is None and "models" in bundle:
                models = bundle["models"]
                lbr = models.get("lgb") or next((m for m in models.values() if m is not None), None)

            expected_feats = [str(f) for f in bundle.get("features", [])]

            # Parity check 1: exact feature list present and ordered buildable
            can_build = all(f in feat_map_today for f in expected_feats)

            # Parity check 2: model expects same #features
            n_expected = len(expected_feats)
            if hasattr(lbr, "n_features_in_"):
                can_build = can_build and (lbr.n_features_in_ == n_expected)

            # Parity check 3: sanity variance (avoid all-const inputs)
            has_variance = all(
                np.nanstd(np.asarray(feat_map_today[f], dtype=float)) > 0
                for f in expected_feats
            ) if can_build else False

            # Only apply if all checks pass
            if (lbr is not None) and can_build and has_variance:
                Xrk_today = np.column_stack([np.asarray(feat_map_today[f], dtype=float) for f in expected_feats]).astype(np.float32)
                learned_rank_score = lbr.predict(Xrk_today)
                # Optional weak parity sanity: basic correlation with baseline ranker signal
                try:
                    corr = np.corrcoef(learned_rank_score, ranker_today)[0, 1]
                except Exception:
                    corr = 0.0
                if np.isfinite(corr):
                    st.write(f"Learning ranker applied. Corr(baseline_rk, learned)={corr:.3f}")
                ranker_z = zscore(learned_rank_score)
            else:
                missing = [f for f in expected_feats if f not in feat_map_today]
                st.warning({
                    "learning_ranker_applied": False,
                    "reason": "feature_mismatch_or_shape_or_variance",
                    "expected_count": len(expected_feats),
                    "built_count": sum(f in feat_map_today for f in expected_feats),
                    "missing_features": missing
                })
        except Exception as e:
            st.warning(f"Could not load/apply learning ranker (fail-closed): {e}")

    # ---------- Final blended score using *retuned* weights ----------
    log_overlay = np.log(today_df["final_multiplier"].values + 1e-9)
    W = best_W  # from OOF micro-retune
    ranked_score = expit(
        W["w_prob"]    * logit_p
      + W["w_overlay"] * log_overlay
      + W["w_ranker"]  * zscore(ranker_z)
      + W["w_rrf"]     * zscore(rrf)
      - W["w_penalty"] * dis_penalty
    )
    # ================= Leaderboard Build & Outputs (with tie-breaking) =================
    def build_leaderboard(df, calibrated_probs, final_score, prob_2tb, prob_rbi, label="hr_probability_iso_T"):
        df = df.copy()

        # core scores
        df[label] = np.asarray(calibrated_probs)
        df["ranked_probability"] = np.asarray(final_score)
        df["prob_2tb"] = np.asarray(prob_2tb)
        df["prob_rbi"] = np.asarray(prob_rbi)

        # tie-breaking: ranked_probability ↓, then prob_2tb ↓, then prob_rbi ↓
        df = df.sort_values(by=["ranked_probability", "prob_2tb", "prob_rbi"], ascending=[False, False, False]).reset_index(drop=True)
        df["hr_base_rank"] = df[label].rank(method="min", ascending=False)

        # identifiers if present
        mlb_id_col = None
        for c in ["batter_id", "mlb_id"]:
            if c in df.columns:
                mlb_id_col = c
                break

        cols = []
        if mlb_id_col:
            cols.append(mlb_id_col)
        for c in ["player_name", "team_code", "time"]:
            if c in df.columns:
                cols.append(c)

        cols += [
            label, "ranked_probability",
            "prob_2tb", "prob_rbi",
            "overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
            "final_multiplier_raw", "final_multiplier",
            "temp", "temp_rating", "humidity", "humidity_rating",
            "wind_mph", "wind_rating", "wind_dir_string",
            "condition", "condition_rating",
            # diagnostics
            "rrf_aux", "model_disagreement",
            "hr_outcome",
        ]
        cols = [c for c in cols if c in df.columns]
        out = df[cols].copy()

        # robust rounding helpers (avoid TypeError on non-1D)
        def _safe_round_numeric(series_like, ndigits):
            try:
                return pd.to_numeric(series_like, errors="coerce").astype(float).round(ndigits)
            except Exception:
                return series_like  # leave as-is if it can't be coerced safely

        for c in [label, "ranked_probability", "prob_2tb", "prob_rbi"]:
            if c in out.columns:
                out[c] = _safe_round_numeric(out[c], 4)

        for c in [
            "overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
            "final_multiplier_raw", "final_multiplier", "rrf_aux", "model_disagreement"
        ]:
            if c in out.columns:
                out[c] = _safe_round_numeric(out[c], 3)

        return out

    # Attach diagnostics used in CSV/UI
    today_df = today_df.copy()
    today_df["rrf_aux"] = rrf
    today_df["model_disagreement"] = disagree_std

    leaderboard = build_leaderboard(
        today_df, p_base, ranked_score, prob_2tb, prob_rbi, label="hr_probability_iso_T"
    )

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
        def drift_check(train_df, today_df_in, n=6):
            drifted = []
            # only compare overlapping numeric columns
            common = list(set(train_df.columns) & set(today_df_in.columns))
            for c in common:
                if not (np.issubdtype(train_df[c].dtype, np.number) and np.issubdtype(today_df_in[c].dtype, np.number)):
                    continue
                tmean = np.nanmean(train_df[c].to_numpy(dtype=float))
                tstd  = np.nanstd(train_df[c].to_numpy(dtype=float))
                dmean = np.nanmean(today_df_in[c].to_numpy(dtype=float))
                if tstd > 0 and np.isfinite(tstd) and abs(tmean - dmean) / tstd > n:
                    drifted.append(c)
            return drifted

        drifted = drift_check(X, X_today, n=6)
        if drifted:
            st.markdown("#### ⚡ **Feature Drift Diagnostics**")
            st.write("These features show unusual mean/std changes:", drifted)
    except Exception:
        pass

    gc.collect()
    st.success("✅ HR Prediction pipeline complete. Leaderboard generated and ready.")
    st.caption(
        "Meta-ensemble + calibrated probs (Adaptive-K) + segmented models + prior blend + RRF + disagreement control "
        "+ learner (fail-closed). 2+TB and RBI proxies included with tie-breaking."
        )
