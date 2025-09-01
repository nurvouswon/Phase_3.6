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
# - PERF/ROBUSTNESS FIXES:
#     • TE columns batch-assign to avoid fragmentation warnings
#     • Vectorized overlays (fast) + NA-safe boolean math
#     • TT-Aug noise std aligned to X_today (fix shape mismatch)
#     • Leaderboard rounding fully Series/array-safe
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

# ===================== Overlays & Ratings =====================
def _getv(row, keys, default=np.nan):
    for k in keys:
        if k in row and pd.notnull(row[k]):
            return row[k]
    return default

def _hand(row, batter=True):
    return str(_getv(row, ["stand","batter_hand"] if batter else ["pitcher_hand","p_throws"], "R")).upper() or "R"

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
    if pfs: edge *= pfs[0]

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
            edge *= 1.02

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
        out = ("out" in wind_dir); inn = ("in" in wind_dir)
        has_lf = "lf" in wind_dir; has_rf = "rf" in wind_dir
        has_cf = ("cf" in wind_dir) or ("center" in wind_dir)

        hi_pull = pd.notnull(b_pull) and (b_pull >= PULL_HI)
        lo_pull = pd.notnull(b_pull) and (b_pull <= PULL_LO)
        hi_bfb  = pd.notnull(b_fb)   and (b_fb   >= FB_HI_B)
        hi_pfb  = pd.notnull(p_fb)   and (p_fb   >= FB_HI_P)

        OUT_CF_BOOST, OUT_PULL_BOOST, OPPO_TINY = 1.11, 1.20, 1.05
        IN_CF_FADE, IN_PULL_FADE = 0.92, 0.85

        if has_cf and hi_bfb: wind_factor *= OUT_CF_BOOST if out else IN_CF_FADE if inn else 1.0
        if has_lf and pulled_field == "lf" and hi_pull: wind_factor *= OUT_PULL_BOOST if out else IN_PULL_FADE if inn else 1.0
        if has_rf and pulled_field == "rf" and hi_pull: wind_factor *= OUT_PULL_BOOST if out else IN_PULL_FADE if inn else 1.0

        if out and lo_pull:
            if has_lf and pulled_field == "rf": wind_factor *= OPPO_TINY
            if has_rf and pulled_field == "lf": wind_factor *= OPPO_TINY

        if hi_pfb and (out or inn): wind_factor *= 1.05 if out else 0.97
        if roof_closed: wind_factor = 1.0 + (wind_factor - 1.0) * 0.35

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
    if pd.isna(temp): ratings["temp_rating"]="?"
    elif 75 <= temp <= 85: ratings["temp_rating"]="Excellent"
    elif 68 <= temp < 75 or 85 < temp <= 90: ratings["temp_rating"]="Good"
    elif 60 <= temp < 68 or 90 < temp <= 95: ratings["temp_rating"]="Fair"
    else: ratings["temp_rating"]="Poor"

    hum = _getv(row, ["humidity"])
    if pd.isna(hum): ratings["humidity_rating"]="?"
    elif hum >= 60: ratings["humidity_rating"]="Excellent"
    elif 45 <= hum < 60: ratings["humidity_rating"]="Good"
    elif 30 <= hum < 45: ratings["humidity_rating"]="Fair"
    else: ratings["humidity_rating"]="Poor"

    wind = _getv(row, ["wind_mph"]); wdir = str(_getv(row, ["wind_dir_string"], "")).lower()
    if pd.isna(wind): ratings["wind_rating"]="?"
    elif wind < 6: ratings["wind_rating"]="Excellent"
    elif 6 <= wind < 12: ratings["wind_rating"]="Good"
    elif 12 <= wind < 18: ratings["wind_rating"]="Fair" if "in" in wdir else "Good"
    else: ratings["wind_rating"]="Poor" if "in" in wdir else "Fair"

    cond = str(_getv(row, ["condition"], "")).lower()
    if not cond or cond in ("unknown","none","na"): ratings["condition_rating"]="?"
    elif "clear" in cond or "sun" in cond or "outdoor" in cond: ratings["condition_rating"]="Excellent"
    elif "cloud" in cond or "partly" in cond: ratings["condition_rating"]="Good"
    elif "rain" in cond or "fog" in cond: ratings["condition_rating"]="Poor"
    else: ratings["condition_rating"]="Fair"
    return pd.Series(ratings)

def weak_pitcher_factor(row):
    def _get(*names, default=np.nan):
        for n in names:
            if n in row and pd.notnull(row[n]): return row[n]
        return default

    factor = 1.0
    hr3   = _get("p_rolling_hr_3", "p_hr_count_3")
    pa3   = _get("p_rolling_pa_3")
    if pd.notnull(hr3) and pd.notnull(pa3) and pa3 > 0:
        hr_rate_short = float(hr3) / float(pa3)
        ss_shrink = min(1.0, pa3 / 30.0)
        if hr_rate_short >= 0.10: factor *= (1.12 * (0.5 + 0.5 * ss_shrink))
        elif hr_rate_short >= 0.07: factor *= (1.06 * (0.5 + 0.5 * ss_shrink))

    brl14 = _get("p_fs_barrel_rate_14", "p_barrel_rate_14", "p_hard_hit_rate_14")
    brl30 = _get("p_fs_barrel_rate_30", "p_barrel_rate_30", "p_hard_hit_rate_30")
    qoc = np.nanmax([brl14, brl30]) if any(pd.notnull(x) for x in [brl14, brl30]) else np.nan
    if pd.notnull(qoc):
        v = float(qoc)
        if v >= 0.11: factor *= 1.07
        elif v >= 0.09: factor *= 1.04

    fb14 = _get("p_fb_rate_14", "p_fb_rate_7", "p_fb_rate", "p_fb_pct")
    gb14 = _get("p_gb_rate_14", "p_gb_rate_7", "p_gb_rate", "p_gb_pct")
    if pd.notnull(fb14):
        if float(fb14) >= 0.42: factor *= 1.04
        elif float(fb14) >= 0.38: factor *= 1.02
    if pd.notnull(gb14) and float(gb14) <= 0.40: factor *= 1.02

    bb_rate = _get("p_bb_rate_14", "p_bb_rate_30", "p_bb_rate")
    xwoba_con = _get("p_xwoba_con_14", "p_xwoba_con_30", "p_xwoba_con")
    if pd.notnull(bb_rate) and float(bb_rate) >= 0.09: factor *= 1.02
    if pd.notnull(xwoba_con):
        if float(xwoba_con) >= 0.40: factor *= 1.05
        elif float(xwoba_con) >= 0.36: factor *= 1.03

    ev_allowed = _get("p_avg_exit_velo_14", "p_avg_exit_velo_7", "p_avg_exit_velo_30",
                      "p_exit_velocity_avg", "p_avg_exit_velo")
    if pd.notnull(ev_allowed) and float(ev_allowed) >= 90.0: factor *= 1.03

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

    if (b_hand == "L" and p_hand == "R") or (b_hand == "R" and p_hand == "L"): factor *= 1.015
    return float(np.clip(factor, 0.90, 1.18))

def short_term_hot_factor(row):
    def _first_non_null(row, *cands, default=np.nan):
        for c in cands:
            if c in row and pd.notnull(row[c]): return row[c]
        return default
    ev = _first_non_null(row, "b_avg_exit_velo_5", "b_avg_exit_velo_3")
    la = _first_non_null(row, "b_la_mean_5", "b_la_mean_3")
    br = _first_non_null(row, "b_barrel_rate_5", "b_barrel_rate_3")
    factor = 1.0
    try:
        if pd.notnull(ev) and float(ev) >= 91: factor *= 1.03
        if pd.notnull(la) and 12 <= float(la) <= 24: factor *= 1.02
        if pd.notnull(br):
            if float(br) >= 0.12: factor *= 1.05
            elif float(br) >= 0.08: factor *= 1.02
    except: pass
    return float(np.clip(factor, 0.96, 1.10))

# ===================== APP CONTINUATION =====================

# Optional learning ranker upload (we'll fail closed if not perfect match)
lr_file = st.file_uploader("Optional: upload learning_ranker.pkl", type=["pkl"], key="lrpk")

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read_cached(event_file)
        today_df = safe_read_cached(today_file)

        # Basic cleaning
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

    # ---- Outlier removal ----
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

    # add TE columns to TODAY in one concat (avoids fragmentation)
    te_today_parts = {}
    if "park" in today_df.columns:
        te_today_parts["te_park"] = _map_series_to_te(today_df["park"], te_maps.get("te_park", {}), global_means.get("te_park", y.mean()))
    if "team_code" in today_df.columns:
        te_today_parts["te_team"] = _map_series_to_te(today_df["team_code"], te_maps.get("te_team", {}), global_means.get("te_team", y.mean()))
    if set(["park","batter_hand"]).issubset(today_df.columns):
        te_today_parts["te_park_hand"] = _map_series_to_te(_combo(today_df["park"], today_df["batter_hand"]),
                                                           te_maps.get("te_park_hand", {}), global_means.get("te_park_hand", y.mean()))
    if set(["pitcher_team_code","batter_hand"]).issubset(today_df.columns):
        te_today_parts["te_pteam_hand"] = _map_series_to_te(_combo(today_df["pitcher_team_code"], today_df["batter_hand"]),
                                                            te_maps.get("te_pteam_hand", {}), global_means.get("te_pteam_hand", y.mean()))
    if te_today_parts:
        X_today = pd.concat([X_today, pd.DataFrame(te_today_parts, index=X_today.index)], axis=1).astype(np.float32)

    # ---------- Build overlay features for TRAIN (aligned copy) ----------
    event_aligned = event_df.copy()
    if order_idx is not None: event_aligned = event_aligned.loc[order_idx].reset_index(drop=True)
    event_aligned = event_aligned.loc[X.index].reset_index(drop=True)

    def compute_overlay_cols(df):
        df = df.copy()
        ratings_df = df.apply(rate_weather, axis=1)
        for col in ratings_df.columns: df[col] = ratings_df[col]
        df["overlay_multiplier"] = df.apply(overlay_multiplier, axis=1)
        df["weak_pitcher_factor"] = df.apply(weak_pitcher_factor, axis=1).astype(np.float32)
        df["hot_streak_factor"]   = df.apply(short_term_hot_factor, axis=1).astype(np.float32)
        df["final_multiplier_raw"] = (
            df["overlay_multiplier"].astype(float).clip(0.68, 1.44)
            * df["weak_pitcher_factor"].astype(float)
            * df["hot_streak_factor"].astype(float)
        ).clip(0.60, 1.65).astype(np.float32)
        roof = df.get("roof_status", pd.Series("", index=df.index)).astype(str).str.lower()
        roof_closed = roof.str.contains("closed|indoor|domed", regex=True)
        has_temp = df["temp"].notna() if "temp" in df.columns else pd.Series(False, index=df.index)
        has_hum  = df["humidity"].notna() if "humidity" in df.columns else pd.Series(False, index=df.index)
        has_wind = df["wind_mph"].notna() if "wind_mph" in df.columns else pd.Series(False, index=df.index)
        conf_base = (has_temp.astype(float) + has_hum.astype(float) + has_wind.astype(float)) / 3.0
        conf_roof = np.where(roof_closed, 0.35, 1.0)
        confidence = np.clip(conf_base * conf_roof, 0.0, 1.0)
        alpha = 0.5 + 0.5 * confidence
        df["final_multiplier"] = (1.0 + alpha * (df["final_multiplier_raw"].values - 1.0)).astype(np.float32)
        return df

    event_aligned = compute_overlay_cols(event_aligned)

    # ---------- Train/Validation via Embargoed Time Splits ----------
    seeds = [42, 101, 202, 404]
    P_xgb_oof = np.zeros(len(y), dtype=np.float32)
    P_lgb_oof = np.zeros(len(y), dtype=np.float32)
    P_cat_oof = np.zeros(len(y), dtype=np.float32)
    P_xgb_today, P_lgb_today, P_cat_today = [], [], []

    # === TT-Aug std vector aligned to X_today ===
    feat_std_train = pd.Series(np.asarray(X.std(axis=0), dtype=float), index=X.columns)
    feat_std_vec = feat_std_train.reindex(X_today.columns)
    fill_val = float(np.nanmedian(feat_std_train.values)) if np.isfinite(np.nanmedian(feat_std_train.values)) else 1e-3
    feat_std_vec = np.maximum(1e-6, feat_std_vec.fillna(fill_val).to_numpy(dtype=np.float32))

    def tt_aug_preds(clf, Xtd, B=3, noise_scale=0.003, kind="proba"):
        if isinstance(Xtd, pd.DataFrame):
            Xtd_mat = Xtd.to_numpy(dtype=np.float32)
        else:
            Xtd_mat = np.asarray(Xtd, dtype=np.float32)

        preds = []
        for _ in range(B):
            noise = np.random.normal(0.0, noise_scale, size=Xtd_mat.shape).astype(np.float32)
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
    P_today_meta = meta.predict_proba(scaler_meta.transform(P_today_base))[:,1]

    # ---------- Calibration (Isotonic + Adaptive-K Temp) ----------
    st.markdown("### 📊 Calibration (Isotonic + Adaptive-K Temp)")
    oof_pred_meta = meta.predict_proba(X_meta_s)[:,1]
    auc_oof = roc_auc_score(y, oof_pred_meta)
    ll_oof  = log_loss(y, oof_pred_meta)
    st.success(f"OOF Meta AUC: {auc_oof:.4f} | OOF Meta LogLoss: {ll_oof:.4f}")

    ir = IsotonicRegression(out_of_bounds="clip")
    y_oof_iso = ir.fit_transform(oof_pred_meta, y.values)
    today_iso = ir.transform(P_today_meta)

    K_adapt = choose_adaptive_K(len(y))
    best_T, K_used, hits_at_K = tune_temperature_for_topk_adaptive(y_oof_iso, y.values, K=K_adapt)
    st.write(f"Adaptive K used: {K_used} | Best T: {best_T:.3f} | OOF Hits@K: {hits_at_K}")
    logits_today = logit(np.clip(today_iso, 1e-6, 1-1e-6))
    today_iso_t = expit(logits_today * best_T)

    # ---------- Park/hand Bayesian prior blend ----------
    prior_today = X_today.get("te_park_hand", pd.Series(y.mean(), index=pd.RangeIndex(len(X_today)))).astype(float).values
    beta_prior = 0.06
    p_base_cal = (1.0 - beta_prior) * today_iso_t + beta_prior * prior_today

    prior_oof = X.get("te_park_hand", pd.Series(y.mean(), index=pd.RangeIndex(len(X)))).astype(float).values
    logits_oof = logit(np.clip(y_oof_iso, 1e-6, 1-1e-6))
    y_oof_iso_t = expit(logits_oof * best_T)
    p_oof_cal = (1.0 - beta_prior) * y_oof_iso_t + beta_prior * prior_oof

    # ---------- Handedness-segmented small models ----------
    def segment_indices(df_ref):
        hand = df_ref.get("batter_hand", df_ref.get("stand", pd.Series("R", index=df_ref.index))).astype(str).str.upper().fillna("R")
        seg_R = hand != "L"
        seg_L = hand == "L"
        return seg_L.values, seg_R.values

    segL_idx, segR_idx = segment_indices(event_aligned)
    segL_today, segR_today = segment_indices(today_df)

    def train_segmented_preds(mask_tr, mask_td):
        idx = np.where(mask_tr)[0]
        if len(idx) < 200:  # too small
            return None, None, None
        X_loc = X.iloc[idx]; y_loc = y.iloc[idx]
        P_oof = np.zeros(len(y_loc), dtype=np.float32)
        P_td_parts = []
        for (tr_idx, va_idx) in folds:
            tr_m = np.intersect1d(idx, tr_idx, assume_unique=False)
            va_m = np.intersect1d(idx, va_idx, assume_unique=False)
            if len(tr_m)==0 or len(va_m)==0: continue
            loc_tr = np.searchsorted(idx, tr_m)
            loc_va = np.searchsorted(idx, va_m)
            X_tr, X_va = X_loc.iloc[loc_tr], X_loc.iloc[loc_va]
            y_tr, y_va = y_loc.iloc[loc_tr], y_loc.iloc[loc_va]
            spw_fold = max(1.0, (len(y_tr)-y_tr.sum())/max(1.0, y_tr.sum()))
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=700, learning_rate=0.03, num_leaves=63,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                reg_lambda=2.0, n_jobs=1, is_unbalance=True, random_state=77
            )
            lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            P_oof[loc_va] = lgb_clf.predict_proba(X_va)[:,1]
            X_td_sub = X_today[mask_td]
            if len(X_td_sub):
                P_td_parts.append(lgb_clf.predict_proba(X_td_sub)[:,1])
        P_td = np.mean(P_td_parts, axis=0) if P_td_parts else None
        return P_oof, P_td, len(idx)

    P_segL_oof, P_segL_td, _ = train_segmented_preds(segL_idx, segL_today)
    P_segR_oof, P_segR_td, _ = train_segmented_preds(segR_idx, segR_today)

    P_today_meta_seg = P_today_meta.copy()
    if P_segL_td is not None:
        P_today_meta_seg[segL_today] = 0.5*P_today_meta_seg[segL_today] + 0.5*np.asarray(P_segL_td)
    if P_segR_td is not None:
        P_today_meta_seg[segR_today] = 0.5*P_today_meta_seg[segR_today] + 0.5*np.asarray(P_segR_td)

    p_base = (1.0 - beta_prior) * expit(logit(np.clip(P_today_meta_seg, 1e-6, 1-1e-6)) * best_T) + beta_prior * prior_today

    # ---------- TODAY overlay columns ----------
    today_df = compute_overlay_cols(today_df)

    # ---------- OOF helpers for micro weight retune ----------
    def _rank_desc(x):
        x = np.asarray(x)
        return pd.Series(-x).rank(method="min").astype(int).values

    r_prob_oof    = _rank_desc(p_oof_cal)
    r_ranker_oof  = _rank_desc(ranker_oof)
    r_overlay_oof = _rank_desc(event_aligned["final_multiplier"].values if "final_multiplier" in event_aligned.columns else np.ones_like(r_prob_oof))
    k_rrf = 60.0
    rrf_oof = 1.0/(k_rrf + r_prob_oof) + 1.0/(k_rrf + r_ranker_oof) + 1.0/(k_rrf + r_overlay_oof)
    rrf_oof_z = zscore(rrf_oof)

    disagree_std_oof = np.std(np.vstack([P_xgb_oof, P_lgb_oof, P_cat_oof]), axis=0)
    dis_penalty_oof = np.clip(zscore(disagree_std_oof), 0, 3)

    # ---------- TODAY RRF + disagreement penalty ----------
    r_prob    = _rank_desc(p_base)
    r_ranker  = _rank_desc(ranker_today)
    r_overlay = _rank_desc(today_df["final_multiplier"].values)
    rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
    rrf_z = zscore(rrf)

    p_xgb = np.mean(P_xgb_today, axis=0)
    p_lgb = np.mean(P_lgb_today, axis=0)
    p_cat = np.mean(P_cat_today, axis=0)
    disagree_std = np.std(np.vstack([p_xgb, p_lgb, p_cat]), axis=0)
    dis_penalty = np.clip(zscore(disagree_std), 0, 3)

    # ---------- 2TB / RBI proxies ----------
    def pick_best_col(df, base, windows=(14, 30, 7, 20, 60, 5, 3)):
        for w in windows:
            c = f"{base}_{w}"
            if c in df.columns: return pd.to_numeric(df[c], errors="coerce").astype(float)
        return pd.Series(np.nan, index=df.index, dtype="float32")

    def zsafe(s):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        mu = np.nanmean(s.values); sd = np.nanstd(s.values) + 1e-9
        return pd.Series((s.values - mu)/sd, index=s.index)

    logit_p = logit(np.clip(p_base, 1e-6, 1 - 1e-6))
    b_slg = pick_best_col(today_df, "b_slg")
    b_hh  = pick_best_col(today_df, "b_hard_hit_rate")
    b_hc  = pick_best_col(today_df, "b_hard_contact_rate")
    b_fb  = pick_best_col(today_df, "b_fb_rate")
    b_brl = pick_best_col(today_df, "b_barrel_rate")
    z_slg = zsafe(b_slg.fillna(b_slg.median()))
    z_hh  = zsafe(b_hh.fillna(b_hh.median()))
    z_hc  = zsafe(b_hc.fillna(b_hc.median()))
    z_fb  = zsafe(b_fb.fillna(b_fb.median()))
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

    # ---------- Micro retune of w_prob / w_ranker ----------
    base_W = DEFAULT_WEIGHTS.copy()
    grid = [0.85, 1.0, 1.15]
    best_loss, best_W = 1e9, base_W.copy()
    logit_p_oof = logit(np.clip(p_oof_cal, 1e-6, 1-1e-6))
    for m_prob in grid:
        for m_rank in grid:
            W = base_W.copy()
            W["w_prob"]   *= m_prob
            W["w_ranker"] *= m_rank
            comb = (W["w_prob"]   * logit_p_oof
                  + W["w_overlay"]* np.log(event_aligned["final_multiplier"].values + 1e-9)
                  + W["w_ranker"] * zscore(ranker_oof)
                  + W["w_rrf"]    * rrf_oof_z
                  - W["w_penalty"]* dis_penalty_oof)
            p_hat = expit(comb)
            loss = log_loss(y, np.clip(p_hat, 1e-6, 1-1e-6))
            if loss < best_loss:
                best_loss, best_W = loss, W
    st.write(f"OOF micro-retune selected weights: w_prob={best_W['w_prob']:.4f}, w_ranker={best_W['w_ranker']:.4f} (OOF logloss {best_loss:.5f})")

    # ---------- Learning ranker (fail-closed parity check) ----------
    feat_map_today = {
        "base_prob":           np.asarray(p_base, dtype=float),
        "logit_p":             logit_p,
        "log_overlay":         np.log(today_df["final_multiplier"].values + 1e-9),
        "ranker_z":            zscore(ranker_today),
        "overlay_multiplier":  today_df.get("overlay_multiplier", pd.Series(1.0, index=today_df.index)).to_numpy(dtype=float),
        "final_multiplier":    today_df.get("final_multiplier",   pd.Series(1.0, index=today_df.index)).to_numpy(dtype=float),
        "final_multiplier_raw":today_df.get("final_multiplier_raw",pd.Series(1.0, index=today_df.index)).to_numpy(dtype=float),
        "rrf_aux":             rrf,
        "model_disagreement":  disagree_std,
        "prob_2tb":            prob_2tb,
        "prob_rbi":            prob_rbi,
        "temp":                today_df.get("temp", pd.Series(np.nan, index=today_df.index)).to_numpy(dtype=float),
        "humidity":            today_df.get("humidity", pd.Series(np.nan, index=today_df.index)).to_numpy(dtype=float),
        "wind_mph":            today_df.get("wind_mph", pd.Series(np.nan, index=today_df.index)).to_numpy(dtype=float),
    }

    ranker_z = zscore(ranker_today)
    if lr_file is not None:
        try:
            bundle = pickle.load(lr_file)
            lbr = bundle.get("model")
            if lbr is None and "models" in bundle:
                models = bundle["models"]
                lbr = models.get("lgb") or next((m for m in models.values() if m is not None), None)
            expected_feats = [str(f) for f in bundle.get("features", [])]

            can_build = all(f in feat_map_today for f in expected_feats)
            n_expected = len(expected_feats)
            if hasattr(lbr, "n_features_in_"): can_build = can_build and (lbr.n_features_in_ == n_expected)
            has_variance = all(np.nanstd(np.asarray(feat_map_today[f], dtype=float)) > 0 for f in expected_feats) if can_build else False

            if lbr is not None and can_build and has_variance:
                Xrk_today = np.column_stack([np.asarray(feat_map_today[f], dtype=float) for f in expected_feats]).astype(np.float32)
                learned_rank_score = lbr.predict(Xrk_today)
                try:
                    corr = np.corrcoef(learned_rank_score, ranker_today)[0,1]
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

    # ---------- Final blended score ----------
    log_overlay = np.log(today_df["final_multiplier"].values + 1e-9)
    W = best_W
    ranked_score = expit(
        W["w_prob"]    * logit_p
      + W["w_overlay"] * log_overlay
      + W["w_ranker"]  * zscore(ranker_z)
      + W["w_rrf"]     * zscore(rrf)
      - W["w_penalty"] * dis_penalty
    )

    # ================= Leaderboard Build & Outputs =================
    def build_leaderboard(df, calibrated_probs, final_score, prob_2tb, prob_rbi, label="hr_probability_iso_T"):
        df = df.copy()
        df[label] = np.asarray(calibrated_probs)
        df["ranked_probability"] = np.asarray(final_score)
        df["prob_2tb"] = np.asarray(prob_2tb)
        df["prob_rbi"] = np.asarray(prob_rbi)

        df = df.sort_values(by=["ranked_probability","prob_2tb","prob_rbi"], ascending=[False, False, False]).reset_index(drop=True)
        df["hr_base_rank"] = df[label].rank(method="min", ascending=False)

        mlb_id_col = None
        for c in ["batter_id", "mlb_id"]:
            if c in df.columns:
                mlb_id_col = c; break
        cols = []
        if mlb_id_col: cols.append(mlb_id_col)
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
            "rrf_aux", "model_disagreement",
            "hr_outcome",
        ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    # rounding for readability (guard against non-Series/objects)
    for c in [label, "ranked_probability", "prob_2tb", "prob_rbi"]:
        if c in out.columns and isinstance(out[c], pd.Series):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float).round(4)

    for c in [
        "overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
        "final_multiplier_raw", "final_multiplier", "rrf_aux", "model_disagreement"
    ]:
        if c in out.columns and isinstance(out[c], pd.Series):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float).round(3)

    return out

    # Attach diagnostics
    today_df["rrf_aux"] = rrf
    today_df["model_disagreement"] = disagree_std

    leaderboard = build_leaderboard(
        today_df, p_base, ranked_score, prob_2tb, prob_rbi, label="hr_probability_iso_T"
    )

    # ===== Render Leaderboard =====
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
        def drift_check(train, today, n=6):
            drifted = []
            for c in train.columns:
                if c not in today.columns:
                    continue
                tmean = np.nanmean(train[c]); tstd = np.nanstd(train[c])
                dmean = np.nanmean(today[c])
                if tstd > 0 and abs(tmean - dmean) / tstd > n:
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
