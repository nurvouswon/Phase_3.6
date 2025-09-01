# app.py
# ============================================================
# 🏆 MLB Home Run Predictor — "Max Power" Edition (Fixed)
#
# Fixes:
# - TE reduced to te_park_hand only (aligned train/today)
# - Overlay: only vectorized version (row-apply removed)
# - X_today forced to X.columns order (no shape mismatch)
# - 2 folds preserved
# - CatBoost kept full strength
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
    for v, c in grp_cnt.items():
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

# ========== Overlay + Weather functions ==========
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
    temp     = _series(df, "temp")
    humidity = _series(df, "humidity")
    wind     = _series(df, "wind_mph")
    wdir     = _strs(df, "wind_dir_string").str.lower()
    roof     = _strs(df, "roof_status").str.lower()
    altitude = _series(df, "park_altitude")

    pf_base  = _series(df, "park_hr_rate")
    hand = _strs(df, "batter_hand").str.upper()
    hand = np.where(hand=="", _strs(df, "stand").str.upper(), hand)
    pf_rhb = _series(df, "park_hr_pct_rhb")
    pf_lhb = _series(df, "park_hr_pct_lhb")
    pf_hand = np.where(hand=="R", pf_rhb, np.where(hand=="L", pf_lhb, np.nan))

    def cap_pf(x): return np.clip(x, 0.80, 1.22)

    edge = np.ones(len(df), dtype=np.float64)
    pfs = np.where(~np.isnan(pf_hand), cap_pf(pf_hand), np.where(~np.isnan(pf_base), cap_pf(pf_base), np.nan))
    edge *= np.where(np.isnan(pfs), 1.0, pfs)

    # Batter + pitcher factors (pull, FB, barrel, hot streak)
    b_pull = _series(df, "b_pull_rate_7")
    b_pull = np.where(np.isnan(b_pull), _series(df, "b_pull_rate_14"), b_pull)
    b_fb   = _series(df, "b_fb_rate_7")
    b_fb   = np.where(np.isnan(b_fb), _series(df, "b_fb_rate_14"), b_fb)
    b_brl  = _series(df, "b_barrel_rate_7")
    b_brl  = np.where(np.isnan(b_brl), _series(df, "b_barrel_rate_14"), b_brl)
    b_hot  = _series(df, "b_hr_per_pa_7")
    b_hot  = np.where(np.isnan(b_hot), _series(df, "b_hr_per_pa_5"), b_hot)

    p_fb = _series(df, "p_fb_rate_14")
    p_fb = np.where(np.isnan(p_fb), _series(df, "p_fb_rate_7"), p_fb)
    roof_closed = roof.str.contains("closed|indoor|domed", regex=True)

    edge *= np.where(b_brl>=0.12, 1.04, np.where(b_brl>=0.08, 1.02, 1.0))
    edge *= np.where(b_hot>0.09, 1.04, np.where(b_hot<0.025, 0.97, 1.0))
    edge *= np.where(altitude>=5000, 1.05, np.where(altitude>=3000, 1.02, 1.0))
    with np.errstate(invalid='ignore'):
        edge *= np.power(1.035, (np.nan_to_num(temp, nan=70.0) - 70.0) / 10.0)
    edge *= np.where(humidity>=65, 1.02, np.where(humidity<=35, 0.98, 1.0))

    # Wind directional boosts/fades
    pulled_field = np.where(hand=="R", "lf", "rf")
    wind_factor = np.ones(len(df), dtype=np.float64)
    valid_wind = (wind>=6) & (wdir!="")
    out = wdir.str.contains("out"); inn = wdir.str.contains("in")
    has_lf = wdir.str.contains("lf"); has_rf = wdir.str.contains("rf"); has_cf = wdir.str.contains("cf|center")

    hi_pull = (~np.isnan(b_pull)) & (b_pull>=0.35)
    lo_pull = (~np.isnan(b_pull)) & (b_pull<=0.28)
    hi_bfb  = (~np.isnan(b_fb)) & (b_fb>=0.22)
    hi_pfb  = (~np.isnan(p_fb)) & (p_fb>=0.25)

    OUT_CF_BOOST, OUT_PULL_BOOST, OPPO_TINY = 1.11, 1.20, 1.05
    IN_CF_FADE, IN_PULL_FADE = 0.92, 0.85

    wind_factor *= np.where(valid_wind & has_cf & hi_bfb & out, OUT_CF_BOOST, 1.0)
    wind_factor *= np.where(valid_wind & has_cf & hi_bfb & inn, IN_CF_FADE, 1.0)
    wind_factor *= np.where(valid_wind & has_lf & (pulled_field=="lf") & hi_pull & out, OUT_PULL_BOOST, 1.0)
    wind_factor *= np.where(valid_wind & has_lf & (pulled_field=="lf") & hi_pull & inn, IN_PULL_FADE, 1.0)
    wind_factor *= np.where(valid_wind & has_rf & (pulled_field=="rf") & hi_pull & out, OUT_PULL_BOOST, 1.0)
    wind_factor *= np.where(valid_wind & has_rf & (pulled_field=="rf") & hi_pull & inn, IN_PULL_FADE, 1.0)
    wind_factor *= np.where(valid_wind & out & lo_pull & has_lf & (pulled_field=="rf"), OPPO_TINY, 1.0)
    wind_factor *= np.where(valid_wind & out & lo_pull & has_rf & (pulled_field=="lf"), OPPO_TINY, 1.0)
    wind_factor *= np.where(valid_wind & (out|inn) & hi_pfb, np.where(out, 1.05, 0.97), 1.0)
    wind_factor = np.where(roof_closed.values, 1.0 + (wind_factor - 1.0) * 0.35, wind_factor)

    extra = np.maximum(0.0, (np.nan_to_num(wind, nan=0.0) - 8.0) / 3.0)
    wind_factor *= np.where(valid_wind & out, np.minimum(1.08, 1.0 + 0.01 * extra), 1.0)
    wind_factor *= np.where(valid_wind & inn, np.maximum(0.92, 1.0 - 0.01 * extra), 1.0)

    edge *= wind_factor
    cond_boost = ((temp>=75) & (wind>=7) & out & (~roof_closed))
    cond_small = ((temp>=65) & (wind>=5) & (~roof_closed))
    edge *= np.where(cond_boost, 1.05, np.where(cond_small, 1.02, 0.985))

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
    hr3 = pick("p_rolling_hr_3","p_hr_count_3"); pa3 = pick("p_rolling_pa_3")
    with np.errstate(invalid='ignore', divide='ignore'):
        hr_rate_short = hr3 / pa3
    ss_shrink = np.minimum(1.0, np.nan_to_num(pa3, nan=0.0) / 30.0)
    cond_hi  = (hr_rate_short >= 0.10) & (pa3 > 0)
    cond_mid = (hr_rate_short >= 0.07) & (hr_rate_short < 0.10) & (pa3 > 0)
    factor *= np.where(cond_hi, (1.12 * (0.5 + 0.5 * ss_shrink)), 1.0)
    factor *= np.where(cond_mid, (1.06 * (0.5 + 0.5 * ss_shrink)), 1.0)

    brl14 = pick("p_barrel_rate_14","p_hard_hit_rate_14")
    brl30 = pick("p_barrel_rate_30","p_hard_hit_rate_30")
    qoc = np.nanmax(np.vstack([brl14.values, brl30.values]), axis=0)
    factor *= np.where(qoc>=0.11, 1.07, np.where(qoc>=0.09, 1.04, 1.0))

    fb14 = pick("p_fb_rate_14","p_fb_rate_7","p_fb_rate")
    gb14 = pick("p_gb_rate_14","p_gb_rate_7","p_gb_rate")
    factor *= np.where(fb14>=0.42, 1.04, np.where(fb14>=0.38, 1.02, 1.0))
    factor *= np.where((~np.isnan(gb14)) & (gb14<=0.40), 1.02, 1.0)

    bb_rate = pick("p_bb_rate_14","p_bb_rate_30","p_bb_rate")
    factor *= np.where(bb_rate>=0.09, 1.02, 1.0)

    xw = pick("p_xwoba_con_14","p_xwoba_con_30","p_xwoba_con")
    factor *= np.where(xw>=0.40, 1.05, np.where(xw>=0.36, 1.03, 1.0))

    ev_allowed = pick("p_avg_exit_velo_14","p_avg_exit_velo_7","p_avg_exit_velo_30")
    factor *= np.where(ev_allowed>=90.0, 1.03, 1.0)

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

def compute_overlay_cols_vectorized(df):
    df = df.copy()
    ratings = rate_weather_vectorized(df)
    overlay = overlay_multiplier_vectorized(df).astype(np.float32)
    weak_p  = weak_pitcher_factor_vectorized(df).astype(np.float32)
    hot_b   = short_term_hot_factor_vectorized(df).astype(np.float32)

    final_raw = np.clip(
        np.clip(overlay, 0.68, 1.44) *
        np.clip(weak_p, 0.90, 1.18) *
        np.clip(hot_b, 0.96, 1.10),
        0.60, 1.65
    ).astype(np.float32)

    roof = df.get("roof_status", pd.Series("", index=df.index)).astype(str).str.lower()
    roof_closed = roof.str.contains("closed|indoor|domed", regex=True)
    has_temp = df["temp"].notna() if "temp" in df.columns else pd.Series(False, index=df.index)
    has_hum  = df["humidity"].notna() if "humidity" in df.columns else pd.Series(False, index=df.index)
    has_wind = df["wind_mph"].notna() if "wind_mph" in df.columns else pd.Series(False, index=df.index)

    conf_base = (has_temp.astype(float) + has_hum.astype(float) + has_wind.astype(float)) / 3.0
    conf_roof = np.where(roof_closed, 0.35, 1.0)
    confidence = np.clip(conf_base * conf_roof, 0.0, 1.0)
    alpha = 0.5 + 0.5 * confidence

    final_mult = (1.0 + alpha * (final_raw - 1.0)).astype(np.float32)

    new_cols = pd.DataFrame({
        "overlay_multiplier": overlay,
        "weak_pitcher_factor": weak_p,
        "hot_streak_factor": hot_b,
        "final_multiplier_raw": final_raw,
        "final_multiplier": final_mult,
    }, index=df.index)

    return pd.concat([df, new_cols, ratings], axis=1)

# ========== Train/Validation via Embargoed Time Splits ==========
    seeds = [42, 101, 202, 404]
    folds = embargo_time_splits(event_df.get("game_date", pd.Series("2000-01-01")), n_splits=2)

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

    # ---------------- Base model OOFs ----------------
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
        st.write(f"Fold {fold+1}/{len(folds)} done in {timedelta(seconds=int(fold_time))}, ETA {timedelta(seconds=int(est_time_left))}")

    # ---------------- Ranker ----------------
    days = pd.to_datetime(event_df.get("game_date", pd.Series("2000-01-01"))).dt.floor("D")
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

    # ---------------- Meta stacker ----------------
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

    # ---------------- Calibration ----------------
    oof_pred_meta = meta.predict_proba(X_meta_s)[:, 1]
    auc_oof = roc_auc_score(y, oof_pred_meta)
    ll_oof  = log_loss(y, oof_pred_meta)
    st.success(f"OOF Meta AUC: {auc_oof:.4f} | LogLoss: {ll_oof:.4f}")

    ir = IsotonicRegression(out_of_bounds="clip")
    y_oof_iso = ir.fit_transform(oof_pred_meta, y.values)
    today_iso = ir.transform(P_today_meta)

    K_adapt = choose_adaptive_K(len(y))
    best_T, K_used, hits_at_K = tune_temperature_for_topk_adaptive(y_oof_iso, y.values, K=K_adapt)
    st.write(f"Adaptive K={K_used} | Best T={best_T:.3f} | Hits@K={hits_at_K}")
    logits_today = logit(np.clip(today_iso, 1e-6, 1-1e-6))
    today_iso_t = expit(logits_today * best_T)

    # ---------------- Final leaderboard ----------------
    def build_leaderboard(df, calibrated_probs, final_score, label="hr_probability_iso_T"):
        df = df.copy()
        df[label] = np.asarray(calibrated_probs)
        df["ranked_probability"] = np.asarray(final_score)
        df = df.sort_values(by=["ranked_probability"], ascending=False).reset_index(drop=True)
        df["hr_base_rank"] = df[label].rank(method="min", ascending=False)
        return df

    leaderboard = build_leaderboard(today_df, today_iso_t, today_iso_t)

    # ---------------- Output ----------------
    top_n = st.sidebar.number_input("Top-N to display", min_value=10, max_value=100, value=30, step=5)
    st.markdown(f"### 🏆 Top {int(top_n)} HR Leaderboard")
    leaderboard_top = leaderboard.head(int(top_n))
    st.dataframe(leaderboard_top, use_container_width=True)

    st.download_button(
        label=f"⬇️ Download Top {int(top_n)} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{int(top_n)}_leaderboard.csv",
        mime="text/csv",
    )
    st.download_button(
        label="⬇️ Download Full Prediction CSV",
        data=leaderboard.to_csv(index=False),
        file_name="today_hr_predictions.csv",
        mime="text/csv",
    )

    gc.collect()
    st.success("✅ HR Prediction pipeline complete")
