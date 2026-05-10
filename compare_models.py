import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
PRICE_COLS = [f'p{i}' for i in range(1, 41)]
MAX_SELL = 4000

train = pd.read_csv('train.csv', index_col='ID')
test  = pd.read_csv('test.csv',  index_col='ID')
print(f'Train: {train.shape}  |  Test: {test.shape}')

# ── Shared RSI helper ──────────────────────────────────────────────────────────
def compute_rsi(prices, period=14):
    deltas = np.diff(prices[-period - 1:])
    gain   = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
    loss   = np.mean(-deltas[deltas < 0]) if np.any(deltas < 0) else 0.0
    if loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


# ══════════════════════════════════════════════════════════════════════════════
# PETER'S MODEL  —  33 engineered features, target = p50, 4-model ensemble
# ══════════════════════════════════════════════════════════════════════════════
def peter_features(df):
    p = np.clip(df[PRICE_COLS].values, 1e-6, None)
    n = len(df)
    f = {}
    f['p40'] = p[:, -1];  f['p1'] = p[:, 0]
    f['log_ret_full']    = np.log(p[:, -1] / p[:, 0])
    f['last_change']     = p[:, -1] - p[:, -2]
    f['last_change_pct'] = f['last_change'] / p[:, -2]
    for w in [5, 10, 20]:
        f[f'ret_{w}d'] = np.log(p[:, -1] / p[:, -1 - w])
    f['ma5']  = p[:, -5:].mean(axis=1);  f['ma10'] = p[:, -10:].mean(axis=1)
    f['ma20'] = p[:, -20:].mean(axis=1); f['ma40'] = p.mean(axis=1)
    f['p40_vs_ma5']  = f['p40'] - f['ma5']
    f['p40_vs_ma20'] = f['p40'] - f['ma20']
    f['p40_vs_ma40'] = f['p40'] - f['ma40']
    f['std10'] = p[:, -10:].std(axis=1); f['std20'] = p[:, -20:].std(axis=1)
    f['std40'] = p.std(axis=1)
    lr = np.diff(np.log(p), axis=1)
    f['vol10'] = lr[:, -10:].std(axis=1); f['vol20'] = lr[:, -20:].std(axis=1)
    f['vol40'] = lr.std(axis=1)
    upper = f['ma20'] + 2 * f['std20']; lower = f['ma20'] - 2 * f['std20']
    band  = upper - lower
    f['bb_pos'] = np.where(band > 0, (f['p40'] - lower) / band, 0.5)
    f['hist_min']   = p.min(axis=1); f['hist_max'] = p.max(axis=1)
    f['hist_range'] = f['hist_max'] - f['hist_min']
    f['p40_vs_min'] = f['p40'] - f['hist_min']
    f['p40_vs_max'] = f['hist_max'] - f['p40']
    f['p40_pct_range'] = np.where(f['hist_range'] > 0, f['p40_vs_min'] / f['hist_range'], 0.5)
    f['rsi14'] = np.array([compute_rsi(p[i]) for i in range(n)])
    t = np.arange(40); tn = (t - t.mean()) / t.std(); lp = np.log(p)
    f['trend_slope'] = (lp * tn).mean(axis=1) - lp.mean(axis=1) * tn.mean()
    f['ret_2nd_half']   = np.log(p[:, -1] / p[:, 19])
    f['ret_1st_half']   = np.log(p[:, 19] / p[:, 0])
    f['momentum_accel'] = f['ret_2nd_half'] - f['ret_1st_half']
    out = pd.DataFrame(f, index=df.index)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(out.median(), inplace=True)
    return out


print('\n── Peter model: building features ──')
X_peter_all  = peter_features(train)
X_peter_test = peter_features(test)
y_peter = train['p50']

X_tr, X_val, y_tr, y_val = train_test_split(
    X_peter_all, y_peter, test_size=0.2, random_state=RANDOM_STATE
)
p40_val = train.loc[X_val.index, 'p40']
p50_val = train.loc[X_val.index, 'p50']

lgbm_params = dict(
    objective='regression', metric='rmse', learning_rate=0.05,
    num_leaves=63, min_child_samples=20, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=1.0,
    n_estimators=1000, random_state=RANDOM_STATE, verbose=-1
)
lgbm_m = lgb.LGBMRegressor(**lgbm_params)
lgbm_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

xgb_m = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.05, max_depth=5,
    n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
    verbosity=0, early_stopping_rounds=50
)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

cat_m = CatBoostRegressor(
    iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    random_seed=RANDOM_STATE, eval_metric='RMSE',
    early_stopping_rounds=50, verbose=False
)
cat_m.fit(X_tr, y_tr, eval_set=(X_val, y_val))

hgb_m = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
    min_samples_leaf=20, l2_regularization=1.0,
    random_state=RANDOM_STATE, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=30
)
hgb_m.fit(X_tr, y_tr)

val_preds = (lgbm_m.predict(X_val) + xgb_m.predict(X_val) +
             cat_m.predict(X_val) + hgb_m.predict(X_val)) / 4
K_val = int(0.4 * len(X_val))
gains_val = p40_val.values - val_preds
ranked = np.argsort(gains_val)[::-1]
sell_v = np.zeros(len(val_preds), dtype=int)
sell_v[ranked[:K_val]] = 1
peter_val_R = float((sell_v * (p40_val.values - p50_val.values)).sum())
print(f'Peter validation R: {peter_val_R:.2f}')

# Retrain on full data
lgbm_f = lgb.LGBMRegressor(**{**lgbm_params, 'n_estimators': lgbm_m.best_iteration_})
lgbm_f.fit(X_peter_all, y_peter)
xgb_f = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.05, max_depth=5,
    n_estimators=xgb_m.best_iteration, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, verbosity=0
)
xgb_f.fit(X_peter_all, y_peter)
cat_f = CatBoostRegressor(
    iterations=cat_m.best_iteration_, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, random_seed=RANDOM_STATE, verbose=False
)
cat_f.fit(X_peter_all, y_peter)
hgb_f = HistGradientBoostingRegressor(
    max_iter=hgb_m.n_iter_, learning_rate=0.05, max_leaf_nodes=63,
    min_samples_leaf=20, l2_regularization=1.0, random_state=RANDOM_STATE
)
hgb_f.fit(X_peter_all, y_peter)

peter_test_pred = (lgbm_f.predict(X_peter_test) + xgb_f.predict(X_peter_test) +
                   cat_f.predict(X_peter_test) + hgb_f.predict(X_peter_test)) / 4
peter_gain_test = test['p40'].values - peter_test_pred
peter_sell = np.zeros(len(test), dtype=int)
peter_sell[np.argsort(peter_gain_test)[::-1][:MAX_SELL]] = 1
print(f'Peter sells: {peter_sell.sum()}')

peter_sub = pd.DataFrame({'ID': test.index, 'sell': peter_sell})
peter_sub.to_csv('peter_submission.csv', index=False)
print('Saved peter_submission.csv')


# ══════════════════════════════════════════════════════════════════════════════
# PHUC'S MODEL  —  73 features (raw + engineered), target = gain, XGB+CAT, CV
# ══════════════════════════════════════════════════════════════════════════════
def phuc_features(df):
    p = np.clip(df[PRICE_COLS].values, 1e-6, None)
    lr = np.diff(np.log(p), axis=1)
    f = {}
    for i, col in enumerate(PRICE_COLS):
        f[col] = p[:, i]
    f['log_ret_full']    = np.log(p[:, -1] / p[:, 0])
    f['last_change']     = p[:, -1] - p[:, -2]
    f['last_change_pct'] = f['last_change'] / p[:, -2]
    for w in [5, 10, 20]:
        f[f'ret_{w}d'] = np.log(p[:, -1] / p[:, -1 - w])
    f['ma5']  = p[:, -5:].mean(axis=1);  f['ma10'] = p[:, -10:].mean(axis=1)
    f['ma20'] = p[:, -20:].mean(axis=1); f['ma40'] = p.mean(axis=1)
    f['p40_vs_ma5']  = p[:, -1] - f['ma5']
    f['p40_vs_ma20'] = p[:, -1] - f['ma20']
    f['p40_vs_ma40'] = p[:, -1] - f['ma40']
    f['std10'] = p[:, -10:].std(axis=1); f['std20'] = p[:, -20:].std(axis=1)
    f['std40'] = p.std(axis=1)
    f['vol10'] = lr[:, -10:].std(axis=1); f['vol20'] = lr[:, -20:].std(axis=1)
    f['vol40'] = lr.std(axis=1)
    upper20 = f['ma20'] + 2 * f['std20']; lower20 = f['ma20'] - 2 * f['std20']
    bw = upper20 - lower20
    f['bb_pos'] = np.where(bw > 0, (p[:, -1] - lower20) / bw, 0.5)
    f['hist_min']      = p.min(axis=1); f['hist_max'] = p.max(axis=1)
    f['hist_range']    = f['hist_max'] - f['hist_min']
    f['p40_vs_min']    = p[:, -1] - f['hist_min']
    f['p40_vs_max']    = f['hist_max'] - p[:, -1]
    f['p40_pct_range'] = np.where(f['hist_range'] > 0, f['p40_vs_min'] / f['hist_range'], 0.5)
    t = np.arange(40); t_norm = (t - t.mean()) / t.std(); lp = np.log(p)
    f['trend_slope'] = (lp * t_norm).mean(axis=1) - lp.mean(axis=1) * t_norm.mean()
    f['ret_2nd_half']   = np.log(p[:, -1] / p[:, 19])
    f['ret_1st_half']   = np.log(p[:, 19] / p[:, 0])
    f['momentum_accel'] = f['ret_2nd_half'] - f['ret_1st_half']
    out = pd.DataFrame(f, index=df.index)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(out.median(), inplace=True)
    return out


print('\n── Phuc model: building features ──')
X_phuc_all  = phuc_features(train)
X_phuc_test = phuc_features(test)
y_phuc = train['p40'] - train['p50']

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
PENALTY = 100
total_R = 0; xgb_iters = []; cat_iters = []

print(f"{'Fold':>4}  {'Sells':>6}  {'Fold R':>10}")
print("-" * 25)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_phuc_all), start=1):
    X_tr_f, X_val_f = X_phuc_all.iloc[tr_idx], X_phuc_all.iloc[val_idx]
    y_tr_f, y_val_f = y_phuc.iloc[tr_idx],     y_phuc.iloc[val_idx]

    xgb_fold = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, early_stopping_rounds=50,
    )
    xgb_fold.fit(X_tr_f, y_tr_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    xgb_iters.append(xgb_fold.best_iteration)

    cat_fold = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6,
        l2_leaf_reg=3, random_seed=42, early_stopping_rounds=50, verbose=False,
    )
    cat_fold.fit(X_tr_f, y_tr_f, eval_set=(X_val_f, y_val_f))
    cat_iters.append(cat_fold.best_iteration_)

    gain_pred = (xgb_fold.predict(X_val_f) + cat_fold.predict(X_val_f)) / 2
    sigma     = X_val_f['std10'].values
    scores    = gain_pred / (sigma + PENALTY)
    K_fold    = int(0.4 * len(val_idx))
    sell_fold = np.zeros(len(val_idx), dtype=int)
    sell_fold[np.argsort(scores)[::-1][:K_fold]] = 1

    p40_f = train['p40'].iloc[val_idx].values
    p50_f = train['p50'].iloc[val_idx].values
    fold_R = float((sell_fold * (p40_f - p50_f)).sum())
    total_R += fold_R
    print(f"{fold:>4}  {sell_fold.sum():>6}  {fold_R:>10.2f}")

print("-" * 25)
print(f"Phuc CV R: {total_R:.2f}")

# Retrain Phuc on full data
xgb_best = int(np.mean(xgb_iters)); cat_best = int(np.mean(cat_iters))
xgb_final_p = xgb.XGBRegressor(
    n_estimators=xgb_best, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
)
xgb_final_p.fit(X_phuc_all, y_phuc)
cat_final_p = CatBoostRegressor(
    iterations=cat_best, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, random_seed=42, verbose=False,
)
cat_final_p.fit(X_phuc_all, y_phuc)

gain_test_p   = (xgb_final_p.predict(X_phuc_test) + cat_final_p.predict(X_phuc_test)) / 2
sigma_test_p  = X_phuc_test['std10'].values
scores_test_p = gain_test_p / (sigma_test_p + PENALTY)
phuc_sell = np.zeros(len(test), dtype=int)
phuc_sell[np.argsort(scores_test_p)[::-1][:MAX_SELL]] = 1
print(f'Phuc sells: {phuc_sell.sum()}')

phuc_sub = pd.DataFrame({'ID': test.index, 'sell': phuc_sell})
phuc_sub.to_csv('phuc_submission.csv', index=False)
print('Saved phuc_submission.csv')


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 55)
print('SUBMISSION COMPARISON')
print('=' * 55)

peter_ids = set(peter_sub[peter_sub['sell'] == 1]['ID'])
phuc_ids  = set(phuc_sub[phuc_sub['sell']  == 1]['ID'])

both_sell  = peter_ids & phuc_ids
only_peter = peter_ids - phuc_ids
only_phuc  = phuc_ids  - peter_ids
neither    = set(test.index) - (peter_ids | phuc_ids)

print(f"Both models sell:        {len(both_sell):>5}  ({len(both_sell)/MAX_SELL:.1%} of sells)")
print(f"Only Peter sells:        {len(only_peter):>5}  ({len(only_peter)/MAX_SELL:.1%} of sells)")
print(f"Only Phuc sells:         {len(only_phuc):>5}  ({len(only_phuc)/MAX_SELL:.1%} of sells)")
print(f"Neither sells:           {len(neither):>5}")
print(f"\nAgreement (Jaccard):     {len(both_sell) / len(peter_ids | phuc_ids):.3f}")
print(f"Peter val R:             {peter_val_R:.2f}")
print(f"Phuc   CV R:             {total_R:.2f}")

# Save full comparison
comp = pd.DataFrame({'ID': test.index})
comp['peter_sell'] = peter_sell
comp['phuc_sell']  = phuc_sell
comp['agreement']  = (comp['peter_sell'] == comp['phuc_sell']).astype(int)
comp['both_sell']  = ((comp['peter_sell'] == 1) & (comp['phuc_sell'] == 1)).astype(int)
comp.to_csv('model_comparison.csv', index=False)
print('\nSaved model_comparison.csv')
print('\nAgreement breakdown:')
print(comp.groupby(['peter_sell', 'phuc_sell']).size().rename('count'))
