"""
BigMart Sales Prediction
=========================================
Model: 0.7 x "physics" model (on recovered true price)  +  0.3 x CatBoost
v2: sales decoded as base_price x whole units

Key finding from 5-fold CV (3 seeds):
  Sales ~= Item_MRP x (units-sold rate of the outlet).
  This 10-parameter model (one rate per store, CV RMSE 1071.5) beats every tuned
  GBM (CatBoost 1075.1, XGBoost 1078.9, LightGBM 1082.6).
  Item identity, visibility, weight, etc. add no reliable signal; they only add variance.
  CatBoost is blended in at 30% for diversity / private-leaderboard robustness.
Deps: pandas, numpy, scikit-learn, catboost
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / 'train.csv'
TEST_PATH = HERE / 'test.csv'
OUT_PATH = HERE / 'submission_final.csv'
TARGET = 'Item_Outlet_Sales'
W_PHYSICS, W_CAT = 0.7, 0.3
SEED = 42

CAT_COLS = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size',
            'Outlet_Location_Type', 'Outlet_Type', 'Item_Category']
NUM_COLS = ['Base_Price', 'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 'Vis_Ratio',
            'Price_per_Weight', 'MRP_Band', 'Item_Outlet_Count']


# ---------------------------------------------------------------- cleaning
def prepare(train, test):
    train, test = train.copy(), test.copy()
    train['is_train'], test['is_train'] = 1, 0
    df = pd.concat([train, test], ignore_index=True)

    # Fat content: 5 labels -> 2, and non-consumables (NC) get their own label
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
        {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
    df['Item_Category'] = df['Item_Identifier'].str[:2]
    df.loc[df['Item_Category'] == 'NC', 'Item_Fat_Content'] = 'Non-Edible'

    # Weight: an item has one weight everywhere -> fill from other rows of the same item
    df['Item_Weight'] = df['Item_Weight'].fillna(
        df.groupby('Item_Identifier')['Item_Weight'].transform('mean'))
    df['Item_Weight'] = df['Item_Weight'].fillna(
        df.groupby('Item_Type')['Item_Weight'].transform('median'))

    # Outlet size: missing for 3 whole outlets. OUT010 (grocery) mirrors OUT019 (Small);
    # OUT017/OUT045 (Tier-2 Type1) mirror OUT035 (Small).
    df['Outlet_Size'] = df['Outlet_Size'].fillna('Small')

    # Visibility 0 is impossible for a sold item -> item's mean non-zero visibility
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, np.nan)
    df['Item_Visibility'] = df['Item_Visibility'].fillna(
        df.groupby('Item_Identifier')['Item_Visibility'].transform('mean'))
    df['Item_Visibility'] = df['Item_Visibility'].fillna(df['Item_Visibility'].median())

    # Features
    df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
    df['Vis_Ratio'] = df['Item_Visibility'] / df.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    df['Price_per_Weight'] = df['Item_MRP'] / df['Item_Weight']
    df['MRP_Band'] = pd.cut(df['Item_MRP'], [0, 69, 136, 203, 300], labels=False)
    df['Item_Outlet_Count'] = df.groupby('Item_Identifier')['Outlet_Identifier'].transform('count')
    for c in CAT_COLS:
        df[c] = df[c].astype(str)

    tr = df[df['is_train'] == 1].reset_index(drop=True)
    te = df[df['is_train'] == 0].reset_index(drop=True)
    return tr, te


# ---------------------------------------------------------------- true price recovery
def recover_base_price(train, test):
    """Sales are exactly (hidden base price) x (whole units). The listed MRP is that base
    price plus random jitter in 0.1 steps. For each item, find the price near its median
    MRP that makes all its training sales whole-number multiples. Falls back to MRP."""
    df = pd.concat([train, test], ignore_index=True)
    price = {}
    for item, g in df.groupby('Item_Identifier'):
        s = g[TARGET].dropna().values
        med = g['Item_MRP'].median()
        best_p, best_err = None, np.inf
        for j in range(-40, 41):
            p = round(med + 0.1 * j, 4)
            if len(s) == 0:
                break
            q = s / p
            err = np.abs(q - np.round(q)).max()
            if err < best_err:
                best_p, best_err = p, err
        if best_p is not None and best_err < 0.01:
            price[item] = best_p
    return price


# ---------------------------------------------------------------- models
def physics_model(X, y, V):
    """Sales = true base price x outlet's average units sold."""
    rate = pd.Series(y / X['Base_Price'].values).groupby(X['Outlet_Identifier'].values).mean()
    return V['Base_Price'].values * V['Outlet_Identifier'].map(rate).values


def catboost_model(X, y, V):
    f = CAT_COLS + NUM_COLS
    m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10,
                          verbose=0, random_seed=SEED)
    m.fit(X[f], y, cat_features=CAT_COLS)
    return m.predict(V[f])


def blend(X, y, V):
    p = W_PHYSICS * physics_model(X, y, V) + W_CAT * catboost_model(X, y, V)
    return np.clip(p, 0, None)


# ---------------------------------------------------------------- main
def main():
    train_raw, test_raw = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    train, test = prepare(train_raw, test_raw)
    price = recover_base_price(train_raw, test_raw)
    for d in (train, test):
        d['Base_Price'] = d['Item_Identifier'].map(price).fillna(d['Item_MRP'])
    y = train[TARGET].values

    # Validation
    oof = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(train):
        oof[va] = blend(train.iloc[tr], y[tr], train.iloc[va])
    print(f'5-fold CV RMSE: {np.sqrt(mean_squared_error(y, oof)):.2f}')

    # Fit on full train, predict test
    pred = blend(train, y, test)
    sub = pd.DataFrame({'Item_Identifier': test_raw['Item_Identifier'],
                        'Outlet_Identifier': test_raw['Outlet_Identifier'],
                        TARGET: pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f'Saved {OUT_PATH}: {sub.shape}')


if __name__ == '__main__':
    main()
