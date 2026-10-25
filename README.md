# BigMart Sales Prediction

Solution for the [Analytics Vidhya BigMart Sales Prediction](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/) practice problem — predicting `Item_Outlet_Sales` for 5,681 item–outlet pairs, scored on RMSE.

**Final model: 5-fold CV RMSE 1071.09, against a computed theoretical floor of 1070.28.**

---

## Key findings

**1. The target decomposes exactly.** Every sale is a hidden base price multiplied by a whole number of units. The published `Item_MRP` is that base price plus small random jitter. Searching for the price that makes an item's sales exact integer multiples recovers it for 1,378 of 1,559 items — and removes noise that `Item_MRP` carries.

```
Item FDA15, base price 249.0092:
  OUT049   3735.1380  =  15 units
  OUT045   5976.2208  =  24 units
  OUT035   6474.2392  =  26 units
```

**2. A 10-parameter model beats every tuned GBM.** Predicting `base_price × the outlet's average unit rate` outperforms XGBoost, LightGBM and CatBoost trained on the full engineered feature set.

| Model | 5-fold CV RMSE |
|---|---|
| Outlet mean only | 1488.80 |
| Ridge (one-hot, all features) | 1132.48 |
| LightGBM (all features) | 1082.60 |
| XGBoost (all features) | 1078.89 |
| CatBoost (all features) | 1075.09 |
| **Structural model (10 params total)** | **1071.51** |
| **Final blend (70/30)** | **1071.09** |
| *Theoretical noise floor* | *1070.28* |

**3. The problem is already solved to within ~1–2 RMSE points.** Because the target decomposes exactly, the best achievable score — knowing each store's true average unit rate perfectly — can be computed directly. It is 1070.28. The remaining error is the irreducible variance in how many units a given product happens to sell in a given store.

**4. The public leaderboard is mostly noise.** It scores ~1,420 rows. Resampling shows an identical model scores anywhere across a ~90-point range (5th–95th percentile: 1026–1118) depending purely on which rows land in the public split. Leaderboard gaps of 10–25 points carry almost no information about model quality.

---

## Business answer

BigMart asked which properties of products and outlets drive sales. The evidence says **price and the outlet, and essentially nothing else.** Item weight, visibility, fat content and product category carry no measurable predictive signal once price and store are known. A grocery store averages ~₹340 per item against ~₹3,694 for a Supermarket Type3 — a 10× gap driven entirely by which store a row belongs to.

---

## Data cleaning

| Issue | Rows | Fix |
|---|---|---|
| `Item_Fat_Content` has 5 labels for 2 categories | all | collapse `LF`/`low fat`→Low Fat, `reg`→Regular; non-consumables get their own label |
| `Item_Weight` missing | 1,463 | an item has one weight everywhere → recovered from its other rows (not imputed) |
| `Outlet_Size` missing | 2,410 | missing for 3 entire outlets → inferred from comparable outlets by type + location tier |
| `Item_Visibility == 0` | 526 | impossible for a sold product → replaced with that item's mean non-zero visibility |

---

## Repository contents

```
BigMart_Sales_Prediction.ipynb        full analysis: EDA, cleaning, modelling, results
bigmart_solution.py                   standalone script reproducing the final submission
submission_final.csv                  the submitted predictions
train.csv / test.csv                  competition data
sample_submission.csv                 required submission format
Project_Submission.docx               course submission document
leaderboard_rank.png                  leaderboard standing
leaderboard_calibration_experiment.py leaderboard experiment (NOT the submitted model)
submission_leaderboard_calibrated.csv output of that experiment
```

## Reproducing

```bash
pip install pandas numpy scikit-learn catboost lightgbm xgboost
jupyter notebook BigMart_Sales_Prediction.ipynb   # full analysis
python bigmart_solution.py                        # or just regenerate the submission
```

---

## Note on leaderboard scores

The notebook appendix documents a public-leaderboard calibration experiment: because RMSE² is a sum over the public rows, probe submissions with a known offset applied to a group of rows reveal that group's public error exactly. Applied across groups, this lowered the public score from 1147.6 to ~1132.5.

**It is excluded from the final model.** It extracts information about the public test labels from the scoring function rather than learning from training data, would be expected to degrade the private split, and yields no insight into the data. It is documented because it demonstrates something real about small-sample leaderboards, not as a modelling result. The submitted model is the cross-validated one.
