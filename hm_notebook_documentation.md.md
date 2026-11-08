# H&M Personalized Fashion Recommendation & Customer Intelligence: Notebook Documentation

*Source file: `sic-hm-finalproject.ipynb` · Type: descriptive documentation only (no code is written or changed, and no code improvements are proposed) · Cell numbers are 0-based indices in the .ipynb file.*

## 1. Overview

**Purpose.** The notebook is an end-to-end data-science project built on the Kaggle *H&M Personalized Fashion Recommendations* dataset (transactions, customers and articles tables). It contains two components:

- **Customer churn prediction**: a binary classifier that estimates whether a customer makes no purchase in the following 28 days.
- **Personalized recommendation**: a two-stage system (candidate retrieval, then learning-to-rank) that produces a Top-K article list per customer.

**Problem addressed.** Which customers are likely to be inactive, and which articles should be shown to each customer. Both are framed as temporal prediction problems: features are built from history up to a cutoff date, and labels come from a later period.

**Scale and environment.** 191 cells (165 code, 26 markdown) with outputs embedded. It was run on Kaggle (hard-coded `/kaggle/...` paths); outputs show a Tesla T4 GPU and about 31 GiB of RAM. All modeling uses a reproducible 10% random customer sample.

**Nature of the notebook.** It is an experiment log rather than a clean pipeline. It contains repeated or superseded cells (cells 95 and 96 are identical; Top-50 generation appears in cell 137 and again in a `FIXED` version in cell 138), the label `Experiment 10` is used for two different experiments (cells 112 and 134), and there are diagnostic cells (139-148, 172-187) and an endless keep-alive loop (cell 189).

**Headline results as reported by the notebook** (see Section 7 for how to read them):

- **Churn** (final test snapshot 2020-07-13 to 2020-08-10, 130,967 customers, threshold 0.82): ROC-AUC 0.8146, average precision 0.9511, balanced accuracy 0.7405, accuracy 0.6999. The always-predict-churn baseline has accuracy 0.8299.
- **Ranking**: mean NDCG over K = 5, 10, 20, 50 on the 222 validation groups ranges from about 0.52 (LightGBM LambdaRank) to 0.97 (hard-negative MLP, cell 112), depending on the experiment. The model carried into the final pipeline (`Pairwise_Deep_B`) scores about 0.736 on those groups.
- **Retrieval**: the initial three-source generator reaches Recall@150 of 0.051 on previously unpurchased items (0.106 if re-purchases count). Later retrieval experiments report Recall@500 between 0.048 and 0.183, depending on how ground truth is defined.

## 2. Notebook Structure

- **Cell 0**: title, project overview, modeling principles and intended pipeline (raw data → ... → FastAPI/Streamlit deployment).
- **Cells 1-4**: project setup, paths, global configuration, raw-file availability check.
- **Cells 5-18**: dataset loading, validation checks, 10% customer sampling, sample saved to parquet.
- **Cells 19-43**: exploratory data analysis (14 analyses plus summary).
- **Cells 44-49**: temporal split, leakage assertions, customer-overlap audit, eligible customers.
- **Cells 50-59**: leakage-safe feature engineering (customer, item, merged table).
- **Cells 60-66**: candidate generation (history, co-occurrence, popularity).
- **Cells 67-78**: candidate recall, source distribution and coverage evaluation.
- **Cells 79-87**: ranking dataset, labels, negative sampling, ranking features, matrix, customer-level split.
- **Cells 88-99**: ranking Experiments 1-3 (LightGBM baseline with MLflow, tuned LightGBM, multi-model benchmark).
- **Cells 100-114**: ranking Experiments 4-12 (objective comparison, ensembles, hyper-parameter search, CatBoost PairLogit, MLP/Transformer rankers, hard-negative pairwise MLPs).
- **Cells 115-118**: Experiments 13-14 (multi-source retrieval; retrieval followed by deep re-ranking).
- **Cells 119-132**: standalone retrieval Experiments 1-9 (popularity, history, co-occurrence, feature similarity, hybrid, personalized re-ranking variants).
- **Cells 133-138**: final two-stage pipeline: pairwise deep ranker benchmark, leakage/result audit and selection, final training, Top-50 generation.
- **Cells 139-142**: diagnostics (feature schema, candidate-ID check and mapping, memory).
- **Cells 143-171**: churn modeling (setup, baselines, temporal snapshots, Experiments 2-9, error analysis, final test evaluation).
- **Cells 172-188**: deployment artifact inspection and bundling into a zip.
- **Cells 189-190**: keep-alive loop and an empty cell.

## 3. Section-by-Section Description

### 3.1 Imports and setup (cells 1-4)

- Imports `os`, `gc`, `json`, `random`, `pathlib`, `collections`, `numpy`, `pandas`; later cells import `matplotlib`, `lightgbm`, `xgboost`, `catboost`, `sklearn`, `torch`, `mlflow`, `tqdm` where needed.
- Seed 42 for `random` and `numpy`; all warnings are suppressed.
- `DATA_DIR` points to the Kaggle competition folder; `WORK_DIR` is `/kaggle/working/sic_hm_project` with `artifacts`, `models`, `reports`, `cache` sub-folders.
- `CONFIG` holds: sample fraction 0.10, churn lookback 90 days, churn horizon 28 days, `retrieval_k` 100, `final_k` 12, `evaluation_k` \[5, 10, 12, 20, 50, 100\]. The final pipeline does not use `retrieval_k` or `final_k` (it uses 500 candidates and Top-50).
- Cell 4 raises an error if `transactions_train.csv`, `customers.csv` or `articles.csv` is missing.

### 3.2 Data loading (cells 5-18)

- **Tables**: customers 1,371,980 × 7; articles 105,542 × 25; transactions 31,788,324 rows (5 columns loaded with compact dtypes; about 3.9 GB in memory).
- **Checks**: duplicate rows (2,974,905 in transactions, none elsewhere), schema and missingness table, date range 2018-09-20 to 2020-09-22 (734 days), referential-integrity assertions (no unknown customers or articles), null checks, price statistics.
- **Sampling**: 10% of customers (137,198) drawn with seed 42, giving 3,162,621 transactions; all 105,542 articles are kept. Integrity assertions are repeated on the sample.
- **Working copies** are sorted by customer and date and saved as three parquet files; a final checklist asserts uniqueness, valid dates, valid keys and positive prices.

### 3.3 Data audit / EDA (cells 19-43)

- **Activity over time**: daily and monthly transaction volume (monthly peaks in June; September 2020 is a partial month).
- **Customers**: mean 23.2 transactions (median 9), mean 19.9 unique articles (median 8), mean spend 0.646 (median 0.246); histograms of each.
- **Articles**: top-20 popularity, product-group distribution (Garment Upper body 1,247,317 transactions; several groups under 100), department table.
- **Price**: percentile table (median 0.0254, 99th percentile 0.1005, maximum 0.507).
- **Sales channel**: channel 1 has 944,579 and channel 2 has 2,218,042 transactions in the sample.
- **Demographics**: age (mean 36.4, range 16-99), club status (ACTIVE 127,257; PRE-CREATE 9,283; LEFT CLUB 52; missing 606), fashion-news frequency (NONE 87,974; Regularly 47,534; Monthly 69; missing 1,621).
- **Structure**: interaction sparsity 99.973%; the top 1%, 5% and 10% of articles account for 16.7%, 40.1% and 55.9% of purchases.

### 3.4 Data preprocessing and feature engineering (cells 49-59, 79-87)

- **Eligible customers**: those with at least one training-period purchase in the last 90 days before the training cutoff (53,669).
- **Leakage guard**: `validate_history` raises an error if the history frame contains dates after the cutoff; all feature builders take `(history_df, cutoff_date)`.
- **Customer features** (23 columns): frequency, unique articles, first/last purchase, total spend, average price, recency, tenure, purchase velocity, average transaction value, and transaction/unique-article/spend counts over 7, 30, 90 and 180 days.
- **Item features** (15 columns): frequency, unique customers, first/last purchase, average price, recency, and transaction/customer counts over the same windows.
- **Ranking dataset** (cells 79-87): candidate pairs are labelled 1 if the article was bought in the validation week. Customer-item features (purchase count, recency, average price, has-history flag), log transforms, price ratios and a source code are added. After dropping non-numeric and constant columns the ranker uses 55 numeric features. Missing values are filled with 0.
- **Ranking-aware feature expansion** (Experiments 9-14): global transforms and within-group statistics (z-score, percentile, min/max, relative-to-center), followed by variance and correlation filters. Feature counts: 55 → 1,053 → 996 → 394 (Experiment 10); 55 → 614 → 557 → 220 (Experiment 11); 55 → 448 → 443 → 85 (Experiment 12); 55 → 716 → 654 → 215 (Experiment 14).
- **Churn features** (cells 149-170) are built separately from transaction history at each snapshot: a 15-feature baseline (frequency, unique articles, spend, average price, recency, tenure, and 7/30/90-day activity) plus ratio and behavioral features (for example `recency_to_tenure`, `frequency_per_tenure`, `spend_30d_vs_90d`). Feature-set sizes tested: 15, 18, 23, 31, 32, 38, 46, 47, 58. The final model uses 23 core features.

### 3.5 Data splitting (cells 44-49, 86-87, 95, 155-156)

- **Global temporal split** (cell 45): training ≤ 2020-09-08 (3,112,497 transactions); validation 2020-09-09 to 2020-09-15 (25,482); test 2020-09-16 to 2020-09-22 (24,642). Assertions confirm strict ordering. Customer overlap is audited (train 135,179; validation 7,125; test 7,003 customers).
- **Ranking split**: the 1,109 customers with at least one positive are split 80/20 at customer level with `train_test_split` (seed 42): 887 training and 222 validation groups (36,960 and 9,269 rows in the final version, cell 95). An assertion checks zero customer overlap.
- **Churn split**: three 28-day snapshots: train 2020-05-18 → 2020-06-15, validation 2020-06-15 → 2020-07-13, test 2020-07-13 → 2020-08-10. An earlier exploratory run (cells 149-154) uses snapshot 2020-08-11 with a stratified random 80/20 split.
- The global test window of cell 45 is created and checked but is not used afterwards for modeling or evaluation.

### 3.6 Modeling (cells 60-138, 144-171)

**Family A, candidate retrieval.** Initial generator (cells 60-66): last 20 history items, up to 50 co-occurrence neighbors per history item, top-100 global popularity, merged to 150 candidates. Standalone retrieval experiments (cells 119-132) test popularity, history, co-occurrence, cosine feature similarity, weighted hybrids, and four personalized re-ranking variants with 500 candidates per customer. The final retrieval configuration (Experiment 9) blends history 0.30, attribute 0.30, co-purchase 0.15, popularity 0.10, recency 0.15 (30-day half-life).

**Family B, ranking.**

- Gradient-boosted trees: LightGBM LambdaRank and XE-NDCG, XGBoost `rank:ndcg`, CatBoost YetiRank and PairLogit.
- Neural: MLP rankers, a Transformer ranker, pairwise MLPs with hard negatives, and a listwise variant.
- Ensembles: simple average, NDCG-weighted, rank averaging.
- Progression of the best mean NDCG reported: Experiment 3 LightGBM 0.545; 4 LightGBM LambdaRank Top-10 0.523; 6 CatBoost PairLogit 0.543; 7 CatBoost PairLogit Deep 0.557; 9 MLP Ranker Deep 0.598; 10 HardNegative MLP 0.974; 11 0.727; 12 0.929; 14 (retrieved candidates) 0.281. Cell 134 then reports the three-network pairwise benchmark: `Pairwise_Deep_A` 0.737, `B` 0.736, `C` 0.735, ensemble 0.736.
- Final network `Pairwise_Deep_B`: MLP 55 → 256 → 128 → 64 → 1 with batch normalization and dropout 0.2 (56,321 parameters), trained 35 epochs, batch size 512, learning rate 2e-4, on 43,121 hard-negative pairs.

**Family C, churn.** LightGBM (several configurations), CatBoost, XGBoost, and an MLP (plain and class-weighted loss). A `DummyClassifier` (most frequent class) and an all-churn rule serve as baselines.

### 3.7 Experiment tracking

- **MLflow** is installed in cell 89 and used only in ranking Experiments 1-2 (experiment names `H&M_Ranking_LightGBM` and `H&M_Ranking_LightGBM_Tuned`); it logs parameters, metrics, models and feature lists, and prints run IDs. Four LightGBM configurations (`LGBM_Tuned_1`-`4`) are tracked.
- **All other experiments** are tracked by files written to `/kaggle/working`: `experiment_N_results.csv`, `*_model_ranking.csv`, `*_best_per_k.csv`, `churn_expN_config.json`, `retrieval_expN_*.pkl`, and so on. There is no experiment registry beyond file names.

### 3.8 Model evaluation

- **Retrieval**: Recall@K (K = 20-150 in cells 67-78; up to 500 in later experiments), HitRate, Precision, NDCG, MAP, candidate coverage, source contribution, retrieval time.
- **Ranking**: Recall, Precision, HitRate, NDCG and MAP at K = 5, 10, 20, 50, computed per customer group (customers without a positive are skipped). "Mean" metrics average across the four K values. Model choice uses mean NDCG, with MAP as tie-break.
- **Churn**: ROC-AUC, average precision, accuracy, balanced accuracy, macro F1, class-wise precision/recall, confusion matrix, threshold sweeps (cells 154, 158, 161).

### 3.9 Explainability

There is no SHAP or LIME. Explainability is limited to LightGBM gain-based feature importance for churn (`recency_to_tenure` dominates in every experiment, followed by `unique_articles` and `recency_days`) and importance-driven feature selection in ranking Experiment 9. The neural rankers have no explanation output.

### 3.10 Final model selection and threshold analysis

- **Ranking**: cell 135 combines metrics into a `Final_Score` and selects `Pairwise_Deep_B`; "final K" is set to 50.
- **Churn**: thresholds from 0.30 to 0.90 are scanned on validation (balanced accuracy, macro F1, class-wise recall). The selected model is Experiment 8 `D_core_shallow` (LightGBM, 23 features, 374 iterations) with threshold 0.82, applied to the test snapshot without retraining. Test result: ROC-AUC 0.8146, average precision 0.9511, accuracy 0.6999, balanced accuracy 0.7405, macro F1 0.6330; churn precision 0.9436, churn recall 0.6790; non-churn precision 0.3387, recall 0.8020; predicted churn rate 0.597 versus an actual rate of 0.830.

### 3.11 Model saving / artifacts

Ranking: `LGBM_Tuned_*_model.txt` and feature lists, `experiment_10/12/14` checkpoints (`.pth`) with scalers and selectors, `final_pairwise_deep_B.pt`. Churn: LightGBM text models, a CatBoost `.cbm`, an MLP state with preprocessing arrays, configs and results. Retrieval: candidate pickles and configs. Cells 172-188 assemble a `hm_deployment` folder (churn model, config, feature importance, results, test report, `manifest.json`) and a `hm_deployment_bundle.zip` (161 files, 20.78 MB).

### 3.12 Additional sections

- **Leakage and integrity checks**: assertions on referential integrity and temporal ordering (cells 12-18, 47), the `validate_history` guard (cell 54), customer-overlap assertions in ranking splits (cells 87, 95), a distribution and result audit for the final ranker (cell 135).
- **Error analysis**: churn Experiment 8 (cell 169): false positives are 23.3% of non-churners and false negatives 29.1% of churners, with mean features compared across correct, false-positive and false-negative groups.
- **Diagnostics**: candidate-ID validation (cell 140) shows saved candidate lists hold row positions of the articles table rather than `article_id` values; cell 141 maps them back by position.
- **Housekeeping**: artifact searches, reading the Kaggle notebook source file (cells 178-187), zip creation, and an infinite 5-minute keep-alive loop (cell 189).

## 4. Key Outputs

- **Sampled data**: `transactions_sample.parquet`, `customers_sample.parquet`, `articles_sample.parquet` (in `sample_data/`); `train_churn.parquet`, `valid_churn.parquet`, `test_churn.parquet`.
- **Result tables**: `experiment_1` to `experiment_14` result, ranking and best-per-K CSVs; `retrieval_exp1` to `retrieval_exp9` results, coverage and comparison CSVs; `churn_exp2` to `churn_exp9` results and configs; `experiment_10_final_model_selection.csv` and `experiment_10_final_leakage_audit.csv`.
- **Ranking models**: `LGBM_Tuned_1`-`4` models and feature lists; `experiment_10_best_model.pth`, `experiment_12_best_model.pth`, `experiment_14_models.pth` (three seeds: 42, 123, 2026) with scalers, variance selectors and correlation-feature arrays; `final_pairwise_deep_B.pt` and its summary CSV.
- **Churn models**: LightGBM text models (`churn_lgbm_exp2/3/5`, `churn_exp6/7/8` models, `churn_final_model.txt`), CatBoost `churn_exp4_best.cbm`, MLP `churn_exp9_mlp_state.pt` with `churn_exp9_preprocessing.npz`.
- **Recommendations**: `final_top50_pairwise_deep_B.csv` and `final_top50_pairwise_deep_B_FIXED.csv` (222 customers × 50 = 11,100 rows).
- **Churn evaluation**: `churn_final_test_predictions.csv`, `churn_final_test_report.json`, `churn_threshold.json`.
- **Retrieval artifacts**: candidate pickles for each retrieval experiment, `snap_test_exp6.pkl`.
- **Deployment**: `hm_deployment/` folder and `hm_deployment_bundle.zip` (161 files, 20.78 MB).
- **Tracking**: an `mlruns/` directory (MLflow runs for Experiments 1-2).
- **Figures (displayed in the notebook only)**: EDA histograms and trends (cells 22-40), recall curves (cells 72, 78).

## 5. Dependencies

No versions are pinned or printed anywhere in the notebook, there is no requirements file, and the only install command is `%pip install -q mlflow` (cell 89). The libraries imported are:

- **Core**: `numpy`, `pandas`, `matplotlib`, `tqdm`, `joblib`, `IPython`, plus standard library modules (`os`, `gc`, `json`, `pickle`, `pathlib`, `collections`, `zipfile`, `glob`, `shutil`, `subprocess`, `importlib`, and others).
- **Modeling**: `scikit-learn`, `lightgbm`, `xgboost`, `catboost`.
- **Deep learning**: `torch` (CUDA used when available).
- **Tracking**: `mlflow`.
- **Implicit requirement**: a parquet engine (`pyarrow` or `fastparquet`), since parquet files are written and read.
- **Runtime**: notebook metadata lists Python 3.13 and kernel `python3`.


## 6. Execution Notes

- **External data**: the Kaggle competition files `transactions_train.csv`, `customers.csv`, `articles.csv` under `/kaggle/input/competitions/h-and-m-personalized-fashion-recommendations`. Article images are not used.
- **Memory**: the full transactions table is about 3.9 GB in memory before sampling; later cells are written to be memory-conscious (the environment showed about 31 GiB total, 17 GiB in use). Several cells delete large objects and call garbage collection.
- **GPU**: a Tesla T4 is used by the pairwise MLP cells (134, 136, 138); GBDT experiments run on CPU.
- **Timing as printed**: LightGBM ranker fits take under 1 s; CatBoost YetiRank 3.5 s; each pairwise MLP about 14.6 s; Experiment 9 retrieval 16.9 s for 222 customers; Experiment 13 retrieval 81 s. Total runtime is not recorded.
- **Order dependence**: later cells read variables created earlier (`train_tx`, `ranking_train`, `X_train`, `groups_valid`, and others) and files written to `/kaggle/working`; some cells search globals by name and fall back to reconstruction (cell 128). The notebook should be run top to bottom in one session.
- **Kaggle-specific**: cells 178-187 read `/kaggle/working/.virtual_documents/__notebook_source__.ipynb`; they cannot work elsewhere.
- **Cell 189** is an endless loop that prints a heartbeat every five minutes and must be interrupted manually.
- **Reproducibility**: seed 42 is set for the sample, splits and models, but GPU training and warning suppression (`warnings.filterwarnings('ignore')`) are not controlled for.

## 7. Limitations

This section relies only on what the notebook contains or lacks. Each item states the limitation, its potential impact, and whether the notebook addresses it (no remedies are proposed).

### 7.1 Data Limitations

- **D1. Sample size and sufficiency**
  - *Limitation:* Only 10% of customers are used (137,198 of 1,371,980). The ranker is trained and validated on 1,109 customers (887 and 222) with 1,450 positive pairs (1,159 and 291).
  - *Impact:* Small validation sets give high variance; differences of a few hundredths in NDCG may be noise, and findings may not transfer to the full customer base.
  - *Addressed?* Partly. The sample is seeded and integrity-checked, but there are no confidence intervals, repeated samples or full-data run.
- **D2. Class imbalance**
  - *Limitation:* Ranking has 1,450 positives among 8,034,536 candidate pairs (0.018%, about 5,540:1). In churn, 79-84% of customers are labelled churn depending on the snapshot.
  - *Impact:* Accuracy is misleading (the all-churn baseline scores 0.83). Training on sampled negatives (positive rate 3.14% instead of 0.018%) means model scores are not probabilities for the natural population.
  - *Addressed?* Partly. Negative sampling (about 30.9:1), balanced or positive-weighted churn runs and balanced-accuracy thresholds are used; the effect of sampling on score scale is not examined.
- **D3. Only customers with labels are modeled**
  - *Limitation:* Only 1,109 of 53,669 eligible customers (2.1%) have a validation positive inside their candidate list, and only these form ranker training and validation groups. 5,256 eligible customers actually bought in the validation week.
  - *Impact:* Customers whose purchases the retriever missed, or who bought nothing, never reach the ranker, so metrics describe an optimistic sub-population.
  - *Addressed?* No. It is a deliberate choice stated in cell 81 but not quantified further.
- **D4. Data quality**
  - *Limitation:* 2,974,905 duplicate transaction rows (9.4%) are retained; `FN` and `Active` are 65-66% missing; age is missing for 1.2% (maximum 99); 9,520 sample transactions have product group `Unknown`; prices are anonymized and scaled (maximum 0.507, 99th percentile 0.1005).
  - *Impact:* Duplicates may be genuine repeat purchases, but they inflate counts, frequency and popularity features. Missing demographic fields are harmless only because they are not used as features.
  - *Addressed?* Partly. Integrity checks and assertions exist; duplicates and outliers are neither removed nor analyzed.
- **D5. Coverage and cold start**
  - *Limitation:* Candidates cover 63.85% of the training catalog (54,013 of 84,598 items). Item features exist only for items bought before the cutoff (84,598 of 105,542 articles). Only customers with a purchase in the last 90 days are eligible (53,669 of 135,179). The top 10% of articles account for 55.9% of purchases.
  - *Impact:* New and long-tail articles and inactive or new customers get little or no coverage; popularity bias is likely.
  - *Addressed?* No. Coverage is measured (cell 75) but there is no cold-start or diversity mechanism.
- **D6. Provenance**
  - *Limitation:* The data are the public Kaggle competition files. The notebook does not document collection, anonymization (hashed IDs, scaled prices) or representativeness. Article images and text descriptions are unused.
  - *Impact:* Source reliability and bias cannot be assessed from the notebook, and potentially informative content signals are left out.
  - *Addressed?* No.
- **D7. Temporal coverage and drift**
  - *Limitation:* Data end on 2020-09-22. The ranker is labelled on one week (2020-09-09 to 2020-09-15); churn uses three 28-day snapshots between May and August 2020. Monthly volumes show strong swings.
  - *Impact:* Results describe one narrow period and data that are about six years old relative to October 2026; behavior and assortment may have changed.
  - *Addressed?* No drift analysis and no multi-week or multi-season validation.
- **D8. Privacy and licensing**
  - *Limitation:* Licensing and privacy are not discussed. The deployment bundle contains `snap_test_exp6.pkl` (transaction rows of 222 customers) and per-customer recommendations keyed by hashed customer IDs.
  - *Impact:* Redistribution terms of the competition data and re-identification risk are not assessed.
  - *Addressed?* No.

### 7.2 Methodological Limitations

- **M1. One labelled week serves training, tuning and reporting**
  - *Limitation:* Ranking labels come only from the validation week. The ranker trains on 80% of those customers; the other 222 groups drive early stopping, the comparison of 14 experiments and the final reported numbers.
  - *Impact:* Selection bias; the reported validation scores are optimistic and are not an unbiased estimate of future performance.
  - *Addressed?* Partly. The customer-level split avoids customer overlap; there are no rolling windows, nested validation or untouched test period for the recommender.
- **M2. Zero-filled features for unseen pairs**
  - *Limitation:* 27,919 of 46,229 sampled pairs (60.4%) had no row in the pair-feature table and all their customer and item features were set to 0; only 39.6% have customer-item history. The final Top-50 builder (cell 138) assembles features from separate lookup tables, and cell 135 reports a mean absolute feature shift of 0.396 (maximum mean shift 4.08) between train and validation.
  - *Impact:* Training rows for new-item candidates carry placeholder zeros, and the `has_customer_item_history` flag and source code offer a shortcut (positive rate 0.089% for history candidates versus 0.011% for co-occurrence). Training and inference feature distributions may differ.
  - *Addressed?* A duplicate-merge problem was fixed (cell 84); the zero-fill and train/inference consistency are not resolved.
- **M3. Sampled groups differ from serving groups**
  - *Limitation:* Training and validation groups hold 28-48 candidates (all positives plus sampled negatives, quotas 20/25/5 by source), whereas retrieval returns 500 per customer. Group-wise engineered features (Experiments 9-14) are computed inside each group.
  - *Impact:* Metrics are not comparable to ranking a full candidate list. Recall@50 equals 1.0 trivially. Mean NDCG is 0.93-0.97 in Experiments 10 and 12 but 0.28 when the same approach ranks retrieved candidates (Experiment 14).
  - *Addressed?* The gap is visible but not explained, and the final model is re-evaluated on sampled groups (cell 134), not on retrieved candidates.
- **M4. Inconsistent ground truth across retrieval experiments**
  - *Limitation:* Retrieval Experiment 6 keeps only transactions after the cutoff (1,368 transactions, 1,163 positive pairs). Experiments 7-9 use `snap_test`, which cell 128 builds from all transactions of the 222 customers regardless of date (22,060 rows, 17,443 unique pairs, mean 78.6 per customer), although a code comment calls it future positives. Retrieval Experiments 1-5 use the 291 ranking positives.
  - *Impact:* HitRate@5 jumps from 0.09 (Experiment 6) to 0.68-0.72 (Experiments 7-9) while recall falls, consistent with past purchases counting as hits. These experiments are not comparable and Experiments 7-9 are probably inflated.
  - *Addressed?* No.
- **M5. Validation customers recovered by assumption**
  - *Limitation:* Retrieval Experiments 6-9 and the final Top-50 use 222 customers recovered as the last 222 entries of `ranking_train` (cell 128; its comment states the ordering is assumed preserved). The real split used random `train_test_split`, and explicit training IDs were unavailable so the overlap check was skipped.
  - *Impact:* These customers may include ranker training customers, so some evaluation could be on training data.
  - *Addressed?* No.
- **M6. Candidate generation is basic and results conflict**
  - *Limitation:* Only three sources. Implicit collaborative filtering is named in cell 60 and `als_items` is configured in cell 61, but nothing is implemented. The weighted hybrid (Mean NDCG 0.0009) scores below its components (history 0.0117). Experiment 13 reports Recall = HitRate = 1.0 at K = 50-500, while standalone retrieval on the same groups peaks at Recall 0.155 (K = 500). Saved candidate lists contain table row positions rather than article IDs.
  - *Impact:* First-stage recall is low (Recall@150 of 0.051 for unseen items) and caps any ranker; the contradictory figures cast doubt on Experiment 13.
  - *Addressed?* Partly. Coverage and source analysis were added; the discrepancy is not reconciled.
- **M7. Feature selection**
  - *Limitation:* The ranker uses only transaction-derived features. Not used: age, club status, news frequency, sales channel, article attributes (colour, type, department), text, images, season, item similarity. Article attributes appear only in retrieval scoring (Experiments 6-9).
  - *Impact:* Little signal for new items; plausibly related to the plateau of churn AUC around 0.813 across 20+ configurations.
  - *Addressed?* No.
- **M8. Model selection**
  - *Limitation:* Only tree ensembles and MLP/Transformer variants were tried. Not tried: sequence or session models, two-tower or graph recommenders, matrix factorization, survival-style churn models, logistic regression for churn, a plain 'repurchase recent items' ranking baseline.
  - *Impact:* It is unknown whether the chosen models are competitive with strong standard approaches.
  - *Addressed?* Partly. More than three families are compared, within a narrow set.
- **M9. Tuning scope**
  - *Limitation:* Small manual grids (four LightGBM and eight CatBoost configurations per experiment), no automated search, no cross-validation, mostly single seeds. Every LightGBM LambdaRank run stops at best iteration 1 (at iteration 25 training NDCG@10 is 0.628 against 0.479 validation). Churn variants differ by less than 0.002 AUC.
  - *Impact:* Tuning had little measurable effect and may mostly fit noise; repeated selection on one validation set inflates the winner's score.
  - *Addressed?* Partly (early stopping, regularization variants). The principle in cell 91 of not tuning repeatedly on validation is not followed in practice.
- **M10. Churn target and population**
  - *Limitation:* Churn means no purchase within 28 days, for every customer with any earlier history, including long-inactive ones (median recency 150 days). The same customers appear in the train, validation and test snapshots.
  - *Impact:* The label mixes real attrition with ordinary purchase gaps (churn rate 79-84%); correlated rows across snapshots can flatter temporal validation; threshold transfer across windows is uncertain.
  - *Addressed?* No.
- **M11. Preprocessing assumptions**
  - *Limitation:* Missing numeric values are filled with 0; co-occurrence uses each customer's last 30 unique items without order or time; popularity uses all-time counts; history candidates are the 20 most recent purchases; prices are used as given.
  - *Impact:* Zero imputation conflates missing and zero; popularity may favor items that are no longer sold.
  - *Addressed?* No.
- **M12. Test-set handling**
  - *Limitation:* The churn test snapshot was scored in cells 159 and 160 (headed FINAL TEMPORAL TEST RESULTS) before later experiments whose headers say test data was never loaded. The exploratory random-split run (cell 153) uses its evaluation split for early stopping. The global test window of Section 3 is never used.
  - *Impact:* Earlier exposure weakens the claim of an untouched test set.
  - *Addressed?* Partly. Later selection uses validation only, and the final test run (cell 171) does not retune.
- **M13. Explainability reliability**
  - *Limitation:* There is no SHAP or LIME. Gain importance for churn is a global, split-count-sensitive measure that is unstable under correlated features (many churn features are ratios of one another). Neural rankers have no explanation.
  - *Impact:* No per-customer or per-recommendation reasons; the dominance of `recency_to_tenure` cannot be verified by another method.
  - *Addressed?* No.

### 7.3 Model Limitations

- **Mo1. Train/validation gap and weak learning signal**
  - *Limitation:* Every LightGBM LambdaRank run stops at best iteration 1; at iteration 25 training NDCG@10 is 0.628 against 0.479 on validation (cell 90). The final pairwise MLP reaches a training loss of 0.00036 after 35 epochs; no train-versus-validation gap is reported for it.
  - *Impact:* The tree rankers add almost nothing beyond a single tree, and the neural ranker may fit its training pairs almost perfectly.
  - *Addressed?* Partly (early stopping and dropout/regularization variants); the gap is not analyzed.
- **Mo2. Performance cannot be summarized by one number**
  - *Limitation:* Reported ranker NDCG ranges from 0.28 to 0.97 across experiments and protocols. The selected `Pairwise_Deep_B` scores about 0.736 on 28-48-candidate groups. The final Top-50 file is generated for the 222 customers but cell 138 contains no metric computation against ground truth.
  - *Impact:* The end-to-end quality of the final recommender is not measured anywhere in the notebook.
  - *Addressed?* No.
- **Mo3. Rare and borderline cases**
  - *Limitation:* Only 4 of 1,450 positive pairs come from the popularity source; customers have a median of 9 transactions. For churn at threshold 0.82, non-churn precision is 0.339, with 34,890 missed churners and 4,412 false alarms; the predicted churn rate is 59.7% against an actual 83.0%.
  - *Impact:* Behavior for sparse customers, niche articles and the minority class is poorly characterized.
  - *Addressed?* Partly (one churn error analysis, cell 169); nothing for the recommender.
- **Mo4. Sensitivity to noise and outliers**
  - *Limitation:* No perturbation or robustness tests. Activity is heavy-tailed (maximum 1,188 transactions for one customer in the churn table) and price outliers are kept; only log transforms and clipping at 0 are applied.
  - *Impact:* Behavior under noisy or shifted inputs is unknown.
  - *Addressed?* No.
- **Mo5. Generalization**
  - *Limitation:* The ranker is validated on one week and never on a later period. Churn performance is stable across snapshots (AUC 0.811 on validation, 0.815 on test) but every model and feature variant plateaus near 0.813.
  - *Impact:* Churn generalization looks reasonable within the sampled customers and months; ranker generalization to other weeks, customers or seasons is unknown.
  - *Addressed?* Partly for churn (temporal validation and test); no for ranking.
- **Mo6. Interpretability**
  - *Limitation:* The selected ranker is a neural network (a black box here); the churn model is a tree ensemble explained only by gain importance.
  - *Impact:* Recommendations cannot be justified to business users, and influential features cannot be audited.
  - *Addressed?* No.
- **Mo7. Calibration**
  - *Limitation:* There is no calibration analysis (no reliability curves, Brier score or calibrator). Scores come from negative-sampled ranking data and from balanced or weighted churn runs. Selected churn thresholds range from 0.49 to 0.83 across experiments.
  - *Impact:* Scores are not usable as probabilities; thresholds do not transfer reliably, as the gap between predicted and actual churn rate shows.
  - *Addressed?* No (thresholds are tuned instead).
- **Mo8. Tied scores in final output**
  - *Limitation:* Printed previews of the final Top-50 (cells 137, 138) show runs of identical scores (for example 0.629148 and 0.696777) across several items of one customer.
  - *Impact:* The order among tied items is arbitrary, so the ranked list is partly unordered.
  - *Addressed?* No.
- **Mo9. Behavior under missing history**
  - *Limitation:* There is no multimodal input. Missing customer-item or item history is handled by zero-fill; customers without recent activity or new articles are excluded from the pipeline by design.
  - *Impact:* Performance for new customers and new items is not defined.
  - *Addressed?* No.


### 7.4 Evaluation Limitations

- **E1. No independent test for the recommender**
  - *Limitation:* The ranker is trained, tuned, selected and reported on data from one week; the test window (2020-09-16 to 2020-09-22) is created but never used. Retrieval evaluation uses the same 222 customers, which may overlap with training (M5). For churn, a temporal test exists but shares customers with training snapshots and was viewed early (M12).
  - *Impact:* Reported ranking metrics cannot be taken as held-out performance.
  - *Addressed?* No for ranking; partly for churn.
- **E2. Metric adequacy for decisions**
  - *Limitation:* Ranking metrics are computed on tiny sampled groups (K = 50 exceeds the group size, so Recall@50 is 1.0), customers without positives are skipped, and MAP@12 (`final_k` in `CONFIG`) is not reported. No revenue, conversion, diversity or novelty measure exists. Churn accuracy at the final threshold (0.70) is below the always-churn baseline (0.83), and no cost model weighs the 34,890 missed churners against the 4,412 false alarms.
  - *Impact:* The metrics do not support business decisions such as campaign targeting or placement of recommendations.
  - *Addressed?* Partly (balanced accuracy and macro F1 are added for churn).
- **E3. Error analysis**
  - *Limitation:* Churn has one error analysis (Experiment 8). The recommender has none beyond a per-source positive-rate table.
  - *Impact:* Failure patterns of the recommender (by source, category or customer activity) are unknown.
  - *Addressed?* Partly.
- **E4. Subgroup analysis**
  - *Limitation:* No breakdown by age, club status, sales channel, activity level or product group, even though these fields are loaded and profiled.
  - *Impact:* Uneven performance and possible fairness issues across segments are undetected.
  - *Addressed?* No.
- **E5. Statistical uncertainty**
  - *Limitation:* Single runs with no confidence intervals, bootstrap or repeated splits; model winners are chosen on gaps as small as 0.001-0.02.
  - *Impact:* Rankings between near-tied models are not statistically supported.
  - *Addressed?* No.
- **E6. Scope of the leakage audit**
  - *Limitation:* Date-ordering assertions, a history guard (it checks only that the latest date is within the cutoff) and customer-overlap assertions exist, and cell 135 audits feature distributions and result files. There is no audit of the ground-truth definition or the recovered validation customers in retrieval Experiments 6-9.
  - *Impact:* The most consequential leakage-like issues (M4, M5) pass unchecked.
  - *Addressed?* Partly.
- **E7. Reporting consistency**
  - *Limitation:* Narrative cells quote figures typed by hand: cell 91 gives baseline NDCG@10 as 0.5456 while the rerun in cell 98 shows 0.5564; row counts differ between reruns (37,000 / 9,229 in cell 87 against 36,960 / 9,269 in cell 95); cell 115 names `HardNegative_MLP_3` (NDCG about 0.974) as the next-stage ranker, whereas the final pipeline uses `Pairwise_Deep_B` (about 0.736); the label `Experiment 10` is reused.
  - *Impact:* A reader can mistake superseded figures for final results.
  - *Addressed?* No.

## 8. Conclusion

The notebook builds a complete experimental workflow on a 10% sample of the H&M data: leakage-aware temporal feature construction, a three-source candidate generator, many ranking experiments (LightGBM, XGBoost, CatBoost, MLP, Transformer, pairwise hard-negative networks), and a churn classifier family. Its final churn model (LightGBM, 23 features, threshold 0.82) is evaluated on a temporal test snapshot with ROC-AUC 0.8146. The final recommender (`Pairwise_Deep_B` over 500 retrieved candidates, Top-50 output) is produced but not evaluated against ground truth.

The main limitations are: a recommender evaluated on a single labelled week with no independent test; sampled ranking groups whose scores do not carry over to full candidate lists; inconsistent ground-truth definitions and an assumed set of validation customers in later retrieval experiments; zero-filled features for most candidate pairs; no calibration, subgroup or uncertainty analysis. Deployment and operational limitations are documented separately in Deployment.md 
