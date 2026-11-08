# SIC Final Project — Full Modeling & Retrieval Evaluation

**Source:** `sic-final-project (6).ipynb`. This report extracts the completed modeling/retrieval experiments, their reported numbers, model settings, and the current V9.1→LTR status. It does not invent missing results.

**Dataset snapshot reported in the notebook:** 31,788,324 transactions; 1,371,980 customers; 105,542 articles. The working 10% sample contains 137,198 customers, 3,162,621 sampled transactions, 105,542 articles, and 9,974 copied images.

---

## Executive conclusions

- **Churn:** the strongest completed churn result in the notebook is the expanded Top-15 LightGBM model: Test ROC-AUC **0.6956** and Test AP **0.8246**. Its best reported threshold is 0.52, with F1 0.832544 and accuracy 0.727288.
- **Retrieval:** V9.1 is the strongest completed retrieval version reported here, reaching **51.4950%** candidate coverage and **92.8571%** customer hit rate on the 301-future-item evaluation. V8 reached 25.2492%, V4 21.9269%, V2 20.2658%, and V7 19.6013%.
- **Ranking:** the notebook has a completed V9 LightGBM ranker experiment, but its top-N ranking results are weak: Recall 0 at Top-5/10/20, 1 hit at Top-50, and 2 hits at Top-100. The newer historical V9.1 ranking dataset has been started but was not successfully completed in the supplied notebook; therefore it is not presented as a finished result.
- **Important comparison warning:** V9.3 used a different evaluation state with only 9 future transactions/items, while V9.1/V9.4 used 301 future items. Those numbers should not be compared as if they were the same test.

---

## 1. Data and preprocessing

| Dataset | Rows / size | Reported result |
|---|---|---|
| transactions_train | 31,788,324 rows | 5 columns |
| customers | 1,371,980 rows | 7 columns |
| articles | 105,542 rows | 25 columns |
| 10% customer sample | 137,198 customers | 3,162,621 transactions |
| Cleaned 10% sample | 2,866,749 transactions | 295,872 exact duplicates removed |
| Images | 9,974 copied | 26 missing |

**Missing-value handling:** customer `FN` and `Active` were filled with 0; `club_member_status` and `fashion_news_frequency` with 'Unknown'; article `detail_desc` with 'Unknown'; age median was 32.0 and remaining age missing values were filled with 32.0.

**Outlier/skew results:**

- Customer `purchase_count`: skew 4.5759, kurtosis 42.7149
- `total_spend`: skew 5.5425, kurtosis 64.2437
- `avg_price`: skew 4.3134, kurtosis 48.4930
- Article `purchase_count`: 99th percentile = 313, 99.9th percentile = 855.234

**Customer feature summary (means):** purchase_count 21.0317, unique_articles 19.9385, active_days 6.6580, items_per_active_day 3.1631, repeat_rate 0.03145, total_spend 0.5832, avg_price 0.02879.

---

## 2. Churn modeling — what the experiments did

The churn task predicts whether a customer will churn in a future 28-day horizon. The notebook used temporal snapshots, with train/validation/test dates reported as 2020-06-30, 2020-07-28, and 2020-08-25. The improved feature set contains 26 features; the later selected Top-15 contains the most important behavioral/demographic variables.

**Historical improved snapshots:**

| Snapshot | Rows | Churn rate |
|---|---|---|
| 2020-03-31 | 46,778 | 0.7327 |
| 2020-04-28 | 48,354 | 0.7228 |
| 2020-05-26 | 47,869 | 0.6545 |

Combined historical data = 143,001 rows.

The corrected expanded churn training set contains 196,252 rows, 45 columns, 72,826 unique customers, and churn rate 0.6976.

### 2.1 Completed churn model comparison

| Model | ROC-AUC | PR/AP | Balanced Acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7249 | 0.9566 | 0.5023 | 0.9814 | 0.0055 | 0.0109 |
| Random Forest | 0.7130 | 0.9542 | 0.5329 | 0.9818 | 0.0796 | 0.1472 |
| HistGradientBoosting | 0.7223 | 0.9559 | 0.6567 | 0.9441 | 0.7027 | 0.8057 |
| XGBoost | 0.7193 | 0.9553 | 0.6546 | 0.9437 | 0.6983 | 0.8027 |
| LightGBM | 0.7169 | 0.9547 | 0.6531 | 0.9430 | 0.7043 | 0.8064 |

**Simple interpretation:** the very high churn precision/recall at low thresholds is partly driven by the strong class imbalance: validation churn is 90.33% and non-churn is only 9.67%. Balanced accuracy and the non-churn recall are therefore essential. The tree models substantially improve minority-class detection compared with Logistic Regression at threshold 0.90.

### 2.2 Improved Logistic Regression

Validation: ROC-AUC 0.7258, PR-AUC 0.9568, balanced accuracy 0.6602, precision 0.9475, recall 0.6645, F1 0.7812.
Confusion matrix = `[[3382, 1775], [16170, 32027]]`.

### 2.3 Improved HistGradientBoosting

Validation: ROC-AUC 0.7223, PR-AUC 0.9559, balanced accuracy 0.5123, precision 0.9055, recall 0.9978, F1 0.9494.
At the selected threshold 0.90: precision 0.9441, recall 0.7027, F1 0.8057, balanced accuracy 0.6567, non-churn recall 0.6108.
Confusion matrix = `[[3150, 2007], [14330, 33867]]`.

### 2.4 XGBoost

Validation: ROC-AUC 0.7193, PR-AUC 0.9553, balanced accuracy 0.5162, precision 0.9062, recall 0.9967, F1 0.9493.
At threshold 0.90: precision 0.9437, recall 0.6983, F1 0.8027, balanced accuracy 0.6546, non-churn recall 0.6108.
Confusion matrix = `[[3150, 2007], [14540, 33657]]`.

### 2.5 LightGBM

Validation: ROC-AUC 0.7169, PR-AUC 0.9547, balanced accuracy 0.5171, precision 0.9064, recall 0.9966, F1 0.9493.
At threshold 0.90: precision 0.9430, recall 0.7043, F1 0.8064, balanced accuracy 0.6531, non-churn recall 0.6019.
Confusion matrix = `[[3104, 2053], [14250, 33947]]`.

### 2.6 Top-15 feature selection

| Feature set | N | Val ROC-AUC | Val AP | Test ROC-AUC | Test AP | Best iter |
|---|---|---|---|---|---|---|
| Top_10 | 10 | 0.703064 | 0.827221 | 0.692225 | 0.823268 | 406 |
| Top_15 | 15 | 0.702864 | 0.827075 | 0.692616 | 0.823825 | 280 |
| Top_20 | 20 | 0.702894 | 0.827103 | 0.692508 | 0.823754 | 272 |
| Top_25 | 25 | 0.702745 | 0.826913 | 0.692598 | 0.823643 | 260 |
| Top_30 | 30 | 0.702572 | 0.826754 | 0.692120 | 0.823580 | 247 |
| Top_40 | 40 | 0.702673 | 0.826955 | 0.692282 | 0.823483 | 311 |

**Selected Top-15 features:** `n_active_days`, `age`, `median_price`, `days_since_last`, `std_price`, `channel_1_count`, `fashion_news_frequency`, `repeat_rate`, `club_member_status`, `max_price`, `avg_price`, `total_spend`, `items_per_active_day`, `days_between_last_two`, `spend_last_7d`.

### 2.7 Training-strategy experiment

| Strategy | Best iter | Test ROC-AUC | Test AP | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| baseline | 280 | 0.692616 | 0.823825 | 0.755628 | 0.898622 | 0.820945 | 0.725229 |
| balanced | 210 | 0.692788 | 0.823836 | 0.808028 | 0.629488 | 0.707670 | 0.635457 |
| positive_1.25 | 243 | 0.693046 | 0.824003 | 0.743439 | 0.935263 | 0.828391 | 0.728382 |
| positive_1.50 | 244 | 0.692963 | 0.823921 | 0.733383 | 0.959787 | 0.831448 | 0.727232 |

### 2.8 Expanded Top-15 churn champion

Training rows = 196,252; features = 15; best iteration = 84.
Validation ROC-AUC = 0.6991, Validation AP = 0.8236.
Test ROC-AUC = 0.6956, Test AP = 0.8246.
Compared with the previous best: ROC-AUC improved by +0.0026 and AP by +0.0006.

Best threshold reported = 0.52. At this threshold: precision = 0.730833, recall = 0.967142, F1 = 0.832544, accuracy = 0.727288.

### 2.9 MLP churn model

**Expanded Top-15 MLP:** 196,252 training rows, 15 features, 50 iterations. Validation ROC-AUC 0.6941, AP 0.8206. Test ROC-AUC 0.6894, AP 0.8208, precision 0.7516, recall 0.9037, F1 0.8207, accuracy 0.7232. Confusion matrix = `[[4837, 11289], [3639, 34160]]`.

**Improved MLP with 38 numeric + 2 categorical features:** Validation ROC-AUC 0.6961, AP 0.8205; Test ROC-AUC 0.6864, AP 0.8191, precision 0.7317, recall 0.9613, F1 0.8309, accuracy 0.7257. Confusion matrix = `[[2800, 13326], [1464, 36335]]`.

---

## 3. Recommendation / retrieval experiments

Retrieval experiments generate candidate products before ranking. The main metric used in the notebook is **candidate coverage**: the fraction of unique future purchase items that appear in the candidate pool. **Customer hit rate** measures the fraction of evaluation customers for whom at least one future item was captured.

| Experiment | Method | Coverage | Customer hit rate | Avg candidates/customer |
|---|---|---|---|---|
| Popularity | Global popularity | Recall@10 0.0085; @50 0.0279; @100 0.0434 | — | — |
| Item-Item | Item-based | Recall@10 0.0000; @50 0.0038; @100 0.0071 | — | — |
| Item-Based | Item-based | Recall@10 0.0044; @50 0.0058; @100 0.0122 | — | — |
| Weighted Hybrid | Weighted hybrid | Recall@10 0.0044; @50 0.0153; @100 0.0153 | — | — |
| ALS | Implicit ALS | Recall@10 0.0000; @50 0.0024; @100 0.0104 | — | — |
| Content-Based | TF-IDF | Recall@10 0.0048; @50 0.0169; @100 0.0265 | — | — |
| Personalized Popularity | Personalized popularity | Recall@10 0.0000; @50 0.0027; @100 0.0027 | — | — |
| V2 | Hybrid | 20.2658% | — | 3368.41 |
| V3 | Co-purchase | 13.2890% | 25.8929% | 1237.56 |
| V3 Attribute | Personalized index group | 3.6545% | 8.0357% | 281.16 |
| V4 | Multi-window + score-aware | 21.9269% | 36.6071% | 3440.97 |
| V6 | Hybrid score-aware | 5.3156% | 11.6071% | 464.29 |
| V7 | Hybrid union + source quotas | 19.6013% | 33.9286% | 3087.95 |
| V8 | V4 + personalized attribute | 25.2492% | 41.0714% | 4525.38 |
| V9 | Hybrid + LightGBM ranker | 27.2425% | 48.2143% | 4642.86 |
| **V9.1** | **Strong personalized hybrid** | **51.4950%** | **92.8571%** | **635.69** |
| V9.3 | V9.1 + controlled 2-hop | 33.3333%* | 1.7857%* | 817.21* |
| V9.4 | V9.1-style hybrid + 2-hop | 8.9701%** | — | 730.46** |

\* V9.3 output used an evaluation state with only 9 future transactions/items, so it is not directly comparable with V9.1/V9.4's 301 future items.

\*\* V9.4 reported 301 future items but only 27 captured; the notebook output shown here is internally inconsistent with V9.1 and should be treated as a separate experiment, not a replacement for V9.1.

### 3.1 V5 window contribution

| Window | Source | Future items captured | Coverage | Avg candidates |
|---|---|---|---|---|
| recent_10 (10 items) | item_based | 18 | 5.9801% | 335.9 |
| recent_10 (10 items) | content_based | 15 | 4.9834% | 769.8 |
| recent_10 (10 items) | popularity | 5 | 1.6611% | 98.2 |
| medium_30 (30 items) | item_based | 25 | 8.3056% | 740.1 |
| medium_30 (30 items) | content_based | 28 | 9.3023% | 1814.4 |
| medium_30 (30 items) | popularity | 5 | 1.6611% | 98.2 |
| long_60 (60 items) | item_based | 30 | 9.9668% | 995.6 |
| long_60 (60 items) | content_based | 35 | 11.6279% | 2606.0 |
| long_60 (60 items) | popularity | 5 | 1.6611% | 98.2 |
| recent_10 | attribute | 11 | 3.6545% | 302.8 |
| medium_30 | attribute | 11 | 3.6545% | 302.8 |
| long_60 | attribute | 11 | 3.6545% | 302.8 |

### 3.2 V7 source contribution

- item_based: 28/301 = 9.3023%
- content_based: 35/301 = 11.6279%
- attribute: 2/301 = 0.6645%
- popularity: 3/301 = 0.9967%
- **Union captured = 59/301 = 19.6013%**

### 3.3 V8 source coverage

- item_based: 30/301 = 9.9668%
- content_based: 35/301 = 11.6279%
- attribute: 27/301 = 8.9701%
- popularity: 5/301 = 1.6611%
- **Union = 76/301 = 25.2492%**
- Improvement over V4 = +3.3223 percentage points

### 3.4 V9.1 source coverage

| Source | Future items captured | Coverage | Unique contribution |
|---|---|---|---|
| item_based | 120 / 301 | 39.8671% | 6 unique |
| content_based | 72 / 301 | 23.9203% | 33 unique |
| attribute | 0 / 301 | 0.0000% | 0 unique |
| copurchase | 105 / 301 | 34.8837% | 0 unique |
| popularity | 23 / 301 | 7.6412% | 2 unique |
| **Union** | **155 / 301** | **51.4950%** | 155 captured total |

**V9.1 candidate dataset:** 66,112 rows and 14 columns; positive labels = 2,288; positive rate = 3.4608%.
Evaluation customers = 112; future items = 301; average candidates/customer = 635.69.
Customer-level recall summary: mean recall 0.067262, maximum recall 1.0; customer hit rate = 92.8571%.

---

## 4. V9 ranking experiment

V9 is the completed experiment that actually introduced a LightGBM learning-to-rank stage after candidate generation. The ranking dataset had 520,000 rows, 33 features, 86 positive labels, and 104 groups. The LightGBM ranker stopped at best iteration 1.

| Top-N | Hits | Recall | Precision | Mean customer recall |
|---|---|---|---|---|
| 5 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 10 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 20 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 50 | 1 | 0.003425 | 0.000192 | 0.009615 |
| 100 | 2 | 0.006849 | 0.000192 | 0.019231 |

V9 final candidate coverage = 27.2425%; customer hit rate = 48.2143%; average candidates/customer = 4,642.86.

**Feature importance in the ranker:** led by `price_distance` (7), `attribute_score` (6), `price_similarity` (4), `popularity_rank` (4), `content_score` (3), `source_count` (2), `item_based_hits` (2), and `item_customer_count` (1). Several other features had importance 0.

---

## 5. V9.1 → ranking stage status

The notebook then moved to a historical ranking design using four cutoffs: 2020-04-28, 2020-05-26, 2020-06-23, and 2020-07-21, with a 28-day future horizon and minimum history of 2 purchases.

| Cutoff | History customers | Eligible customers | Future transactions | Future customers |
|---|---|---|---|---|
| 2020-04-28 | 123,518 | 18,624 | 104,542 | 21,674 |
| 2020-05-26 | 126,119 | 24,077 | 139,903 | 27,578 |
| 2020-06-23 | 129,034 | 24,318 | 132,506 | 27,279 |
| 2020-07-21 | 131,520 | 21,395 | 106,757 | 23,837 |

The supplied notebook does **not** contain a successful final historical V9.1 ranking result. The run stopped with a Python `TypeError` because a set was sliced with `items[:TOP_ATTR_ITEMS]`. Therefore, no valid V9.1 historical LightGBM ranker metrics should be claimed yet.

---

## 6. MLflow tracking

The notebook created the MLflow experiment `SIC_Candidate_Retrieval` and logged the V9.1 final retrieval run under `V9.1_Final_Personalized_Hybrid`. Logged artifacts were: `retrieval_v91_candidates.pkl`, `retrieval_v91_customer_metrics.csv`, and `retrieval_v91_source_results.csv`.

The churn experiments also logged to MLflow under `Customer_Churn_Improvement`, including the expanded churn champion.

---

## 7. What each experiment means — simple explanation

| Experiment | Simple meaning | Main result |
|---|---|---|
| Popularity | Recommend the globally most bought products. | Recall@100 = 4.34% |
| Item-based | Recommend products bought near the customer's previous products. | Recall@100 = 1.22% |
| ALS | Learn customer–item latent preferences from interactions. | Recall@100 = 1.04% |
| Content-based | Recommend products similar in text/attributes. | Recall@100 = 2.65% |
| V2 | Combine retrieval sources into one candidate pool. | Coverage = 20.27% |
| V4 | Use several history windows and score-aware retrieval. | Coverage = 21.93% |
| V5 | Measure which history window contributes useful items. | Long-60 content = 11.63% |
| V7 | Union multiple sources with explicit quotas. | Coverage = 19.60% |
| V8 | Add personalized attributes to V4. | Coverage = 25.25% |
| V9 | Add LightGBM ranking after retrieval. | Top-100 recall = 0.6849% |
| V9.1 | Strong personalized hybrid retrieval, no ML ranking. | Coverage = 51.50% |
| Churn models | Predict whether a customer will stop buying in the next 28 days. | Expanded Top-15 LightGBM test AP = 0.8246 |

---

## 8. Recommended next modeling step

Do not train the final LTR model from the failed historical run yet. First rebuild the historical V9.1 candidate dataset successfully, verify that each customer–snapshot group has positives and negatives, then create ranking features without future leakage. After that, train LightGBM LambdaRank and report Recall@5/10/20, NDCG@5/10/20, MAP@5/10/20, plus a direct comparison against the existing V9 ranker.

For the final project narrative, separate the two tasks clearly:

1. **Retrieval** answers: *"Did the candidate pool contain the future item?"*
2. **Ranking** answers: *"Did the model put the relevant candidate near the top?"*

V9.1 currently wins retrieval; V9's ranking stage is not yet strong enough to claim a ranking improvement.
