# H&M Personalized Fashion Recommendations: Data Dictionary

## 1. Overview

| Item | Description |
|---|---|
| Source | Kaggle, H&M Personalized Fashion Recommendations competition |
| Content | Purchase history of H&M customers, plus metadata for customers and articles |
| Period | About 2 years of transactions (Sept 2018 to Sept 2020) |
| Task | Predict which articles each customer will buy in the 7 days after the training period ends |
| Granularity | `transactions_train`: one row per purchased item. `customers`: one row per customer. `articles`: one row per article |

### Files

| File | Purpose |
|---|---|
| `transactions_train.csv` | Purchases (the main table) |
| `customers.csv` | Customer metadata |
| `articles.csv` | Article (product) metadata |
| `images/` | Article images, stored in subfolders named by the first 3 digits of `article_id`. Not every article has an image |
| `sample_submission.csv` | Example of the submission format |

### Relationships

```
customers (1) ────< transactions_train >──── (1) articles
        customer_id                article_id
```

Each customer can have many transactions, and each article can appear in many transactions.

---

## 2. Transactions (`transactions_train.csv`)

| Column | Type | Description | Notes |
|---|---|---|---|
| `t_dat` | datetime | Date of purchase | Loaded as a string, so convert with `pd.to_datetime`. Used for time-based train/validation splits |
| `customer_id` | string | Customer who bought the item | Foreign key to `customers`. Anonymized hash |
| `article_id` | int | Article that was bought | Foreign key to `articles` |
| `price` | float | Price paid | Normalized, not the real currency value. Only compare it relative to other prices |
| `sales_channel_id` | int | Where the purchase happened | `1` = in store, `2` = online |

> Duplicate rows are not errors. They mean the same customer bought the same article multiple times on the same day. Don't drop them blindly.

---

## 3. Customers (`customers.csv`)

| Column | Type | Description | Notes |
|---|---|---|---|
| `customer_id` | string | Unique customer identifier | Primary key. Anonymized hash |
| `FN` | float | Flag for whether the customer gets the fashion news newsletter | `1.0` = yes, missing = no. Most values are missing |
| `Active` | float | Flag for whether the customer is active for communication | `1.0` = yes, missing = no. Most values are missing |
| `club_member_status` | category | Membership status in the H&M club | `ACTIVE`, `PRE-CREATE`, `LEFT CLUB` |
| `fashion_news_frequency` | category | How often the customer receives fashion news | `NONE`, `Regularly`, `Monthly` |
| `age` | float | Customer age in years | Has some missing values, and outliers worth checking |
| `postal_code` | string | Customer's postal code | Anonymized hash, so it is only useful for grouping customers |

> For `FN` and `Active`, missing means "no", so fill with `0` rather than treating it as unknown.

---

## 4. Articles (`articles.csv`)

Many columns come as a **code/number + name** pair that describe the same thing. Use the number for modeling and the name for reading and EDA.

### Identification

| Column | Type | Description |
|---|---|---|
| `article_id` | int | Unique ID of one specific article (a product in a specific colour). Primary key |
| `product_code` | int | Code of the product itself. One product can have several articles (e.g. the same shirt in different colours) |
| `prod_name` | string | Name of the product |

### Product type

| Column | Description | Example |
|---|---|---|
| `product_type_no` / `product_type_name` | Specific type of product | Trousers, Dress, Sweater |
| `product_group_name` | Broader group of product types | Garment Upper body, Accessories |

### Appearance

| Column | Description | Example |
|---|---|---|
| `graphical_appearance_no` / `graphical_appearance_name` | Pattern or graphic style | Solid, Stripe, All over pattern |
| `colour_group_code` / `colour_group_name` | Specific colour | Black, Dark Blue |
| `perceived_colour_value_id` / `perceived_colour_value_name` | How light or dark the colour looks | Dark, Light, Dusty Light |
| `perceived_colour_master_id` / `perceived_colour_master_name` | Base colour family | Blue, Black, Beige |

### Store organization (from specific to broad)

| Column | Description | Example |
|---|---|---|
| `department_no` / `department_name` | Department that handles the article | Jersey Basic, Trousers |
| `section_no` / `section_name` | Section in the store | Womens Everyday Basics |
| `garment_group_no` / `garment_group_name` | Group of similar garments | Jersey Fancy, Knitwear |
| `index_code` / `index_name` | Main category index, shown as a letter code | Ladieswear, Menswear, Sportswear |
| `index_group_no` / `index_group_name` | Highest-level category | Ladieswear, Baby/Children, Menswear, Sport, Divided |

### Text

| Column | Type | Description |
|---|---|---|
| `detail_desc` | string | Free-text description of the article. Has a few missing values. Useful for NLP or text embeddings |

---

## 5. Notes

- `article_id` has 10 digits. The first 7 are the `product_code`, and the last 3 are the colour variant.
- To find an image, use `images/<first 3 digits of article_id>/<article_id>.jpg`.
- Check missing values and exact unique values yourself with `df.isna().sum()` and `df[col].value_counts()`. The category values above come from general knowledge of the dataset, so your version may differ slightly.
