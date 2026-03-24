# 🍔 Food Ordering Behavior Analysis
## Predicting High-Value Customers for a Food Delivery Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Business Problem

A food delivery platform operating across five major Indian cities wants to **identify and predict high-value customers** before they churn. High-value customers (top 25% by lifetime spend) disproportionately drive revenue. This project builds a machine learning pipeline that scores every user and enables targeted marketing, personalised retention strategies, and revenue optimisation.

**Business Impact:**
| Metric | Potential Uplift |
|---|---|
| Retention rate (HV customers) | +15–25% |
| AOV via targeted upsell | +10–20% |
| Marketing CAC reduction | −20–30% |

---

## 📁 Project Structure

```
food-ordering-behavior/
│
├── data/
│   └── food_ordering_behavior_dataset.csv   # 50,000 orders · 4,000 users
│
├── notebooks/
│   └── food_behavior_analysis.ipynb         # Full analysis notebook
│
├── README.md
└── requirements.txt
```

---

## 📊 Dataset Description

| Column | Type | Description |
|---|---|---|
| `order_id` | int | Unique order identifier |
| `user_id` | int | Unique customer identifier (4,000 users) |
| `age` | int | Customer age |
| `city` | str | Mumbai, Delhi, Bangalore, Pune, Chandigarh |
| `order_time` | str | Morning / Afternoon / Evening / Night |
| `day_type` | str | Weekday / Weekend |
| `cuisine` | str | Chinese, Biryani, Fast Food, South Indian, Desserts |
| `meal_type` | str | Breakfast, Lunch, Snacks, Dinner |
| `restaurant_type` | str | Budget / Mid-range / Premium |
| `order_value` | int | Order amount in ₹ (100–999) |
| `discount_applied` | str | Yes / No |
| `delivery_fee` | int | Delivery charge in ₹ |
| `time_taken_to_order` | int | Minutes to complete checkout |
| `rating_given` | int | Customer rating (1–5) |
| `is_repeat_order` | str | Yes / No |
| `mood` | str | Happy / Lazy / Celebrating / Bored / Stressed |
| `hunger_level` | str | Low / Medium / High |
| `company` | str | Alone / Partner / Friends / Family |
| `rainy_weather` | str | Yes / No |

**Size:** 50,000 rows · 19 columns · No missing values

---

## 🎯 Target Variable

> **`high_value`** — Binary label: `1` if the user's total lifetime spend ≥ 75th percentile (₹8,266), else `0`.

- Class 0 (Normal): 2,998 users (75%)
- Class 1 (High-Value): 1,002 users (25%)

---

## 🔬 Approach & Methodology

### 1. Exploratory Data Analysis
- Univariate: distributions of order value, age, restaurant type
- Bivariate: order value vs time-of-day, age, mood, discount usage
- Multivariate: heatmap of time × day type, correlation matrix
- Behavioral insights: peak slots, cuisine preferences, repeat order patterns

### 2. Feature Engineering (User-Level)
Order-level data aggregated to **user-level behavioral features**:
- `total_orders` — order frequency
- `avg_order_value` — spending level
- `repeat_rate` — loyalty signal
- `discount_rate` — price sensitivity
- `night_rate` — late-night ordering propensity
- `weekend_rate` — weekend engagement
- `avg_rating` — satisfaction proxy
- `rainy_order_rate` — convenience demand
- Categorical mode features: top city, cuisine, restaurant, mood, company

### 3. Machine Learning Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 98.12% | 96.02% | 96.50% | 96.26% | 0.9989 |
| Random Forest | 98.25% | 98.95% | 94.00% | 96.41% | 0.9984 |
| **Gradient Boosting** | **99.50%** | **99.00%** | **99.00%** | **99.00%** | **0.9997** |
| Extra Trees | 95.87% | 98.27% | 85.00% | 91.15% | 0.9942 |

**Winner:** Gradient Boosting Classifier

### 4. Hyperparameter Tuning
GridSearchCV with StratifiedKFold (3-fold) on Random Forest across:
- `n_estimators`: [100, 200]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5]

---

## 🔑 Key Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | total_orders | 0.608 |
| 2 | avg_order_value | 0.181 |
| 3 | night_rate | 0.023 |
| 4 | discount_rate | 0.022 |
| 5 | avg_rating | 0.021 |
| 6 | repeat_rate | 0.020 |

---

## 💡 Key Business Insights

1. **Volume is the #1 predictor** — `total_orders` accounts for 61% of model variance. Frequency strategies (loyalty tiers, streak rewards) have the highest ROI.
2. **Night orders = premium spend** — Night-time slot shows the highest AOV variance; targeted 9–11 PM campaigns can lift revenue ~15–20%.
3. **Discount paradox** — High-value customers use *fewer* discounts. Blanket discounting trains customers to expect price cuts without delivering loyalty.
4. **Satisfaction → Spend** — `avg_rating` in top 10 features; delivery experience quality is directly linked to LTV.
5. **Context-aware selling** — Mood + company signals can power real-time premium placements (Celebrating + Friends → surface shareable platters).

---

## 🎯 Actionable Recommendations

### Marketing
- Deploy HV score weekly to marketing stack; segment Gold / Silver / Bronze
- Night-time premium campaigns (Fri/Sat 9–11 PM) targeting users with high `night_rate`
- Mood-based triggers: weekend + group orders → auto-surface 'Party Pack' bundles

### Retention
- Churn early-warning: users with stagnant `total_orders` growth → proactive outreach
- 10-order/month loyalty milestone → free delivery + exclusive restaurant access
- Prioritise customer support SLA improvements for Gold (high-spend, high-rated) users

### Revenue Growth
- Up-sell premium restaurant types to mid-tier users with already-high AOV
- Replace blanket discounts with personalised cashback (₹50 off ₹500+)
- Expand premium restaurant supply in Mumbai & Delhi (highest AOV cities)

---

## 🔮 Future Improvements

- Add **recency** feature → full RFM model
- Incorporate **session/app data** (abandoned carts, browse time)
- Deploy as **real-time scoring API** (FastAPI + MLflow)
- Explore **LightGBM / CatBoost** for additional accuracy gains
- Build **CLV regression model** (predict exact future spend)
- A/B test recommendation strategies against model predictions

---

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/your-username/food-ordering-behavior.git
cd food-ordering-behavior

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/food_behavior_analysis.ipynb
```

---

## 📦 Tech Stack

- **Python 3.9+**
- **pandas** — data manipulation
- **numpy** — numerical computing
- **matplotlib + seaborn** — visualisation
- **scikit-learn** — ML models, preprocessing, evaluation

---

## 👤 Author: Krishanu Santra

**Senior Data Scientist** | Consumer Analytics & Food-Tech
Portfolio project demonstrating end-to-end data science: EDA → Feature Engineering → ML → Business Recommendations

---

## 📄 License

This project is licensed under the MIT License.
