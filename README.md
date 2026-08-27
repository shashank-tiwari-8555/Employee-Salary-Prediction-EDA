# 💼 Employee Salary Prediction & Feature Engineering Pipeline

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) project built as part of the **LearnDepth Academy** Machine Learning Internship. This project identifies key drivers of compensation, handles real-world data noise, engineers predictive interaction metrics, and benchmarks regression algorithms.

-------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌 Project Overview
The objective is to analyze historical employee records, clean messy real-world data anomalies, uncover core factors influencing compensation packages, and build regression models to accurately forecast annual salary (`salary_lakh`).

-------------------------------------------------------------------------------------------------------------------------------------------------------

## 📊 Dataset Summary
- **Source File:** `03_employee_salary.csv`
- **Raw Dimensions:** 1,020 rows × 5 columns
- **Cleaned Dimensions:** 985 rows × 7 columns (including engineered features)
- **Target Variable ($y$):** `salary_lakh` (Annual salary in Lakhs INR)
- **Independent Features ($X$):**
  - `experience_years`: Total professional work experience (years)
  - `age`: Employee age (years)
  - `projects_completed`: Number of technical projects delivered
  - `training_hours`: Total completed professional training hours

-------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧹 Data Quality & Preprocessing Workflow
1. **Duplicate Elimination:** Detected and removed **20 identical duplicate rows**.
2. **Domain-Specific Filtering:**
   - Filtered **5 entries** with negative experience values (`experience_years < 0`).
   - Removed **5 invalid records** where employee age was below the legal working threshold (`age < 18`).
   - Removed **5 erroneous rows** with negative compensation values (`salary_lakh < 0`).
3. **Missing Value Imputation:**
   - Addressed 42 total missing entries across `experience_years`, `age`, `projects_completed`, and `training_hours`.
   - Applied **median imputation** across numerical columns to maintain distribution shape without introducing outlier bias.

-------------------------------------------------------------------------------------------------------------------------------------------------------

## ⚙️ Feature Engineering
To capture complex, non-linear relationships between employee experience and productivity:
- **`experience_score`**: An interaction term defined as:
  $$\text{experience\_score} = \text{experience\_years} \times \text{projects\_completed}$$
- **`projects_per_year`**: A career velocity/productivity rate metric with Laplace smoothing:
  $$\text{projects\_per_year} = \frac{\text{projects\_completed}}{\text{experience\_years} + 1}$$

--------------------------------------------------------------------------------------------------------------------------------------------------------

## 📈 Key Exploratory Findings & Insights
- **Primary Driver:** `experience_years` is the strongest positive linear predictor of compensation ($r \approx 0.90$).
- **Interaction Impact:** `experience_score` shows strong predictive power ($r \approx 0.70$), capturing senior-level contributions effectively.
- **Low Direct Impact:** `age` alone shows near-zero independent linear correlation with compensation once experience is factored in ($r \approx -0.03$).

-------------------------------------------------------------------------------------------------------------------------------------------------------

## 🤖 Model Performance & Evaluation
Models were trained using an **80/20 Train-Test split** with `random_state=42`:

| Model | $R^2$ Score | RMSE (Lakh) | MAE (Lakh) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | **0.8429** | **1.9633** | **1.5415** |
| **Random Forest Regressor** | **0.8111** | **2.1528** | **1.7091** |

> **Conclusion:** The baseline Linear Regression model explains ~84.3% of the variance in employee compensation with an average absolute error of ₹1.54 Lakh/year.

-------------------------------------------------------------------------------------------------------------------------------------------------------

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.12
- **Data Wrangling:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
- **Environment:** JupyterLab / Anaconda

-------------------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/shashank-tiwari-8555/Employee-Salary-Prediction-EDA.git](https://github.com/shashank-tiwari-8555/Employee-Salary-Prediction-EDA.git)
   cd Employee-Salary-Prediction-EDA
