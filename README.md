# Task 25 – Basic Data Cleaning (Titanic Dataset)

**Veda Technology · AI & ML Track · Level 1 · Day 25**

## Objective
Combine fundamental data-cleaning techniques (missing-value handling, duplicate removal and data-type correction) into one practical workflow.

## Tools Used
- Python
- Pandas
- Seaborn (to load the dataset)
- Jupyter Notebook

## Dataset
Titanic dataset (`sns.load_dataset("titanic")`) – 891 rows, 15 columns. The untouched copy is saved as `titanic_original.csv`.

## Files in this Repository
| File | Description |
|------|-------------|
| `titanic_original.csv` | Original dataset before any cleaning |
| `titanic_cleaned.csv` | Dataset after all cleaning steps |
| `data_cleaning.ipynb` | Notebook with the complete cleaning workflow |
| `README.md` | Description of every cleaning step and decision |

## Cleaning Workflow

### Step 1 – Inspect the Dataset
Before changing anything, the dataset was inspected using `df.shape`, `df.info()`, `df.isnull().sum()` and `df.duplicated().sum()`.

| Column | Missing Count | Missing % |
|--------|---------------|-----------|
| age | 177 | ~19.9% |
| embarked | 2 | ~0.2% |
| embark_town | 2 | ~0.2% |
| deck | 688 | ~77.2% |

Duplicate rows found: **<N>** *(fill in from your notebook output)*

### Step 2 – Handle Missing Values
```python
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
df = df.drop(columns=["deck"])
```
- **age → median:** Age is skewed and has outliers, so the median is a more stable fill value than the mean.
- **embarked / embark_town → mode:** Categorical columns with only 2 missing values, so the most frequent value adds negligible bias.
- **deck → dropped:** About 77% of values were missing, so imputing would be guesswork.

### Step 3 – Remove Duplicates
```python
df = df.drop_duplicates().reset_index(drop=True)
```
Exact duplicate rows were removed so that no record is counted twice during analysis or model training.

**Note:** The Titanic dataset has no passenger ID column, so some "duplicates" may actually be different passengers with identical values (same class, fare, age, etc.). In a real project, such rows should be verified against a unique identifier before deletion. Here, `drop_duplicates()` was applied as the standard technique for this task.

### Step 4 – Correct Data Types
```python
df["age"] = df["age"].astype(int)
for col in ["sex", "embarked", "embark_town", "class", "who", "pclass"]:
    df[col] = df[col].astype("category")
df["survived"] = df["survived"].astype(bool)
```
- **age → int:** After imputation, ages are whole numbers, so float is unnecessary.
- **Categorical columns → category:** Repeated labels; saves memory and marks them as categorical features.
- **survived → bool:** It is a yes/no flag.

### Step 5 – Verify and Save
After cleaning, `df.info()` confirmed the new types, and the checks `df.isnull().sum().sum()` and `df.duplicated().sum()` returned **0** for both. The result was saved as `titanic_cleaned.csv`.

## Before vs After

| Check | Original | Cleaned |
|-------|----------|---------|
| Rows | 891 | <N> *(fill in)* |
| Columns | 15 | 14 |
| Missing values | 4 columns affected | 0 |
| Duplicate rows | <N> | 0 |

## Key Learnings
- Always inspect the data first, then clean. Every decision should be documented.
- The order matters: handle missing values first, because imputation can change which rows count as duplicates.
- Choose methods by column type and amount of missing data, not by habit.

## Interview Questions – Key Answers
1. **What is data cleaning?** The process of fixing errors, missing values, duplicates and wrong data types so the data is reliable for analysis.
2. **Why is data cleaning important before machine learning?** Models learn from the data they are given. Dirty data lowers accuracy, can bias predictions, and some algorithms fail on missing values or text numbers.
3. **What problems can poor-quality data cause?** Wrong predictions, biased models, runtime errors, wasted time and poor decisions based on incorrect insights.

## How to Run
```bash
pip install pandas seaborn jupyter
jupyter notebook data_cleaning.ipynb
```
