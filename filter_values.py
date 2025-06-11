import numpy as np
import pandas as pd

#--- Generate dummy data -------------------------------
rng = np.random.RandomState(42)
df = pd.DataFrame({
    'age': rng.randint(18, 65, size=10).astype(float),
    'salary': rng.randint(30_000, 120_000, size=10).astype(float),
    'score': rng.rand(10),
    'dept': rng.choice(['Sales', 'Engineering', 'HR'], size=10)
})

# Introduce missing values at random
df.loc[rng.choice(df.index, size=3), 'age'] = np.nan
df.loc[rng.choice(df.index, size=2), 'salary'] = np.nan
df.loc[rng.choice(df.index, size=3), 'score'] = np.nan
df.loc[rng.choice(df.index, size=2), 'dept'] = None

print("Sample dummy data: \n", df)

#--- Deletion based methods -----------------------------

# Drop any row with >= 1 missing value
df_drop_rows = df.dropna()
print("Rows kept:", len(df_drop_rows))

# Drop columns that have more than 70% missing entries
threshold = int(0.7 * len(df))
df_drop_cols = df.dropna(axis=1, thresh=threshold)
print("Columns kept:", len(df_drop_cols.columns))
print("Sample after dropping rows/columns:\n", df_drop_cols.head())


#--- Simple imputation (mean, median, mode) -------------

from sklearn.impute import SimpleImputer

df_simple = df.copy()

# Numeric columns
df_simple['age'] = SimpleImputer(strategy='mean').fit_transform(df_simple[['age']])
df_simple['salary'] = SimpleImputer(strategy='median').fit_transform(df_simple[['salary']])

# Categorical column
df_simple[['dept']] = SimpleImputer(strategy='most_frequent').fit_transform(df_simple[['dept']])

print("Sample after dropping rows/columns:\n", df_simple.head())

#--- KNN & Regression-based imputation ------------------

from sklearn.experimental import enable_iterative_imputer # noqa: F401
# This import is required for IterativeImputer to work
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression


# KNN imputation
df_knn = df.copy()
knn_imp = KNNImputer(n_neighbors=3)
df_knn[['age', 'salary', 'score']] = knn_imp.fit_transform(df_knn[['age', 'salary', 'score']])
print("Sample after KNN imputation:\n", df_knn.head())

# Regression (iterative) imputation
df_reg = df.copy()
iter_imp = IterativeImputer(
    estimator=LinearRegression(),
    max_iter=5,
    random_state=0
)
df_reg[['age', 'salary', 'score']] = iter_imp.fit_transform(df_reg[['age', 'salary', 'score']])
print("Sample after regression imputation:\n", df_reg.head())


# Multivariate Imputation by Chained Equations (MICE)
df_mice = df.copy()
mice_imp = IterativeImputer(max_iter=10, random_state=42)
df_mice[['age', 'salary', 'score']] = mice_imp.fit_transform(df_mice[['age', 'salary', 'score']])
print("Sample after MICE imputation:\n", df_mice.head())

# Indicator Variable Techniques
df_flag = df.copy()
# For each column, add a flag
for col in ['age', 'salary', 'score', 'dept']:
    df_flag[f'{col}_missing'] = df_flag[col].isnull().astype(int)

# Then impute using mean for numerical cols
imp = SimpleImputer(strategy='mean')
df_flag[['age', 'salary', 'score']] = imp.fit_transform(df_flag[['age', 'salary', 'score']])
df_flag[['dept']] = SimpleImputer(strategy='most_frequent').fit_transform(df_flag[['dept']])
print("Sample after adding flags and imputing:\n", df_flag.head())

# Domain-Specific Rules
df_domain = df.copy()

# Example rule: fill missing salary by dept-level median
dept_medians = df_domain.groupby('dept')['salary'].median()
for dept, med in dept_medians.items():
    df_domain.loc[
        (df_domain['salary'].isnull()) & (df_domain['dept'] == dept),
        'salary'
    ] = med

print("Sample after domain-specific imputation:\n", df_domain.head())