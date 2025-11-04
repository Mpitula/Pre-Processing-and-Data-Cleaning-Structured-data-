import pandas as pd

## Load the dataset
df = pd.read_csv("train_and_test2.csv")

# Display first few rows
print(df.head())

# Check dataset info
print(df.info())


## Check for missing values
print("\nMissing values in 'Age' column before handling:")
print(f"Age: {df['Age'].isnull().sum()}")

print("Missing values in 'Cabin' column before handling:")
if 'Cabin' in df.columns:
    print("Cabin:", df['Cabin'].isnull().sum())
else:
    print("No Cabin column found in the dataset")

## Handle missing values
# For Age: impute with median
if df['Age'].isnull().sum() > 0:
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    print("\nMissing values in 'Age' column after imputation:")
    print(f"Age: {df['Age'].isnull().sum()}")

# For Cabin: drop the column
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)
    print("\nCabin column has been dropped from the dataset")

## Remove duplicates
df.drop_duplicates(inplace=True)

# Confirm
print("\nNumber of duplicates remaining:", df.duplicated().sum())
