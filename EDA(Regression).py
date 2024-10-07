import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
from sklearn.preprocessing import LabelEncoder

def detailed_analysis(df, target_column=None):
    """
    Perform detailed analysis on the dataset.
    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    target_column (str): Name of the target column for distribution analysis.
    """

    # 1. Basic Information
    print("\nBasic Information:")
    df.info()

    # 2. Descriptive Statistics
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

    # 3. Missing Values Analysis
    print("\nMissing Values:")
    if df.select_dtypes(include=[np.number]).shape[1] <= 10:
        print("\nPair Plot for Numerical Features (Sampled Data):")
        sampled_df = df.sample(100) if len(df) > 1000 else df
        sns.pairplot(sampled_df)
        plt.show()

    # Visualize missing values
    plt.figure(figsize=(10, 6))
    msno.matrix(df)
    plt.show()

    # 4. Data Types
    print("\nData Types:")
    print(df.dtypes)

    # 5. Imputation of Missing Values
    for col in df.columns:
        if df[col].isnull().any():  # Check for missing values
            if df[col].dtype == 'object':  # Categorical column
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"Imputed missing values in {col} with mode: {mode_value}")
            else:  # Numeric column
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
                print(f"Imputed missing values in {col} with mean: {mean_value:.2f}")

    # 6. Encoding Categorical Columns
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Encoded {col} to numeric values.")

    # 7. Correlation Analysis
    if df.select_dtypes(include=[np.number]).shape[1] > 1:
        print("\nCorrelation Analysis (Numerical Features):")
        numeric_df = df.select_dtypes(include=[np.number])

        if not numeric_df.empty:
            corr = numeric_df.corr()
            if corr.isnull().values.any():
                print("Some features have constant values, unable to compute correlation.")
                corr = corr.fillna(0)

            if len(corr.columns) > 1:
                plt.figure(figsize=(16, 12))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
                plt.title('Correlation Matrix')
                plt.show()
            else:
                print("Not enough numerical features for correlation analysis.")
        else:
            print("No numeric columns found for correlation analysis.")

    # 8. Value Counts of Categorical Columns
    print("\nCategorical Value Counts:")
    for col in cat_columns:
        print(f"\nValue Counts for {col}:")
        print(df[col].value_counts())

    # 9. Pair Plot for numerical features
    if df.select_dtypes(include=[np.number]).shape[1] <= 10:
        print("\nPair Plot for Numerical Features:")
        sns.pairplot(df)
        plt.show()

    # 10. Target Distribution
    if target_column and target_column in df.columns:
        print("\nTarget Distribution:")
        sns.countplot(x=target_column, data=df)
        plt.title('Target Distribution')
        plt.show()

    # 11. Data Skewness
    print("\nSkewness of Numerical Features:")
    numeric_skewness = df.select_dtypes(include=[np.number]).skew().sort_values(ascending=False)
    print(numeric_skewness[abs(numeric_skewness) > 0.5])


# Main function to load dataset and call analysis function
def main():
    dataset_path = input("Enter Dataset (CSV file path): ")
    try:
        df = pd.read_csv(dataset_path)

        # Display the first few rows of the dataset
        print("\nFirst few rows of the dataset:")
        print(df.head())

        # Perform detailed analysis on the dataset
        detailed_analysis(df, target_column='target')  # Pass the target column if exists
    except FileNotFoundError:
        print("Error: The specified file was not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        print("Error: No data found in the file. Please check the file content.")
    except pd.errors.ParserError:
        print("Error: Could not parse the file. Please ensure it is a valid CSV file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
