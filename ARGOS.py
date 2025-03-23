import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load Transaction Data
def load_data(file_path):
    """
    Reads transaction data from a CSV file and returns a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Basic Data Cleaning
def clean_data(df):
    """
    Cleans the transaction data: fills missing values, removes outliers, and formats data.
    """
    df.dropna(inplace=True)  # Remove missing values
    df['Transaction Amount'] = df['Transaction Amount'].abs()  # Ensure amounts are positive
    return df

# Rule-Based Fraud Detection
def detect_fraud_rules(df):
    """
    Applies simple rule-based fraud detection (e.g., transactions above $10,000 are flagged).
    """
    df['Flagged'] = np.where(df['Transaction Amount'] > 10000, 1, 0)
    return df

# Machine Learning Anomaly Detection (Isolation Forest)
def detect_fraud_ml(df):
    """
    Uses Isolation Forest to identify unusual transactions.
    """
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['Anomaly Score'] = model.fit_predict(df[['Transaction Amount']])
    df['ML_Flagged'] = np.where(df['Anomaly Score'] == -1, 1, 0)
    return df

# Visualization
def plot_transactions(df):
    """
    Plots transactions over time and highlights flagged anomalies.
    """
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')

    sns.lineplot(x=df.index, y=df['Transaction Amount'], label='Transaction Amount', color='blue')
    flagged = df[df['Flagged'] == 1]
    ml_flagged = df[df['ML_Flagged'] == 1]

    if not flagged.empty:
        plt.scatter(flagged.index, flagged['Transaction Amount'], color='red', label='Rule-Based Flags')

    if not ml_flagged.empty:
        plt.scatter(ml_flagged.index, ml_flagged['Transaction Amount'], color='purple', label='ML Flags')

    plt.title('Transaction Amounts Over Time')
    plt.xlabel('Transaction Index')
    plt.ylabel('Amount ($)')
    plt.legend()
    plt.show()

# Main Function
def main():
    file_path = "transactions.csv"  # Replace with your actual file path
    df = load_data(file_path)

    if df is not None:
        df = clean_data(df)
        df = detect_fraud_rules(df)
        df = detect_fraud_ml(df)

        print(df[['Transaction ID', 'Transaction Amount', 'Flagged', 'ML_Flagged']].head(10))
        plot_transactions(df)

if __name__ == "__main__":
    main()
