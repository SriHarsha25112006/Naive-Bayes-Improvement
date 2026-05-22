import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import warnings
from time import time

from icnb import InverseCorrelationNB

warnings.filterwarnings("ignore")

# List of 20 diverse OpenML datasets
DATASETS = [
    'diabetes', 'spambase', 'vehicle', 'blood-transfusion-service-center',
    'phoneme', 'qsar-biodeg', 'wdbc', 'kc2', 'kc1', 'pc1',
    'pc4', 'pc3', 'jm1', 'madelon', 'ilpd', 'credit-g', 
    'tic-tac-toe', 'sick', 'hypothyroid', 'kr-vs-kp'
]

def evaluate_20_datasets():
    results = []
    
    print("Fetching and benchmarking 20 datasets from OpenML...\n")
    print(f"{'Dataset':<35} | {'Samples':<7} | {'Feats':<5} | {'GaussianNB':<10} | {'IC-NB':<10} | {'Alpha':<5}")
    print("-" * 83)

    for name in DATASETS:
        try:
            # Fetch data
            data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
            X = data.data
            y = data.target
            
            # Basic preprocessing: drop non-numeric and NaNs for simplicity of benchmark
            X = X.select_dtypes(include=[np.number])
            X = X.dropna(axis=1, how='all')
            
            # Combine X and y to drop row-wise NaNs
            df = pd.concat([X, y], axis=1).dropna()
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            if X.shape[0] < 50 or X.shape[1] < 2:
                continue
                
            y = LabelEncoder().fit_transform(y)
            X = StandardScaler().fit_transform(X)

            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Baseline GaussianNB
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            acc_gnb = accuracy_score(y_test, gnb.predict(X_test))

            # Inverse Correlation NB (optimized)
            icnb = InverseCorrelationNB(optimize_alpha=True, metric='pearson')
            icnb.fit(X_train, y_train)
            acc_icnb = accuracy_score(y_test, icnb.predict(X_test))

            alpha_str = "inf" if np.isinf(icnb.best_alpha_) else f"{icnb.best_alpha_:.2f}"
            
            print(f"{name[:35]:<35} | {X.shape[0]:<7} | {X.shape[1]:<5} | {acc_gnb:<10.4f} | {acc_icnb:<10.4f} | {alpha_str:<5}")

            results.append({
                "Dataset": name,
                "Samples": X.shape[0],
                "Features": X.shape[1],
                "GaussianNB": acc_gnb,
                "IC-NB": acc_icnb,
                "Alpha": alpha_str
            })
            
        except Exception as e:
            # Skip datasets that fail to download or format
            continue

    # Generate Markdown Table
    print("\n\n### Markdown Table for README.md\n")
    df_res = pd.DataFrame(results)
    
    # Calculate improvement
    df_res['Improvement'] = df_res['IC-NB'] - df_res['GaussianNB']
    
    # Format table
    md_table = "| Dataset | Samples | Features | GaussianNB | IC-NB | Best Alpha |\n"
    md_table += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for _, row in df_res.iterrows():
        name = row['Dataset']
        samples = row['Samples']
        feats = row['Features']
        gnb = f"{row['GaussianNB']:.4f}"
        icnb = f"**{row['IC-NB']:.4f}**" if row['IC-NB'] > row['GaussianNB'] else f"{row['IC-NB']:.4f}"
        alpha = row['Alpha']
        
        md_table += f"| {name} | {samples} | {feats} | {gnb} | {icnb} | {alpha} |\n"

    print(md_table)
    
    with open('benchmark_results.md', 'w') as f:
        f.write(md_table)

if __name__ == "__main__":
    evaluate_20_datasets()
