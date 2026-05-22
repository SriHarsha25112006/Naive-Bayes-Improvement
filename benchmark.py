import pandas as pd
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from icnb import InverseCorrelationNB
import warnings

warnings.filterwarnings("ignore")

def evaluate_datasets():
    datasets = {
        "Digits (Large, 64 features)": load_digits(return_X_y=True),
        "Breast Cancer (Med, 30 features)": load_breast_cancer(return_X_y=True),
        "Wine (Small, 13 features)": load_wine(return_X_y=True),
        "Iris (Tiny, 4 features)": load_iris(return_X_y=True)
    }

    results = []

    for name, (X, y) in datasets.items():
        print(f"\nEvaluating {name}...")
        X = StandardScaler().fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 1. Baseline GaussianNB
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        baseline_acc = accuracy_score(y_test, gnb.predict(X_test))
        print(f"  GaussianNB Accuracy: {baseline_acc:.4f}")
        
        # 2. IC-NB with Automatic Optimization
        icnb = InverseCorrelationNB(optimize_alpha=True, metric='pearson')
        icnb.fit(X_train, y_train)
        icnb_acc = accuracy_score(y_test, icnb.predict(X_test))
        print(f"  IC-NB Accuracy:      {icnb_acc:.4f}")
        print(f"  -> Automatically selected Alpha: {icnb.best_alpha_}")
        
        results.append({
            "Dataset": name.split(" ")[0],
            "GaussianNB": baseline_acc,
            "IC-NB": icnb_acc,
            "Alpha": icnb.best_alpha_
        })

    print("\n================ SUMMARY ================")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("=========================================")

if __name__ == "__main__":
    evaluate_datasets()