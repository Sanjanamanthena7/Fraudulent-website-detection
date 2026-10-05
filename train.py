import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from src.feature_extraction import extract_features


def build_synthetic_dataset():
    """Generates a representative baseline dataset for immediate training."""
    raw_urls = [
        # Legitimate URLs (Label 0)
        ("[https://www.google.com](https://www.google.com)", 0),
        ("[https://github.com](https://github.com)", 0),
        ("[https://www.wikipedia.org](https://www.wikipedia.org)", 0),
        ("[https://stackoverflow.com/questions](https://stackoverflow.com/questions)", 0),
        ("[https://aws.amazon.com/console](https://aws.amazon.com/console)", 0),
        ("[https://www.linkedin.com/feed](https://www.linkedin.com/feed)", 0),
        ("[https://portal.azure.com](https://portal.azure.com)", 0),
        ("[https://react.dev/learn](https://react.dev/learn)", 0),
        ("[https://spring.io/projects](https://spring.io/projects)", 0),
        ("[https://www.python.org/downloads](https://www.python.org/downloads)", 0),
        ("[https://developer.mozilla.org/en-US](https://developer.mozilla.org/en-US)", 0),
        ("[https://www.nytimes.com](https://www.nytimes.com)", 0),
        ("[https://medium.com](https://medium.com)", 0),
        ("[https://hub.docker.com](https://hub.docker.com)", 0),
        ("[https://kaggle.com/competitions](https://kaggle.com/competitions)", 0),
        # Fraudulent / Phishing URLs (Label 1)
        ("[http://192.168.1.1/login-verify-account](http://192.168.1.1/login-verify-account)", 1),
        ("[http://paypal-security-update.com](http://paypal-security-update.com)@scam-site.org/signin", 1),
        ("[http://secure-banking-alert-update.servehttp.com](http://secure-banking-alert-update.servehttp.com)", 1),
        ("[http://appleid.apple.com.verify-identity.info](http://appleid.apple.com.verify-identity.info)", 1),
        ("[http://netflix-billing-update-required.xyz/auth?user=test%20](http://netflix-billing-update-required.xyz/auth?user=test%20)", 1),
        ("[http://amazon-account-suspension-recovery.top/signin](http://amazon-account-suspension-recovery.top/signin)", 1),
        ("[http://104.244.42.1/webscr?cmd=_login-run](http://104.244.42.1/webscr?cmd=_login-run)", 1),
        ("[http://chase-bank-verify-profile.tk/secure/login](http://chase-bank-verify-profile.tk/secure/login)", 1),
        ("[http://login.microsoftonline.pass-recovery.support](http://login.microsoftonline.pass-recovery.support)", 1),
        ("[http://wellsfargo-secure-login-attempt.ml](http://wellsfargo-secure-login-attempt.ml)", 1),
        ("[http://free-giftcard-claim-now.xyz/promo?id=123](http://free-giftcard-claim-now.xyz/promo?id=123)", 1),
        ("[http://account-reactivation-google.serveo.net](http://account-reactivation-google.serveo.net)", 1),
        ("[http://secure-instagram-badge-verification.ga](http://secure-instagram-badge-verification.ga)", 1),
        ("[http://online-wallet-metamask-seed.icu](http://online-wallet-metamask-seed.icu)", 1),
        ("[http://binance-kyc-document-reupload.cc/auth](http://binance-kyc-document-reupload.cc/auth)", 1),
    ]

    records = []
    labels = []
    for url, label in raw_urls:
        features_df = extract_features(url)
        records.append(features_df.iloc[0].to_dict())
        labels.append(label)

    df = pd.DataFrame(records)
    df["label"] = labels
    return df


def train_and_save():
    os.makedirs("models", exist_ok=True)
    df = build_synthetic_dataset()

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    # Train Baseline Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    print("=== Random Forest Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(classification_report(y_test, rf_preds, target_names=["Legitimate", "Malicious"]))

    # Save artifacts
    model_path = os.path.join("models", "rf_phishing_model.joblib")
    joblib.dump(rf_model, model_path)
    print(f"Model saved successfully to {model_path}")


if __name__ == "__main__":
    train_and_save()