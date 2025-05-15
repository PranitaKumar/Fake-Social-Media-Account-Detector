import pandas as pd
import re
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your dataset (make sure to replace with actual dataset path)
df = pd.read_csv("Dataset.csv")
df.fillna("", inplace=True)

# Feature extraction (same as you did before)
def extract_features(row):
    username = str(row.get("username", ""))
    bio = str(row.get("bio", ""))
    followers = int(row.get("followers", 0))
    following = int(row.get("following", 0))
    
    return pd.Series({
        "username_length": len(username),
        "digit_count": len(re.findall(r'\d', username)),
        "followers": followers,
        "following": following,
        "ratio": followers / following if following != 0 else 0,
        "bio_length": len(bio),
        "keyword_count": sum(kw in bio.lower() for kw in ["giveaway", "win", "cash", "offer", "deal", "gift","pay for followers"])
    })

# Apply feature extraction
features = df.apply(extract_features, axis=1)

# Attach the label column (the target variable)
features["label"] = df["isFake"]

# Split data into train and test sets
X = features.drop("label", axis=1)
y = features["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# Train the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the scaler and model
joblib.dump(scaler, "scaler.pkl")
joblib.dump(rf_model, "best_rf_model.pkl")

print("✅ Model and scaler saved successfully!")
