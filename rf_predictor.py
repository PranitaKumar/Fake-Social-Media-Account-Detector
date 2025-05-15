import pandas as pd
import re
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
rf_model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature extraction function (same as during training)
def extract_features(username, bio, followers, following):
    return {
        "username_length": len(username),
        "digit_count": len(re.findall(r'\d', username)),
        "followers": followers,
        "following": following,
        "ratio": followers / following if following != 0 else 0,
        "bio_length": len(bio),
        "keyword_count": sum(kw in bio.lower() for kw in ["giveaway", "win", "cash", "offer", "deal", "gift"])
    }

# Function to predict if the account is fake or real
def predict_fake_account(username, bio, followers, following):
    # Step 1: Extract features
    features = extract_features(username, bio, followers, following)
    
    # Step 2: Convert features into a list (same order as during training)
    features_list = list(features.values())
    
    # Step 3: Scale the features
    X_scaled = scaler.transform([features_list])
    
    # Step 4: Predict with the trained model
    prediction = rf_model.predict(X_scaled)
    
    # Step 5: Return result
    if prediction[0] == 1:
        return "This account is likely FAKE."
    else:
        return "This account is REAL."

# Main function to input username and analyze profile
def main():
    # Enter username and profile data manually or by scraping the profile
    username = input("Enter the username: ")
    
    # Here you can replace these static values with scraping logic for real-time profile data
    # This example uses mock data for testing:
    bio = "Here to win a giveaway! Free cash prize available."
    followers = 150
    following = 200

    # Step 6: Run prediction
    result = predict_fake_account(username, bio, followers, following)
    print(f"Analysis result for @{username}: {result}")

if __name__ == "__main__":
    main()
