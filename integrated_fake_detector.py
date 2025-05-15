import re
import random

class IntegratedFakeDetector:
    def __init__(self, model_path=None, scaler_path=None, weights=None):
        """
        Initialize the integrated fake account detector (simplified for demo)
        """
        # Spam keywords for bio analysis
        self.spam_keywords = [
            "gift", "giveaway", "increase followers", "cash", "free", "win", 
            "deal", "offer", "promotion", "boost", "limited time", "making money online",
            "earn from home", "passive income", "quick money", "easy money"
        ]
        
        # Set weights for each analysis component
        if weights is None:
            self.weights = [0.4, 0.3, 0.3]  # Default weights
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w/total for w in weights]
    
    def analyze_account(self, username, custom_bio=None, custom_follower_count=None, custom_following_count=None):
        """
        Simplified analysis function that mimics the full analysis for demo purposes
        """
        print(f"Starting analysis for @{username}...")
        
        # Use custom data or generate mock data
        if custom_bio is not None and custom_follower_count is not None and custom_following_count is not None:
            profile_data = {
                "bio": custom_bio,
                "followers": custom_follower_count, 
                "following": custom_following_count
            }
        else:
            # For demo, generate realistic mock data
            profile_data = self._generate_mock_data(username)
        
        # Run RF analysis
        rf_result = self._run_rf_analysis(username, profile_data["bio"], 
                                        profile_data["followers"], profile_data["following"])
        
        # Run bio analysis
        bio_result = self._analyze_bio_for_spam(profile_data["bio"])
        
        # Run image analysis (simplified for demo)
        image_result = self._analyze_profile_image(username)
        
        # Calculate final scores
        rf_score = 1.0 if rf_result["prediction"] == "fake" else 0.0
        bio_score = 1.0 if bio_result["is_spam"] else 0.0
        image_score = 1.0 if image_result["is_fake_image"] else 0.0
        
        # Add additional check for suspicious follower/following ratio
        ratio_score = 0.0
        if profile_data["following"] > 0:
            ratio = profile_data["followers"] / profile_data["following"]
            if ratio < 0.5:  # Following significantly more than followers
                ratio_score = 0.5  # Add weight for suspicious ratio
        
        # Calculate weighted final score (including ratio assessment)
        final_score = (
            self.weights[0] * rf_score +
            self.weights[1] * bio_score +
            self.weights[2] * image_score +
            0.2 * ratio_score  # Add 20% weight for ratio assessment
        )
        
        # Normalize final score to be between 0 and 1
        final_score = min(final_score, 1.0)
        
        # Determine classification based on refined thresholds
        if final_score < 0.37:
            account_status = "real"
        elif 0.37 <= final_score <= 0.45:
            account_status = "suspicious"
        else:
            account_status = "fake"
        
        # Create comprehensive report
        report = {
            "username": username,
            "final_score": final_score,
            "account_status": account_status,
            "rf_analysis": rf_result,
            "bio_analysis": bio_result,
            "image_analysis": image_result,
            "ratio_analysis": {
                "ratio": profile_data["followers"] / profile_data["following"] if profile_data["following"] > 0 else "N/A",
                "is_suspicious": ratio_score > 0,
                "score": ratio_score
            },
            "account_data": {
                "followers": profile_data["followers"],
                "following": profile_data["following"],
                "bio": profile_data["bio"]
            }
        }
        
        print(f"Analysis complete for @{username}")
        return report
    
    def _generate_mock_data(self, username):
        """Generate mock profile data for testing"""
        # Use the username to deterministically create data
        # This way same username always gives same results
        seed = sum(ord(c) for c in username)
        random.seed(seed)
        
        # Generate followers and following
        followers = random.randint(50, 5000)
        
        # Generate more following than followers for suspicious accounts
        if sum(ord(c) for c in username) % 7 == 0:  # Deterministic way to select some accounts
            following = followers * random.randint(3, 8)
        else:
            following = random.randint(max(50, followers//3), followers*2)
        
        # Generate bio
        bios = [
            f"Living life to the fullest | {username} personal account",
            f"Follow for follow | DM me | {username}",
            f"Instagram official | Photography lover | {username}",
            f"Click the link for a free giveaway! Win $100 gift card #ad",
            f"Travel enthusiast | Dog lover | Food photographer",
            f"Earn passive income from home! DM me to learn how #{username}",
            f"Personal posts | No DMs please | #{username}",
            f"Sharing my journey | Follow for updates | #{username}"
        ]
        
        # Select a bio based on username
        bio_index = sum(ord(c) for c in username) % len(bios)
        bio = bios[bio_index]
        
        return {
            "bio": bio,
            "followers": followers,
            "following": following
        }
    
    def _run_rf_analysis(self, username, bio, followers, following):
        """
        Simplified RF model analysis
        """
        # Extract features
        features = self._extract_features(username, bio, followers, following)
        
        # Use features to determine if account is fake
        score = 0
        
        # Username with lots of digits is suspicious
        if features["digit_count"] > 2:
            score += 0.3
        
        # Low followers to following ratio is suspicious
        if features["ratio"] < 0.5:
            score += 0.3
        
        # Short username is slightly suspicious
        if features["username_length"] < 6:
            score += 0.1
        
        # Keyword-heavy bio is suspicious
        if features["keyword_count"] > 1:
            score += 0.3
        
        # Use username to add some randomness but be deterministic
        seed_value = sum(ord(c) for c in username)
        random.seed(seed_value)
        random_factor = random.random() * 0.2
        score += random_factor
        
        prediction = "fake" if score > 0.5 else "real"
        
        # Return results
        return {
            "prediction": prediction,
            "confidence": score,
            "features": features
        }
    
    def _extract_features(self, username, bio, followers, following):
        """
        Extract features for the model
        """
        return {
            "username_length": len(username),
            "digit_count": len(re.findall(r'\d', username)),
            "followers": followers,
            "following": following,
            "ratio": followers / following if following != 0 else 0,
            "bio_length": len(bio),
            "keyword_count": sum(kw in bio.lower() for kw in ["giveaway", "win", "cash", "offer", "deal", "gift", "money"])
        }
    
    def _analyze_bio_for_spam(self, bio):
        """
        Analyze the bio for spam content
        """
        # Check if any spammy keywords are present in the bio
        spam_detected = False
        detected_keywords = []
        
        # Check exact matches
        for keyword in self.spam_keywords:
            if keyword.lower() in bio.lower():
                spam_detected = True
                detected_keywords.append(keyword)
        
        # Check partial matches for related phrases
        if "money" in bio.lower() and "online" in bio.lower():
            if "money online" not in detected_keywords and "making money online" not in detected_keywords:
                spam_detected = True
                detected_keywords.append("money online")
        
        return {
            "is_spam": spam_detected,
            "detected_keywords": detected_keywords,
            "bio_text": bio
        }
    
    def _analyze_profile_image(self, username):
        """
        Simplified analysis for profile image
        Uses the username to deterministically decide if the image is fake
        """
        # Use username hash to determine if image is fake (for demo)
        username_value = sum(ord(c) for c in username)
        is_fake = (username_value % 5 == 0)  # 20% chance of being fake
        
        return {
            "is_fake_image": is_fake,
            "face_detected": True,
            "message": "Found similar images online" if is_fake else "No similar images found online"
        }