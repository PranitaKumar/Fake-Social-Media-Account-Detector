import concurrent.futures
import os
import numpy as np
import pandas as pd
import re
import joblib
import instaloader
import time
import cv2
import tempfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from deepface import DeepFace
import mediapipe as mp
import requests
from io import BytesIO

class IntegratedFakeDetector:
    def __init__(self, model_path="best_rf_model.pkl", scaler_path="scaler.pkl", weights=None):
        """
        Initialize the integrated fake account detector with the three models
        
        Parameters:
        - model_path: Path to the trained random forest model
        - scaler_path: Path to the trained scaler
        - weights: Optional weights for each analysis component [username_weight, bio_weight, image_weight]
        """
        # Load the RF model and scaler
        self.rf_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Initialize Mediapipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        
        # Spam keywords for bio analysis
        self.spam_keywords = [
            "gift", "giveaway", "increase followers", "cash", "free", "win", 
            "deal", "offer", "promotion", "boost", "limited time", "making money online",
            "earn from home", "passive income", "quick money", "easy money", "followers","increase"
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
        Main function to analyze an Instagram account using all three models in parallel
        
        Returns a comprehensive report with all analysis results
        """
        print(f"Starting comprehensive analysis for @{username}...")
        
        # Fetch account data if not provided
        profile_data = self._fetch_profile_data(username, custom_bio, custom_follower_count, custom_following_count)
        
        if isinstance(profile_data, str) and "Error" in profile_data:
            return {"error": profile_data, "account_status": None}
        
        # Run all three analyses in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks to the executor
            rf_task = executor.submit(
                self._run_rf_analysis,
                username,
                profile_data["bio"],
                profile_data["followers"],
                profile_data["following"]
            )
            
            bio_task = executor.submit(
                self._analyze_bio_for_spam,
                profile_data["bio"]
            )
            
            # Use alternative method to get profile image instead of Selenium
            image_task = executor.submit(
                self._get_profile_image_instaloader,
                username
            )
            
            # Get results as they complete
            rf_result = rf_task.result()
            bio_result = bio_task.result()
            image_result = image_task.result()
        
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
    
    def _fetch_profile_data(self, username, custom_bio=None, custom_follower_count=None, custom_following_count=None):
        """
        Fetch profile data from Instagram or use provided custom data
        """
        # If custom data is provided, use it
        if custom_bio is not None and custom_follower_count is not None and custom_following_count is not None:
            return {
                "bio": custom_bio,
                "followers": custom_follower_count,
                "following": custom_following_count
            }
        
        # Otherwise, fetch from Instagram
        try:
            L = instaloader.Instaloader()
            profile = instaloader.Profile.from_username(L.context, username)
            
            return {
                "bio": profile.biography,
                "followers": profile.followers,
                "following": profile.followees
            }
        except instaloader.exceptions.ProfileNotExistsException:
            return f"Error: The profile '{username}' does not exist."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _run_rf_analysis(self, username, bio, followers, following):
        """
        Run the Random Forest model analysis
        """
        # Extract features
        features = self._extract_features(username, bio, followers, following)
        
        # Convert features into a list
        features_list = list(features.values())
        
        # Scale the features
        X_scaled = self.scaler.transform([features_list])
        
        # Predict with the trained model
        prediction = self.rf_model.predict(X_scaled)
        prediction_proba = self.rf_model.predict_proba(X_scaled)[0][1]  # Probability of being fake
        
        # Return results
        return {
            "prediction": "fake" if prediction[0] == 1 else "real",
            "confidence": prediction_proba,
            "features": features
        }
    
    def _extract_features(self, username, bio, followers, following):
        """
        Extract features for the Random Forest model
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
    
    def _get_profile_image_instaloader(self, username):
        """
        Use instaloader to download profile picture instead of Selenium
        This is more reliable than using a browser
        """
        try:
            # Create a temporary directory to store the profile picture
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Initialize Instaloader
                L = instaloader.Instaloader(dirname_pattern=tmpdirname)
                
                # Download only the profile picture
                L.download_profile(username, profile_pic_only=True)
                
                # Find the profile pic in the directory
                profile_pic_path = None
                for file in os.listdir(tmpdirname):
                    if file.endswith('.jpg') and not file.endswith('_UTC.jpg'):
                        profile_pic_path = os.path.join(tmpdirname, file)
                        break
                
                if not profile_pic_path:
                    return {
                        "is_fake_image": True,
                        "face_detected": False,
                        "message": "No profile image found or account is private"
                    }
                
                # Check if a face is detected in the image
                face_detected = self._detect_face(profile_pic_path)
                
                if not face_detected:
                    return {
                        "is_fake_image": True,
                        "face_detected": False,
                        "message": "Profile image exists but no face detected"
                    }
                
                # Perform reverse image search if we found a face
                # Set up a driver for the reverse image search
                driver = self._setup_driver()
                try:
                    # Load the image
                    img = cv2.imread(profile_pic_path)
                    
                    # Perform reverse image search
                    is_fake = self._process_image_array(driver, img, username)
                    
                    return {
                        "is_fake_image": is_fake,
                        "face_detected": True,
                        "message": "Found similar images online" if is_fake else "No similar images found online"
                    }
                except Exception as e:
                    print(f"Error during reverse image search: {str(e)}")
                    return {
                        "is_fake_image": True,  # Mark as suspicious on error
                        "face_detected": True,
                        "message": f"Error during image analysis: {str(e)}"
                    }
                finally:
                    driver.quit()
        
        except Exception as e:
            return {
                "is_fake_image": True,  # Consider errors as suspicious by default
                "face_detected": False,
                "message": f"Error accessing profile image: {str(e)}"
            }
    
    def _setup_driver(self):
        """
        Set up the Chrome driver for Yandex reverse image search with improved error handling
        """
        try:
            # Create ChromeOptions object
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            
            # Create the WebDriver object
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            # Set page load timeout
            driver.set_page_load_timeout(30)
            
            return driver
        except Exception as e:
            print(f"Error setting up WebDriver: {str(e)}")
            raise
    
    def _detect_face(self, img_path_or_array):
        """
        Detect face in an image using MediaPipe
        Works with both file paths and image arrays
        """
        try:
            if isinstance(img_path_or_array, str):
                # If input is a file path
                img = cv2.imread(img_path_or_array)
                if img is None:
                    return False
            else:
                # If input is already an array
                img = img_path_or_array
                
            # Convert to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_img)
            return bool(results.detections)
        except Exception as e:
            print(f"Error detecting face: {str(e)}")
            return False
    
    def _extract_image_urls(self, driver):
        """
        Extract image URLs from Yandex search results
        """
        try:
            time.sleep(5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            
            # Find image elements using various selectors to increase chances of finding images
            image_elements = driver.find_elements(By.XPATH,
                "//div[contains(@class,'CbirPreview-Image')]//img | //div[contains(@class,'CbirSites-Item')]//img | //div[contains(@class,'Image')]//img")
            
            urls = []
            for img in image_elements:
                try:
                    src = img.get_attribute("src")
                    if src and not src.startswith("data:image") and not src.endswith(".svg"):
                        urls.append(src)
                except Exception:
                    continue
                    
            # If no images found with first method, try a more generic approach
            if not urls:
                all_images = driver.find_elements(By.TAG_NAME, "img")
                for img in all_images:
                    try:
                        src = img.get_attribute("src")
                        if src and not src.startswith("data:image") and not src.endswith(".svg"):
                            urls.append(src)
                    except Exception:
                        continue
            
            return urls[:10]  # Return up to 10 image URLs
        except Exception as e:
            print(f"Error extracting image URLs: {str(e)}")
            return []
    
    def _compare_with_deepface(self, original_path, result_urls):
        """
        Compare the original image with found images using DeepFace
        Returns True if a match is found (indicating the image might be fake/stock)
        """
        for url in result_urls:
            try:
                # Download the image from URL for comparison
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Create temporary file for the URL image
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_url_file:
                        temp_url_path = temp_url_file.name
                        temp_url_file.write(response.content)
                    
                    # Perform verification
                    try:
                        # Use a more lenient verification threshold
                        result = DeepFace.verify(
                            original_path, 
                            temp_url_path, 
                            model_name='Facenet', 
                            distance_metric="cosine",
                            enforce_detection=False
                        )
                        
                        # Clean up temp file
                        os.remove(temp_url_path)
                        
                        # Check if the face matches
                        if result['verified']:
                            return True  # Found a match
                    except Exception as e:
                        print(f"DeepFace verification error: {str(e)}")
                        os.remove(temp_url_path)  # Clean up even on error
                        continue
            except Exception as e:
                print(f"Error downloading or processing URL image: {str(e)}")
                continue
                
        return False  # No matches found
    
    def _process_image_array(self, driver, img_array, username):
        """
        Process the image through Yandex reverse image search and DeepFace
        Returns True if the image is likely fake (found similar images online)
        """
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, img_array)

        try:
            # Double-check face detection with the saved image
            if not self._detect_face(temp_path):
                os.remove(temp_path)
                return True  # Mark as suspicious if no face detected

            # Navigate to Yandex Images for reverse image search
            try:
                driver.get("https://yandex.com/images/")
                
                # Check if the page loaded properly
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
                )
            except Exception as e:
                print(f"Error navigating to Yandex: {str(e)}")
                # Try Google Images as a fallback
                try:
                    driver.get("https://images.google.com/")
                    time.sleep(3)
                    # Click on camera icon for image search
                    camera_button = driver.find_element(By.CLASS_NAME, "K1kFoe")
                    camera_button.click()
                    time.sleep(2)
                except Exception:
                    os.remove(temp_path)
                    return True  # Mark as suspicious if both search engines fail
            
            try:
                # Upload the image - works for both Yandex and Google
                upload_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
                )
                upload_input.send_keys(temp_path)
                time.sleep(15)  # Wait longer for search results to load

                # Extract image URLs from search results
                result_urls = self._extract_image_urls(driver)
                
                if result_urls:
                    # Compare the original image with found images
                    match = self._compare_with_deepface(temp_path, result_urls)
                    os.remove(temp_path)
                    return match  # Return True if similar images found
                else:
                    os.remove(temp_path)
                    return False  # No similar images found

            except Exception as e:
                print(f"Error during image upload or processing: {str(e)}")
                os.remove(temp_path)
                return True  # Mark as suspicious if error occurs during analysis
                
        except Exception as e:
            print(f"General error in image processing: {str(e)}")
            try:
                os.remove(temp_path)
            except:
                pass
            return True  # Mark as suspicious if error occurs


# Example usage
def main():
    # Create the detector
    detector = IntegratedFakeDetector()
    
    # Analyze an account (either fetch from Instagram or provide custom data)
    username = input("Enter the Instagram username to analyze: ")
    
    # Option 1: Let the system fetch all data
    result = detector.analyze_account(username)
    
    # Option 2: Provide custom data (useful for testing or when scraping is limited)
    # result = detector.analyze_account(
    #     username, 
    #     custom_bio="Here to win a giveaway! Free cash prize available.",
    #     custom_follower_count=150,
    #     custom_following_count=200
    # )
    
    # Display the results
    print("\n===== ACCOUNT ANALYSIS RESULTS =====")
    print(f"Username: @{result['username']}")
    
    # Display the account status with color-coded output if supported
    status = result['account_status']
    score = result['final_score']
    
    if status == "fake":
        status_text = "FAKE ACCOUNT"
    elif status == "suspicious":
        status_text = "SUSPICIOUS ACCOUNT (potentially fake)"
    else:
        status_text = "REAL ACCOUNT"
    
    print(f"Final Assessment: {status_text} (Score: {score:.2f})")
    print(f"Classification Details:")
    print(f"  - Score < 0.50: Fake Account")
    print(f"  - Score 0.50-0.55: Suspicious Account")
    print(f"  - Score > 0.55: Real Account")
    
    print("\n--- Individual Analysis Results ---")
    print(f"Random Forest Model: {result['rf_analysis']['prediction']} (Confidence: {result['rf_analysis']['confidence']:.2f})")
    print(f"Bio Analysis: {'Suspicious' if result['bio_analysis']['is_spam'] else 'Clean'}")
    if result['bio_analysis']['is_spam']:
        print(f"  - Suspicious keywords: {', '.join(result['bio_analysis']['detected_keywords'])}")
    print(f"Image Analysis: {'Fake/Stock image or No Profile Picture' if result['image_analysis']['is_fake_image'] else 'Likely authentic'}")
    print(f"  - {result['image_analysis']['message']}")
    print(f"Follower/Following Ratio: {'Suspicious' if result['ratio_analysis']['is_suspicious'] else 'Normal'}")
    print(f"  - Ratio: {result['ratio_analysis']['ratio']}")
    print("\n--- Account Statistics ---")
    print(f"Followers: {result['account_data']['followers']}")
    print(f"Following: {result['account_data']['following']}")
    print(f"Bio: {result['account_data']['bio']}")

if __name__ == "__main__":
    main()