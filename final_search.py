import os
import time
import cv2
import numpy as np
import tempfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from deepface import DeepFace
import mediapipe as mp
import requests
from io import BytesIO

# ✅ Load Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_img)
    return bool(results.detections)

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def extract_image_urls(driver):
    time.sleep(5)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)
    image_elements = driver.find_elements(By.XPATH,
        "//div[contains(@class,'CbirPreview-Image')]//img | //div[contains(@class,'CbirSites-Item')]//img")
    urls = []
    for img in image_elements:
        src = img.get_attribute("src")
        if src and not src.startswith("data:image"):
            urls.append(src)
    return urls[:10]

def compare_with_deepface(original_path, result_urls):
    for idx, url in enumerate(result_urls):
        try:
            result = DeepFace.verify(original_path, url, model_name='Facenet', enforce_detection=False)
            if result['verified']:
                return True
        except Exception:
            continue
    return False

def process_image_array(driver, img_array, filename="input.jpg"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = temp_file.name
        cv2.imwrite(temp_path, img_array)

    if not detect_face(temp_path):
        os.remove(temp_path)
        return False

    driver.get("https://yandex.com/images/")
    try:
        upload_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        upload_input.send_keys(temp_path)
        time.sleep(10)

        result_urls = extract_image_urls(driver)
        if result_urls:
            match = compare_with_deepface(temp_path, result_urls)
            os.remove(temp_path)
            return match
        else:
            os.remove(temp_path)
            return False

    except Exception:
        os.remove(temp_path)
        return False

def analyze_profile_image(username):
    url = f"https://www.instagram.com/{username}/"
    driver = setup_driver()
    try:
        driver.get(url)
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        img_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//img[contains(@alt, 'Profile photo') or contains(@alt, 'Avatar') or contains(@alt, 'profile picture')]"))
        )
        img_url = img_element.get_attribute("src")
        img_data = requests.get(img_url).content
        img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if img_array is not None:
            result = process_image_array(driver, img_array, filename=f"{username}.jpg")
            return result
        else:
            return False

    except Exception as e:
        return False

    finally:
        driver.quit()
