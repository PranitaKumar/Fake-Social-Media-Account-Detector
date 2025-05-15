from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from account_analysis import IntegratedFakeDetector


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Enable CORS for all routes

# Initialize the fake account detector once at startup
detector = IntegratedFakeDetector()

# Define route to serve the frontend HTML
@app.route('/', methods=['GET'])
def serve_frontend():
    return send_from_directory('.', 'frontend.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_account():
    data = request.json
    username = data.get('username')
    
    if not username:
        return jsonify({"error": "Username is required"}), 400
    
    # Use the IntegratedFakeDetector to analyze the account
    try:
        # You can either fetch from Instagram or use custom data for testing
        # For production, use this:
        result = detector.analyze_account(username)
        
        # For testing without Instagram API, use custom data:
        # result = detector.analyze_account(
        #     username, 
        #     custom_bio="I love sharing my photos!",
        #     custom_follower_count=150,
        #     custom_following_count=200
        # )
        
        # Check if an error occurred
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        # Format the results to match what the frontend expects
        formatted_result = {
            "username": username,
            "score": result["final_score"],
            "account_status": result["account_status"],
            "bioAnalysis": {
                "bio": result["bio_analysis"]["bio_text"],
                "risk": _calculate_risk_percentage(0.7 if result["bio_analysis"]["is_spam"] else 0.3),
                "finding": "This account's bio contains suspicious content typically associated with fake or spam accounts." 
                          if result["bio_analysis"]["is_spam"] 
                          else "This account's bio appears authentic and lacks suspicious promotional language."
            },
            "photoAnalysis": {
                "risk": 75 if result["image_analysis"]["is_fake_image"] else 30,
                "authenticity": 10 - (7 if result["image_analysis"]["is_fake_image"] else 3),
                "aiProbability": 70 if result["image_analysis"]["is_fake_image"] else 20,
                "hasProfilePicture": result["image_analysis"]["face_detected"]
            },
            "engagementAnalysis": {
                "risk": 75 if result["ratio_analysis"]["is_suspicious"] else 30,
                "followerRatio": f"1:{result['ratio_analysis']['ratio']:.1f}" if isinstance(result['ratio_analysis']['ratio'], (int, float)) else result['ratio_analysis']['ratio'],
                "engagementRate": "2.1%" if result["account_status"] == "fake" else "3.8%" if result["account_status"] == "suspicious" else "5.2%"
            },
            "overallAnalysis": {
                "risk": _calculate_risk_percentage(result["final_score"]),
                "finding": _generate_overall_finding(result["account_status"], result)
            }
        }
        
        return jsonify(formatted_result)
    
    except Exception as e:
        print(f"Error analyzing account: {str(e)}")
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

def _calculate_risk_percentage(score):
    """Convert model confidence scores to risk percentages (0-100)"""
    # For detector model, higher score = higher risk of being fake
    # Convert to percentage and ensure it's between 0-100
    return min(int(score * 100), 100)

def _generate_overall_finding(account_status, result):
    """Generate an appropriate finding text based on the analysis results"""
    if account_status == "fake":
        return "High probability of being a fake account. Multiple suspicious indicators detected across username, photos, and engagement patterns."
    elif account_status == "suspicious":
        suspicious_factors = []
        if result["rf_analysis"]["prediction"] == "fake":
            suspicious_factors.append("username patterns")
        if result["bio_analysis"]["is_spam"]:
            suspicious_factors.append("bio content")
        if result["image_analysis"]["is_fake_image"]:
            suspicious_factors.append("profile photo")
        if not result["image_analysis"]["face_detected"]:
            suspicious_factors.append("missing profile picture")
        if result["ratio_analysis"]["is_suspicious"]:
            suspicious_factors.append("follower-following ratio")
        
        factors_text = ", ".join(suspicious_factors)
        return f"This profile shows some suspicious indicators that suggest it may be a fake account. Concerning factors include: {factors_text}. Further verification is recommended."
    else:
        return "This profile shows few indicators of being fake. Most characteristics align with typical legitimate user behavior."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)