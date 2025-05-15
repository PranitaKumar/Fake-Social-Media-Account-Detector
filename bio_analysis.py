import instaloader

# List of spammy or suspicious keywords to look for in the bio
spam_keywords = ["gift", "giveaway", "increase followers", "cash", "free", "win", "deal", "offer", "promotion", "boost", "limited time","followers","increase","contact", "enquiries","next","level"]

# Function to analyze user bio for spammy content
def analyze_bio_for_spam(bio):
    # Step 1: Check if any spammy keywords are present in the bio
    for keyword in spam_keywords:
        if keyword.lower() in bio.lower():  # Check case-insensitive
            return "This account is likely FAKE due to suspicious bio content."
    
    # If no spammy keywords are found
    return "This account's bio seems clean. No spammy content detected."

# Function to extract the bio of a user using Instaloader
def extract_bio(username):
    # Initialize Instaloader
    L = instaloader.Instaloader()

    try:
        # Load the profile of the given username
        profile = instaloader.Profile.from_username(L.context, username)
        
        # Extract the bio
        bio = profile.biography
        return bio
    except instaloader.exceptions.ProfileNotExistsException:
        return f"Error: The profile '{username}' does not exist."
    except Exception as e:
        return f"Error: {str(e)}"
