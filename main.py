import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Toxicity Detection API")

# Download NLTK resources on startup
@app.on_event("startup")
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

# Load the trained model
@app.on_event("startup")
def load_model():
    global model
    model_path = os.environ.get("MODEL_PATH", "aurathreads_toxicity_detector.pkl")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Text preprocessing functions
def clean_text(text):
    """Clean the text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Keep alphanumeric and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_text(text):
    """Process text for model input"""
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    # Join back to string
    return " ".join(tokens)

def predict_toxicity(text, threshold=0.5):
    """Predict if text is toxic using the loaded model"""
    if not model:
        return {"error": "Model not loaded"}, 500
    
    processed_text = process_text(text)
    
    # Get prediction
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba([processed_text])
        is_toxic = proba[0][1] > threshold
        toxic_probability = float(proba[0][1])
    else:
        # For models without predict_proba (like LinearSVC)
        decision = model.decision_function([processed_text])
        toxic_probability = float(1 / (1 + np.exp(-decision[0])))
        is_toxic = toxic_probability > threshold
    
    return {
        "is_toxic": bool(is_toxic),
        "toxic_probability": toxic_probability
    }

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Toxicity Detection API is running"}

@app.post("/predict")
async def predict(text: str):
    try:
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text field is required"}
            )
        
        result = predict_toxicity(text)
        return result
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )
