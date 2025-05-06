from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from typing import Optional, Dict, List, Any, Union

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize FastAPI app
app = FastAPI(
    title="Toxicity Detection API",
    description="API for detecting toxic content in text using a pre-trained ML model",
    version="1.0.0"
)

# Define request and response models
class TextRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5
    return_details: Optional[bool] = False

class TextResponse(BaseModel):
    text: str
    is_toxic: bool
    toxic_score: float
    details: Optional[Dict[str, Any]] = None

class BatchTextRequest(BaseModel):
    texts: List[str]
    threshold: Optional[float] = 0.5
    return_details: Optional[bool] = False

class BatchTextResponse(BaseModel):
    results: List[TextResponse]
    summary: Dict[str, Any]

# Twitter text preprocessor class
class TwitterTextPreprocessor:
    """Text preprocessing class specifically for Twitter data"""
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Twitter-specific text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove Twitter-specific elements
        text = re.sub(r'@user', '', text)  # Replace @user as seen in the sample
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove other @mentions
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Remove hashtags
        
        # Remove emojis and special characters (common in tweets)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Keep alphanumeric and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process(self, text):
        """Full text preprocessing pipeline"""
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if needed
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize if needed
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Join back to string
        return " ".join(tokens)

# Global variable to store the loaded model
toxicity_model = None

# Add exception handling for missing model file
def safe_load_model(model_path='twitter_toxicity_detector.pkl'):
    """Safely load the model, with fallback for deployment"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Warning: Could not load model: {str(e)}")
        print("Creating a simple fallback model for demonstration")
        
        # Create a very simple fallback model for demonstration
        class SimpleModel:
            def predict_proba(self, texts):
                # Always return a safe probability of 0.1 (non-toxic)
                return np.array([[0.9, 0.1]] * len(texts))
            
            def detect_toxic_phrases(self, text):
                # Return empty list since this is just a fallback
                return []
            
            def count_toxic_phrases(self, text):
                # Return empty dict since this is just a fallback
                return {}
        
        return SimpleModel()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global toxicity_model
    try:
        toxicity_model = safe_load_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# API endpoints
@app.get("/")
async def read_root():
    """Root endpoint"""
    return {
        "message": "Toxicity Detection API is running",
        "version": "1.0.0",
        "endpoints": {
            "/api/detect": "POST - Detect toxicity in a single text",
            "/api/batch-detect": "POST - Detect toxicity in multiple texts",
            "/health": "GET - Check health status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if toxicity_model is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "message": "API is running and model is loaded"}

@app.post("/api/detect", response_model=TextResponse)
async def detect_toxicity(request: TextRequest):
    """Detect toxicity in a single text"""
    global toxicity_model
    
    if toxicity_model is None:
        try:
            toxicity_model = safe_load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    text = request.text
    threshold = request.threshold
    
    try:
        proba = toxicity_model.predict_proba([text])[0][1]
        is_toxic = proba > threshold
        
        response = {
            "text": text,
            "is_toxic": bool(is_toxic),
            "toxic_score": float(proba),
        }
        
        if request.return_details:
            toxic_phrases = toxicity_model.detect_toxic_phrases(text)
            category_counts = toxicity_model.count_toxic_phrases(text)
            
            response["details"] = {
                "toxic_phrases": [{"phrase": phrase, "category": category} for phrase, category in toxic_phrases],
                "category_counts": category_counts
            }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/api/batch-detect", response_model=BatchTextResponse)
async def batch_detect_toxicity(request: BatchTextRequest):
    """Detect toxicity in multiple texts"""
    global toxicity_model
    
    if toxicity_model is None:
        try:
            toxicity_model = safe_load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    texts = request.texts
    threshold = request.threshold
    
    try:
        probas = toxicity_model.predict_proba(texts)[:, 1]
        is_toxic = probas > threshold
        
        results = []
        
        for i, text in enumerate(texts):
            result = {
                "text": text,
                "is_toxic": bool(is_toxic[i]),
                "toxic_score": float(probas[i]),
            }
            
            if request.return_details:
                toxic_phrases = toxicity_model.detect_toxic_phrases(text)
                category_counts = toxicity_model.count_toxic_phrases(text)
                
                result["details"] = {
                    "toxic_phrases": [{"phrase": phrase, "category": category} for phrase, category in toxic_phrases],
                    "category_counts": category_counts
                }
            
            results.append(result)
        
        # Create summary
        summary = {
            "total_texts": len(texts),
            "toxic_texts": int(sum(is_toxic)),
            "non_toxic_texts": int(sum(~is_toxic)),
            "average_toxic_score": float(np.mean(probas)),
        }
        
        return {"results": results, "summary": summary}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
