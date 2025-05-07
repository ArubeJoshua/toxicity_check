import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from typing import Dict, List, Optional, Any

# Initialize FastAPI app
app = FastAPI(title="Enhanced Toxicity Detection API")

# Define toxic phrases patterns
ENHANCED_TOXIC_PHRASES = {
    'self_harm': [
        r'\b(kill|k[!1i]ll|k\*+ll)\s*(your|ur|yr)\s*(self|s3lf|s[e3]lf|yourself)',
        r'\b(go|just|why\s+don\'*t\s+you)\s+(die|end\s+your\s+life|hang\s+yourself)',
        r'\bkys\b',
        r'\b(commit|attempt)\s+suicide',
        r'\bjump\s+(off|from)\s+a\s+(bridge|building|cliff)',
        r'\bno\s+one\s+would\s+(miss|care\s+about)\s+you',
        r'\bk\W*y\W*s\b',
        r'\bd[i1!]\W*[e3]\b',
        r'\bs[u\*][i1!]\W*c[i1!]d[e3]\b',
        r'\bjust\s+die\b',
        r'\bwe\s+(don\'t|do\s+not)\s+care\s+(about|for)\s+you\b',
        r'\bwe\s+are\s+tired\s+of\s+you\b',
    ],
    'slurs': [
        r'\b(n[i!1]gg[ae3]r|n[i!1]gg[a3]|n[i!1]gg)',
        r'\bf[a@]gg[o0]t|\bf[a@]g\b',
        r'\br[e3]t[a@]rd[e3]d|\br[e3]t[a@]rd',
        r'\bd[y!1]ke',
        r'\btr[a@]nny',
        r'\bspan[i!1]c',
        r'\bc[o0]l[o0]r[e3]d',
        r'\bch[i!1]nk',
        r'\bk[i!1]k[e3]',
        r'\bwh[o0]r[e3]',
        r'\bcunt',
        r'\bc\W*u\W*n\W*t\b',
        r'\bwh[o0]\W*r\W*[e3]\b',
        r'\bf\W*a\W*g\W*([g]\W*[o0]\W*t)?\b',
        r'\bn\W*[i!1]\W*g\W*g\W*[ae3]?\W*r?\b',
        r'\br\W*[e3]\W*t\W*[a@]\W*r\W*d\b',
    ],
    'profanity_targeted': [
        r'\b(fuck|screw|damn)\s+you',
        r'\byou\'*re?\s+(stupid|an\s+idiot|dumb|worthless|pathetic)',
        r'\bi\s+(hate|despise|loathe)\s+you',
        r'\byou\s+(suck|are\s+trash|should\s+die)',
        r'\bnoone\s+(likes|cares\s+about)\s+you',
        r'\byou\s+(deserve|need|should\s+get)\s+(cancer|aids|covid)',
        r'\bf\W*[u\*]\W*c\W*k\s+you\b',
        r'\bf\W{0,2}k\s+you\b',
        r'\bf\W{0,2}u\W{0,2}k\s+you\b',
        r'\bf\W{0,2}ck\s+you\b',
        r'\bf\W{0,2}c\W{0,2}k\s+you\b',
        r'\bf+\s*u+\s*c*\s*k+\s+y+\s*o*\s*u+\b',
        r'\bf\S*k\s+you\b',
        r'\bf\S*\s+you\b',
        r'\bi\s+h\W*[a@]\W*t\W*[e3]\s+you\b',
    ],
    'threats': [
        r'\bi\'*ll?\s+(beat|kill|hunt|find|murder)\s+you',
        r'\bi\'*m\s+going\s+to\s+(beat|kill|hunt|find|murder)\s+you',
        r'\bi\s+(will|would|wanna|want\s+to)\s+(beat|kill|hunt|find|murder)\s+you',
        r'\byou\s+(will|are\s+going\s+to|gonna|should)\s+(die|bleed|suffer)',
        r'\bi\'*ll?\s+find\s+(where|your|ur)\s+(you\s+live|house|family|address)',
        r'\bi\s+w\W*[i!1]\W*l\W*l\s+k\W*[i!1]\W*l\W*l\s+you\b',
        r'\bi\s+w\W*[i!1]\W*l\W*l\s+f\W*[i!1]\W*n\W*d\s+you\b',
    ],
    'hate_speech': [
        r'\b(all|you|those)\s+(should|must|need\s+to|ought\s+to)\s+(die|burn|suffer|perish)',
        r'\b(ethnic\s+cleansing|final\s+solution|white\s+power|white\s+supremacy)',
        r'\b(black\s+people|jews|muslims|gays|immigrants)\s+(are|should)\s+(exterminated|eliminated|wiped\s+out)',
        r'\b(death\s+to|kill\s+all|exterminate)\s+(the\s+)?(jews|muslims|blacks|gays|immigrants)',
        r'\b(racial\s+purity|racial\s+superiority|master\s+race)',
        r'\b(we\s+must|we\s+should)\s+(eliminate|exterminate|get\s+rid\s+of)\s+(them|those\s+people)',
        r'\b(you\s+(people|folks|fucks)\s+should\s+be\s+(wiped\s+out|eliminated))',
        r'\b((all|every)\s+(black\s+people|jews|muslims|gays)\s+are\s+(vermin|scum|subhuman))',
        r'\b((the\s+world|we)\s+would\s+be\s+better\s+off\s+without\s+(blacks|jews|muslims|gays))',
        r'\b(kkk|ku\s*klux\s*klan|nazi|nazis|heil\s+hilter|1488|14\s*words)',
        r'\b(swastika|white\s+pride|aryan\s+nation|blood\s+and\s+honor)',
        r'\b(d\W*e\W*a\W*t\W*h\s+t\W*o\s+t\W*h\W*e\s+j\W*e\W*w\W*s)',
        r'\b(k\W*i\W*l\W*l\s+a\W*l\W*l\s+m\W*u\W*s\W*l\W*i\W*m\W*s)',
        r'\b(w\W*h\W*i\W*t\W*e\s+s\W*u\W*p\W*r\W*e\W*m\W*a\W*c\W*y)',
        r'\b(e\W*x\W*t\W*e\W*r\W*m\W*i\W*n\W*a\W*t\W*e\s+t\W*h\W*e\s+j\W*e\W*w\W*s)',
        r'\b((they|those\s+people)\s+are\s+(vermin|rats|cockroaches|animals|subhuman))',
        r'\b((blacks|jews|muslims)\s+(don\'t|do\s+not)\s+deserve\s+to\s+live)',
        r'\b((exterminate|eliminate)\s+(the\s+)?(jewish|muslim|black)\s+(problem|threat|menace))',
        r'\b(gas\s+the\s+jews|oven\s+the\s+jews|burn\s+the\s+jews)',
        r'\b(lynch\s+(all|the)\s+blacks|hang\s+(all|the)\s+blacks)',
        r'\b(throw\s+(all|the)\s+muslims\s+off\s+cliffs|drown\s+(all|the)\s+muslims)',
        r'\b(round\s+up\s+the\s+(jews|muslims|gays)\s+and\s+(kill|execute|exterminate)\s+them)'
    ],
    'dismissive': [
        r'\bwe\s+(don\'t|do\s+not)\s+care\s+(about|for)\s+you\b',
        r'\bwe\s+are\s+tired\s+of\s+you\b',
        r'\bnobody\s+cares\s+(about|for)\s+you\b',
        r'\bjust\s+go\s+away\b',
        r'\bdisappear\s+already\b',
        r'\bshut\s+up\s+and\s+die\b',
    ]
}

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
            print(f"Model loaded successfully from {model_path}")
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

def check_patterns(text):
    """Check if text matches any of the toxic patterns"""
    text = text.lower()
    results = {}
    
    for category, patterns in ENHANCED_TOXIC_PHRASES.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text)
            if found:
                matches.append(pattern)
        
        if matches:
            results[category] = {
                "detected": True,
                "matched_patterns": matches[:5]  # Limit to 5 matches for readability
            }
        else:
            results[category] = {"detected": False}
    
    # Check if any category has a match
    is_toxic = any(value["detected"] for value in results.values())
    
    return {
        "is_toxic": is_toxic,
        "categories": results
    }

def predict_toxicity_ml(text, threshold=0.5):
    """Predict if text is toxic using the loaded machine learning model"""
    if not model:
        return {"error": "Model not loaded", "is_toxic": False, "toxic_probability": 0.0}
    
    processed_text = process_text(text)
    
    # Get prediction
    try:
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
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return {"error": str(e), "is_toxic": False, "toxic_probability": 0.0}

def two_layer_toxicity_check(text, ml_threshold=0.5, confidence_threshold=0.8):
    """
    Perform two-layer toxicity detection:
    1. Check against regex patterns
    2. Use machine learning model for prediction
    
    Returns combined results with high confidence when both methods agree
    or when one method has high confidence.
    """
    # Layer 1: Pattern-based detection
    pattern_results = check_patterns(text)
    
    # Layer 2: ML-based detection
    ml_results = predict_toxicity_ml(text, threshold=ml_threshold)
    
    # Combine results
    is_toxic_pattern = pattern_results["is_toxic"]
    is_toxic_ml = ml_results.get("is_toxic", False)
    toxic_probability_ml = ml_results.get("toxic_probability", 0.0)
    
    # Decision logic
    if is_toxic_pattern and is_toxic_ml:
        # Both methods agree it's toxic
        final_toxicity = True
        confidence = max(toxic_probability_ml, 0.9)  # High confidence when both agree
        decision_reason = "Both pattern matching and ML model detected toxicity"
    elif is_toxic_pattern:
        # Only pattern matching detected toxicity
        final_toxicity = True
        confidence = max(0.7, toxic_probability_ml)  # Pattern matches are fairly reliable
        decision_reason = "Pattern matching detected explicit toxic content"
    elif is_toxic_ml and toxic_probability_ml > confidence_threshold:
        # ML has high confidence
        final_toxicity = True
        confidence = toxic_probability_ml
        decision_reason = "ML model detected toxic content with high confidence"
    elif is_toxic_ml:
        # ML detected toxicity but with lower confidence
        final_toxicity = True
        confidence = toxic_probability_ml
        decision_reason = "ML model detected potential toxic content"
    else:
        # Both methods agree it's not toxic
        final_toxicity = False
        confidence = 1.0 - toxic_probability_ml
        decision_reason = "No toxicity detected by either method"
    
    return {
        "is_toxic": final_toxicity,
        "confidence": confidence,
        "decision_reason": decision_reason,
        "pattern_analysis": pattern_results,
        "ml_analysis": ml_results
    }

# Input models for API
class ToxicityInput(BaseModel):
    text: str
    ml_threshold: Optional[float] = 0.5
    confidence_threshold: Optional[float] = 0.8

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Enhanced Toxicity Detection API is running"}

@app.post("/check_patterns")
async def api_check_patterns(text: str = Body(..., embed=True)):
    try:
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text field is required"}
            )
        
        result = check_patterns(text)
        return result
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

@app.post("/predict_ml")
async def api_predict_ml(text: str = Body(..., embed=True), threshold: float = Body(0.5, embed=True)):
    try:
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text field is required"}
            )
        
        result = predict_toxicity_ml(text, threshold=threshold)
        return result
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

@app.post("/analyze")
async def analyze_toxicity(input_data: ToxicityInput):
    try:
        if not input_data.text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text field is required"}
            )
        
        result = two_layer_toxicity_check(
            input_data.text, 
            ml_threshold=input_data.ml_threshold,
            confidence_threshold=input_data.confidence_threshold
        )
        return result
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

# Add a batch prediction endpoint for processing multiple texts
class BatchToxicityInput(BaseModel):
    texts: List[str]
    ml_threshold: Optional[float] = 0.5
    confidence_threshold: Optional[float] = 0.8

@app.post("/analyze_batch")
async def analyze_batch(input_data: BatchToxicityInput):
    try:
        if not input_data.texts or len(input_data.texts) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Texts list is required and cannot be empty"}
            )
        
        results = []
        for text in input_data.texts:
            result = two_layer_toxicity_check(
                text,
                ml_threshold=input_data.ml_threshold,
                confidence_threshold=input_data.confidence_threshold
            )
            results.append(result)
        
        return {"results": results}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing batch request: {str(e)}"}
        )

# Add simple test endpoint
@app.get("/test/{text}")
async def test_toxicity(text: str):
    result = two_layer_toxicity_check(text)
    return result
