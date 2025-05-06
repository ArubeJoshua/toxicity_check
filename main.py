from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np
from typing import Optional, Dict, List, Any

app = FastAPI(
    title="Toxicity Detection API",
    description="API for detecting toxic content in text using a pre-trained ML model",
    version="1.0.0"
)

# Request/Response Models (same as before)
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

# Global model variable
toxicity_model = None

def load_model(model_path='toxicity_model.pkl'):
    """Load the pickled model without requiring original class"""
    try:
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load with protocol version that matches how it was saved
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Verify the loaded object has the methods we need
        required_methods = ['predict_proba', 'predict', 'detect_toxic_phrases', 'count_toxic_phrases']
        for method in required_methods:
            if not hasattr(model, method):
                raise AttributeError(f"Loaded model missing required method: {method}")
        
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global toxicity_model
    try:
        toxicity_model = load_model()
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        # In production, you might want to exit if model loading fails
        raise

# API endpoints (same as before)
@app.get("/")
async def read_root():
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
    if toxicity_model is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "message": "API is running and model is loaded"}

@app.post("/api/detect", response_model=TextResponse)
async def detect_toxicity(request: TextRequest):
    if toxicity_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        proba = toxicity_model.predict_proba([request.text])[0][1]
        is_toxic = proba > request.threshold
        
        response = {
            "text": request.text,
            "is_toxic": bool(is_toxic),
            "toxic_score": float(proba),
        }
        
        if request.return_details:
            # These methods will work if they exist in the loaded model
            toxic_phrases = toxicity_model.detect_toxic_phrases(request.text)
            category_counts = toxicity_model.count_toxic_phrases(request.text)
            
            response["details"] = {
                "toxic_phrases": [{"phrase": phrase, "category": category} 
                                for phrase, category in toxic_phrases],
                "category_counts": category_counts
            }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/api/batch-detect", response_model=BatchTextResponse)
async def batch_detect_toxicity(request: BatchTextRequest):
    if toxicity_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        probas = toxicity_model.predict_proba(request.texts)[:, 1]
        is_toxic = probas > request.threshold
        
        results = []
        for text, prob, toxic in zip(request.texts, probas, is_toxic):
            result = {
                "text": text,
                "is_toxic": bool(toxic),
                "toxic_score": float(prob),
            }
            
            if request.return_details:
                toxic_phrases = toxicity_model.detect_toxic_phrases(text)
                category_counts = toxicity_model.count_toxic_phrases(text)
                
                result["details"] = {
                    "toxic_phrases": [{"phrase": phrase, "category": category} 
                                    for phrase, category in toxic_phrases],
                    "category_counts": category_counts
                }
            
            results.append(result)
        
        summary = {
            "total_texts": len(request.texts),
            "toxic_texts": int(sum(is_toxic)),
            "non_toxic_texts": int(sum(~is_toxic)),
            "average_toxic_score": float(np.mean(probas)),
        }
        
        return {"results": results, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
