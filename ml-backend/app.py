"""
FastAPI Backend for ML-based AI Detection
Perplexity + Binoculars (Cross-Perplexity)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Detection ML Backend")

# CORS dla Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bollocks-detector.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ULTRA-LIGHT MODELS (fits in 512MB RAM)
logger.info("Loading ultra-light models...")

# Model A: prajjwal1/bert-tiny (~17MB)
tokenizer_a = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model_a = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
model_a.eval()

# Model B: prajjwal1/bert-mini (~45MB)  
tokenizer_b = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
model_b = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-mini")
model_b.eval()

logger.info("Ultra-light models loaded successfully!")
tokenizer_b = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model_b = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
model_b.eval()

logger.info("Models loaded successfully!")

# === Request/Response Models ===
class AnalyzeRequest(BaseModel):
    text: str
    language: str = "en"

class PerplexityResult(BaseModel):
    score: float
    perplexity: float
    mean_logprob: float
    entropy: float
    rank_histogram: Dict[str, float]

class BinocularsResult(BaseModel):
    score: float
    perplexity_a: float
    perplexity_b: float
    ratio: float

class MLAnalysisResponse(BaseModel):
    perplexity: PerplexityResult
    binoculars: BinocularsResult
    combined_score: float

# === Helper Functions ===
def calculate_perplexity(text: str, model, tokenizer) -> Dict:
    """Calculate perplexity using masked language model"""
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            # Get predictions for all positions
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Get actual token log probs
            token_ids = inputs["input_ids"][0]
            token_log_probs = []
            
            for i in range(1, len(token_ids) - 1):  # Skip [CLS] and [SEP]
                token_id = token_ids[i].item()
                log_prob = log_probs[0, i, token_id].item()
                token_log_probs.append(log_prob)
            
            if not token_log_probs:
                return {"perplexity": 1.0, "mean_logprob": 0.0, "entropy": 0.0}
            
            # Calculate metrics
            mean_logprob = float(np.mean(token_log_probs))
            perplexity = float(np.exp(-mean_logprob))
            
            # Entropy (simplified)
            probs = np.exp(token_log_probs)
            entropy = float(-np.sum(probs * token_log_probs) / len(probs))
            
            # Token ranks (simplified - top-k analysis)
            ranks = []
            for i in range(1, min(len(token_ids) - 1, 50)):  # Sample first 50 tokens
                token_id = token_ids[i].item()
                token_logits = logits[0, i, :].numpy()
                sorted_indices = np.argsort(token_logits)[::-1]
                rank = int(np.where(sorted_indices == token_id)[0][0]) + 1
                ranks.append(rank)
            
            # Rank histogram
            rank_bins = [1, 10, 50, 100, 500]
            histogram = {}
            for bin_val in rank_bins:
                histogram[str(bin_val)] = float(np.mean([r <= bin_val for r in ranks]))
            
            return {
                "perplexity": perplexity,
                "mean_logprob": mean_logprob,
                "entropy": entropy,
                "rank_histogram": histogram
            }
            
    except Exception as e:
        logger.error(f"Perplexity calculation error: {e}")
        return {"perplexity": 1.0, "mean_logprob": 0.0, "entropy": 0.0}

def analyze_perplexity(text: str) -> PerplexityResult:
    """Analyze text using Model A (DistilBERT)"""
    metrics = calculate_perplexity(text, model_a, tokenizer_a)
    
    # Convert perplexity to AI probability score
    # Lower perplexity often indicates AI (more predictable)
    # Calibrate this threshold based on your data
    ppl = metrics.get("perplexity", 1.0)
    
    # Sigmoid mapping: perplexity 1-100 -> probability
    # AI tends to have perplexity 10-30, humans 30-80
    normalized_ppl = (ppl - 10) / 70  # normalize to roughly 0-1
    score = 1 / (1 + np.exp(normalized_ppl * 3))  # sigmoid
    
    return PerplexityResult(
        score=float(score),
        perplexity=metrics.get("perplexity", 1.0),
        mean_logprob=metrics.get("mean_logprob", 0.0),
        entropy=metrics.get("entropy", 0.0),
        rank_histogram=metrics.get("rank_histogram", {})
    )

def analyze_binoculars(text: str) -> BinocularsResult:
    """Cross-perplexity analysis using two models"""
    metrics_a = calculate_perplexity(text, model_a, tokenizer_a)
    metrics_b = calculate_perplexity(text, model_b, tokenizer_b)
    
    ppl_a = metrics_a.get("perplexity", 1.0)
    ppl_b = metrics_b.get("perplexity", 1.0)
    
    # Calculate ratio
    ratio = ppl_a / (ppl_b + 1e-8)
    
    # AI text tends to have more consistent perplexity across models
    # Ratio close to 1.0 = consistent = likely AI
    # Ratio far from 1.0 = inconsistent = likely human
    
    # Map ratio to score
    # Calibration: if ratio is 0.8-1.2, likely AI
    deviation = abs(np.log(ratio))  # log-scale deviation from 1.0
    score = 1 / (1 + np.exp(deviation * 2))  # sigmoid
    
    return BinocularsResult(
        score=float(score),
        perplexity_a=float(ppl_a),
        perplexity_b=float(ppl_b),
        ratio=float(ratio)
    )

# === API Endpoints ===
@app.get("/")
def root():
    return {"status": "AI Detection ML Backend", "models": ["DistilBERT", "BERT-tiny"]}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/analyze", response_model=MLAnalysisResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze text using both Perplexity and Binoculars methods
    """
    try:
        text = request.text.strip()
        
        if len(text) < 50:
            raise HTTPException(status_code=400, detail="Text too short (minimum 50 characters)")
        
        if len(text) > 5000:
            text = text[:5000]  # Truncate long texts
        
        logger.info(f"Analyzing text of length {len(text)}")
        
        # Run both analyses
        perplexity_result = analyze_perplexity(text)
        binoculars_result = analyze_binoculars(text)
        
        # Combine scores (weighted average)
        combined = (perplexity_result.score * 0.6) + (binoculars_result.score * 0.4)
        
        return MLAnalysisResponse(
            perplexity=perplexity_result,
            binoculars=binoculars_result,
            combined_score=float(combined)
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
