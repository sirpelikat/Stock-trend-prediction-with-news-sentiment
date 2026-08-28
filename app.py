import os
import re
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import TweetTokenizer
import gensim.downloader as api
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yfinance as yf

# Global variables for models and data
glove_model = None
keras_model = None
tokenizer = TweetTokenizer()
max_len = 30
embedding_dim = 50
label2idx = {"downtrend": 0, "flat": 1, "uptrend": 2}
idx2label = {0: "downtrend", 1: "flat", 2: "uptrend"}

def load_all_artifacts():
    global glove_model, keras_model
    if glove_model is None:
        print("Loading GloVe embeddings (glove-wiki-gigaword-50)...")
        glove_model = api.load("glove-wiki-gigaword-50")
        print("GloVe embeddings loaded.")
    
    if keras_model is None:
        model_path = os.path.join("model_artifacts", "lstm_model.keras")
        if os.path.exists(model_path):
            print(f"Loading trained Keras model from {model_path}...")
            keras_model = tf.keras.models.load_model(model_path)
            print("Keras LSTM model loaded.")
        else:
            print("Model file not found. Please run train_and_save_model.py first.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_artifacts()
    yield

app = FastAPI(title="Stock Trend Prediction with News Sentiment Dashboard", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Text cleaner function identical to notebook
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Convert text to padded GloVe embedding sequences
def text_to_embedding_vector(text: str):
    cleaned = clean_text(text)
    tokens = tokenizer.tokenize(cleaned)
    vectors = []
    token_details = []
    
    for tok in tokens:
        in_vocab = tok in glove_model
        token_details.append({
            "token": tok,
            "in_vocab": in_vocab
        })
        if in_vocab:
            vectors.append(glove_model[tok])
        else:
            # Reproducible pseudo-random for OOV
            np.random.seed(abs(hash(tok)) % (2**32))
            vectors.append(np.random.normal(scale=0.6, size=embedding_dim))
            
    if len(vectors) == 0:
        vectors.append(np.zeros(embedding_dim))
        token_details.append({"token": "<EMPTY>", "in_vocab": False})
        
    padded = pad_sequences([np.array(vectors)], maxlen=max_len, dtype="float32", padding="post", truncating="post")
    return padded, token_details, cleaned

class PredictRequest(BaseModel):
    text: str
    company: Optional[str] = ""

@app.post("/api/predict")
def predict_trend(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    if keras_model is None or glove_model is None:
        load_all_artifacts()
    
    padded_seq, tokens, cleaned_text = text_to_embedding_vector(req.text)
    prob = keras_model.predict(padded_seq, verbose=0)[0]
    
    pred_idx = int(np.argmax(prob))
    pred_label = idx2label[pred_idx]
    
    # Probabilities
    prob_dict = {
        "downtrend": float(prob[0]),
        "flat": float(prob[1]),
        "uptrend": float(prob[2])
    }
    
    # Verdict calculation
    confidence = float(np.max(prob))
    if pred_label == "uptrend":
        signal = "BULLISH 🚀"
        badge_color = "emerald"
        sentiment_summary = "Positive market outlook inferred from headline content."
    elif pred_label == "downtrend":
        signal = "BEARISH 📉"
        badge_color = "rose"
        sentiment_summary = "Negative market sentiment inferred from headline content."
    else:
        signal = "NEUTRAL ⚖️"
        badge_color = "amber"
        sentiment_summary = "Stable / low volatility expected based on news phrasing."
        
    return {
        "text": req.text,
        "cleaned_text": cleaned_text,
        "prediction": pred_label,
        "signal": signal,
        "badge_color": badge_color,
        "confidence": confidence,
        "probabilities": prob_dict,
        "sentiment_summary": sentiment_summary,
        "tokens": tokens,
        "token_count": len(tokens)
    }

@app.get("/api/hmm")
def get_hmm_data(observation_type: str = Query("volatility", pattern="^(volatility|day_of_week|title_length)$")):
    try:
        train_df = pd.read_csv("train_data_final.csv")
    except Exception:
        raise HTTPException(status_code=500, detail="train_data_final.csv not found.")
    
    # 1. Start probabilities (pi)
    pi = train_df["Trend"].value_counts(normalize=True).to_dict()
    
    # 2. Transition matrix (A)
    train_df["Next_Trend"] = train_df["Trend"].shift(-1)
    trans = pd.crosstab(train_df["Trend"], train_df["Next_Trend"], normalize="index").fillna(0)
    
    # Format transition matrix
    states = ["downtrend", "flat", "uptrend"]
    A_matrix = {}
    for s in states:
        A_matrix[s] = {}
        for s2 in states:
            val = float(trans.loc[s, s2]) if (s in trans.index and s2 in trans.columns) else 0.0
            A_matrix[s][s2] = round(val, 4)
            
    # 3. Emission matrix (B)
    if observation_type == "day_of_week":
        obs_col = "Day_of_Week"
        emission_raw = pd.crosstab(train_df["Trend"], train_df[obs_col], normalize="index").fillna(0)
    elif observation_type == "title_length":
        train_df["Title_Length_Category"] = pd.qcut(train_df["Title_Word_Count"], q=3, labels=["Short", "Medium", "Long"])
        obs_col = "Title_Length_Category"
        emission_raw = pd.crosstab(train_df["Trend"], train_df[obs_col], normalize="index").fillna(0)
    else: # Volatility / Relative change magnitude
        def categorize_vol(change):
            abs_c = abs(change)
            if abs_c > 0.15:
                return "High Volatility"
            elif abs_c > 0.05:
                return "Medium Volatility"
            else:
                return "Low Volatility"
        train_df["Vol_Category"] = train_df["Relative_Change"].apply(categorize_vol)
        obs_col = "Vol_Category"
        emission_raw = pd.crosstab(train_df["Trend"], train_df[obs_col], normalize="index").fillna(0)
        
    obs_symbols = list(emission_raw.columns)
    B_matrix = {}
    for s in states:
        B_matrix[s] = {}
        for o in obs_symbols:
            val = float(emission_raw.loc[s, o]) if (s in emission_raw.index and o in emission_raw.columns) else 0.0
            B_matrix[s][str(o)] = round(val, 4)
            
    return {
        "states": states,
        "observations": obs_symbols,
        "observation_type": observation_type,
        "start_probabilities": {k: round(float(v), 4) for k, v in pi.items()},
        "transition_matrix": A_matrix,
        "emission_matrix": B_matrix
    }

class ViterbiRequest(BaseModel):
    observations: List[str]
    observation_type: Optional[str] = "volatility"

@app.post("/api/hmm/viterbi")
def compute_viterbi(req: ViterbiRequest):
    hmm = get_hmm_data(observation_type=req.observation_type)
    states = hmm["states"]
    pi = hmm["start_probabilities"]
    A = hmm["transition_matrix"]
    B = hmm["emission_matrix"]
    obs_seq = req.observations

    if not obs_seq:
        raise HTTPException(status_code=400, detail="Observation sequence cannot be empty.")
    
    # Check invalid observations
    valid_obs = set(hmm["observations"])
    for o in obs_seq:
        if o not in valid_obs:
            raise HTTPException(status_code=400, detail=f"Invalid observation '{o}'. Valid options: {list(valid_obs)}")

    # Viterbi Algorithm
    V = [{}]
    path = {}

    # Initialize base cases (t = 0)
    first_obs = obs_seq[0]
    for s in states:
        emit_p = B.get(s, {}).get(first_obs, 1e-6)
        start_p = pi.get(s, 1e-6)
        V[0][s] = start_p * emit_p
        path[s] = [s]

    # Run Viterbi for t > 0
    for t in range(1, len(obs_seq)):
        V.append({})
        newpath = {}
        current_obs = obs_seq[t]

        for cur_state in states:
            emit_p = B.get(cur_state, {}).get(current_obs, 1e-6)
            (prob, prev_state) = max(
                (V[t - 1][p_state] * A.get(p_state, {}).get(cur_state, 1e-6) * emit_p, p_state)
                for p_state in states
            )
            V[t][cur_state] = prob
            newpath[cur_state] = path[prev_state] + [cur_state]

        path = newpath

    # Best final state
    (best_prob, best_last_state) = max((V[-1][s], s) for s in states)
    best_path = path[best_last_state]

    return {
        "observation_sequence": obs_seq,
        "most_likely_states": best_path,
        "path_probability": float(best_prob),
        "step_details": [
            {
                "step": t + 1,
                "observation": obs_seq[t],
                "assigned_state": best_path[t],
                "state_probabilities": {s: float(V[t][s]) for s in states}
            }
            for t in range(len(obs_seq))
        ]
    }

@app.get("/api/metrics")
def get_model_metrics():
    metrics_path = os.path.join("model_artifacts", "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found. Run train_and_save_model.py first.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/dataset-stats")
def get_dataset_stats():
    summary_path = os.path.join("model_artifacts", "dataset_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    try:
        df = pd.read_csv("train_data_final.csv")
        changes = df["Relative_Change"].dropna().values
        hist_counts, bin_edges = np.histogram(changes, bins=25, range=(-0.5, 0.5))
        hist_data = {
            "counts": hist_counts.tolist(),
            "bins": [round(float(b), 3) for b in bin_edges[:-1]]
        }
        
        raw_dist = df["Trend"].value_counts().to_dict()
        max_c = max(raw_dist.values()) if raw_dist else 0
        smote_dist = {k: max_c for k in raw_dist.keys()}
        top_companies = df["Name"].value_counts().head(10).to_dict() if "Name" in df.columns else {}
    except Exception:
        hist_data = {"counts": [], "bins": []}
        raw_dist = {}
        smote_dist = {}
        top_companies = {}

    return {
        "summary": summary,
        "price_histogram": hist_data,
        "class_distributions": {
            "before_smote": raw_dist,
            "after_smote": smote_dist
        },
        "top_companies": top_companies
    }

@app.get("/api/dataset-sample")
def get_dataset_sample(
    split: str = Query("train", pattern="^(train|val|test|raw)$"),
    trend: str = Query("all", pattern="^(all|uptrend|flat|downtrend)$"),
    limit: int = 15
):
    filename = f"{split}_data_final.csv" if split != "raw" else "stock_trend.csv"
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
    
    df = pd.read_csv(filename)
    if "Trend" in df.columns and trend != "all":
        df = df[df["Trend"] == trend]
        
    sample = df.head(limit).fillna("").to_dict(orient="records")
    return {
        "split": split,
        "total_records": len(df),
        "columns": df.columns.tolist(),
        "sample": sample
    }

@app.get("/api/live-ticker")
def get_live_ticker(symbol: str = "AAPL"):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        last_price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)
        
        if last_price is not None and prev_close is not None:
            pct_change = ((last_price - prev_close) / prev_close) * 100
        else:
            pct_change = 0.0
            
        raw_news = ticker.news or []
        analyzed_news = []
        
        for item in raw_news[:6]:
            title = item.get("title")
            if not title and isinstance(item.get("content"), dict):
                title = item.get("content", {}).get("title")
            if not title:
                continue
                
            publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName", "Finance News")
            link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url", "#")
            
            if keras_model is not None and glove_model is not None:
                padded_seq, _, _ = text_to_embedding_vector(title)
                prob = keras_model.predict(padded_seq, verbose=0)[0]
                pred_idx = int(np.argmax(prob))
                pred_label = idx2label[pred_idx]
                conf = float(np.max(prob))
            else:
                pred_label = "flat"
                conf = 0.5
                
            analyzed_news.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "prediction": pred_label,
                "confidence": conf
            })
            
        return {
            "symbol": symbol.upper(),
            "price": round(float(last_price), 2) if last_price else None,
            "change_pct": round(float(pct_change), 2),
            "news_count": len(analyzed_news),
            "news": analyzed_news
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

