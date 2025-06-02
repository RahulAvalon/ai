from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize app once
app = FastAPI(title="Disease Intervention Predictor")

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://equist.dataforall.org","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load files
MODEL_PATH = "intervention_model.pkl"
DISEASES_PATH = "known_diseases.csv"
EMBEDDINGS_PATH = "disease_embeddings.npy"
DATA_PATH = "all_interventions_flat.json"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

known_diseases = pd.read_csv(DISEASES_PATH).iloc[:, 0]
disease_embeddings = np.load(EMBEDDINGS_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Models
class DiseaseInput(BaseModel):
    disease_name: str

class InterventionResponse(BaseModel):
    input_disease: str
    matched_disease: str
    similarity_score: float
    interventions: List[str]
    selected_epi: List[str]  # NEW FIELD

# Route
@app.post("/predict", response_model=InterventionResponse)
def predict(data_input: DiseaseInput, threshold: float = 0.05):
    input_disease = data_input.disease_name.strip()
    if not input_disease:
        raise HTTPException(status_code=400, detail="Disease name is required")

    input_embedding = embedder.encode([input_disease], convert_to_tensor=False)
    similarities = cosine_similarity(input_embedding, disease_embeddings).flatten()
    closest_idx = int(np.argmax(similarities))
    similarity_score = float(similarities[closest_idx])

    if similarity_score < threshold:
        raise HTTPException(
            status_code=404,
            detail=f"No close match found (similarity: {similarity_score:.2f})"
        )

    closest_disease = known_diseases.iloc[closest_idx]

    matched_interventions = set()
    selected_epi_set = set()

    for entry in data:
        if entry["disease"].strip().lower() == closest_disease.strip().lower():
            matched_interventions.add(entry["intervention"])
            selected_epi_set.add(entry.get("selectedEpi", "N/A"))

    return InterventionResponse(
        input_disease=input_disease,
        matched_disease=closest_disease,
        similarity_score=similarity_score,
        interventions=sorted(matched_interventions),
        selected_epi=sorted(selected_epi_set)
    )
