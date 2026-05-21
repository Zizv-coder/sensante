# api/main.py
# API FastAPI pour SenSante - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP / UCAD

from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np


# --- Schemas Pydantic ---

class PatientInput(BaseModel):
    """Donnees d'entree : les symptomes d'un patient."""
    age         : int   = Field(..., ge=0,    le=120,  description="Age en annees")
    sexe        : str   = Field(...,                   description="Sexe : M ou F")
    temperature : float = Field(..., ge=35.0, le=42.0, description="Temperature en Celsius")
    tension_sys : int   = Field(..., ge=60,   le=250,  description="Tension systolique")
    toux        : bool  = Field(...,                   description="Presence de toux")
    fatigue     : bool  = Field(...,                   description="Presence de fatigue")
    maux_tete   : bool  = Field(...,                   description="Presence de maux de tete")
    region      : str   = Field(...,                   description="Region du Senegal")


class DiagnosticOutput(BaseModel):
    """Donnees de sortie : le resultat du diagnostic."""
    diagnostic  : str   = Field(..., description="Diagnostic predit")
    probabilite : float = Field(..., description="Probabilite du diagnostic")
    confiance   : str   = Field(..., description="Niveau de confiance")
    message     : str   = Field(..., description="Recommandation")


# --- Creer l'application ---

app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

# --- Autoriser les requetes depuis le frontend ---  ← CORRIGE : au niveau module

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # En dev : tout accepter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Charger le modele et les encodeurs au demarrage ---

print("Chargement du modele ...")
model        = joblib.load("models/model.pkl")
le_sexe      = joblib.load("models/encoder_sexe.pkl")
le_region    = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

label_map = {
    0: "grippe",
    1: "palu",
    2: "sain",
    3: "typh"
}

print(f"Modele charge : {type(model).__name__}")
print(f"Classes : {[label_map[i] for i in model.classes_]}")


# --- Route GET /health ---

@app.get("/health")
def health_check():
    """Verification de l'etat de l'API."""
    return {
        "status"  : "ok",
        "message" : "SenSante API is running"
    }


# --- Route GET /model-info ---

@app.get("/model-info")
def model_info():
    """Informations sur le modele charge."""
    return {
        "type"         : type(model).__name__,
        "n_arbres"     : model.n_estimators,
        "classes"      : [label_map[i] for i in model.classes_],
        "n_features"   : model.n_features_in_,
        "feature_cols" : list(feature_cols)
    }


# --- Route POST /predict ---

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """
    Predire un diagnostic a partir des symptomes d'un patient.
    Recoit les symptomes en JSON, renvoie le diagnostic,
    la probabilite et une recommandation.
    """
    # 1. Encoder les variables categoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 3. Predire
    diagnostic = label_map[model.predict(features)[0]]
    probas     = model.predict_proba(features)[0]
    proba_max  = float(probas.max())

    # 4. Determiner le niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 5. Generer la recommandation
    messages = {
        "palu"  : "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typh"  : "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain"  : "Pas de pathologie detectee. Continuez a surveiller."
    }

    # 6. Renvoyer le resultat
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )