import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("../data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

le_sexe = LabelEncoder()
le_region = LabelEncoder()
le_diagnostic = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = le_diagnostic.fit_transform(df['diagnostic'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Classes : {list(le_diagnostic.classes_)}")

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': le_diagnostic.inverse_transform(y_test[:10]),
    'Prediction': le_diagnostic.inverse_transform(y_pred[:10])
})
print(comparison)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=le_diagnostic.classes_))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_diagnostic.classes_,
            yticklabels=le_diagnostic.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee dans figures/confusion_matrix.png")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

joblib.dump(le_sexe,       "models/encoder_sexe.pkl")
joblib.dump(le_region,     "models/encoder_region.pkl")
joblib.dump(le_diagnostic, "models/encoder_diagnostic.pkl")  # ✅ ajouté
joblib.dump(feature_cols,  "models/feature_cols.pkl")
print("Encodeurs et metadata sauvegardes.")

# --- Chargement ---
model_loaded         = joblib.load("models/model.pkl")
le_sexe_loaded       = joblib.load("models/encoder_sexe.pkl")
le_region_loaded     = joblib.load("models/encoder_region.pkl")
le_diagnostic_loaded = joblib.load("models/encoder_diagnostic.pkl")  # ✅ ajouté
feature_cols_loaded  = joblib.load("models/feature_cols.pkl")

print(type(model_loaded).__name__)
print(list(le_diagnostic_loaded.classes_))  # ✅ noms, pas entiers

# --- Nouveau patient ---
nouveau_patient = {
    'age': 28, 
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True, 
    'fatigue': True,
    'maux_tete': True, 
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [[
    nouveau_patient['age'], 
    sexe_enc, 
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'], 
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']), 
    int(nouveau_patient['maux_tete']),
    region_enc
]]

# ✅ DataFrame pour éviter le UserWarning
features_df = pd.DataFrame(features, columns=feature_cols_loaded)

diagnostic_code = model_loaded.predict(features_df)[0]
diagnostic_nom = le_diagnostic_loaded.inverse_transform([diagnostic_code])[0]  # ✅ décodé
probas = model_loaded.predict_proba(features_df)[0]
proba_max = probas.max()

print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic_nom}")   # ✅ nom, pas entier
print(f"Probabilite : {proba_max:.1%}")

print(f"\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    nom = le_diagnostic_loaded.inverse_transform([classe])[0]  # ✅ décode l'entier
    bar = '#' * int(proba * 30)
    print(f"  {nom:12s} : {proba:.1%} {bar}")  # ✅ :12s sur string, pas int