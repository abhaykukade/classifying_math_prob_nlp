import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import os

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_DIR = "output/model-best"
LABELS = ['Algebra', 'Geometry', 'Calculus', 'Statistics', 'Number_theory', 'Combinatorics', 'Linear_Algebra', 'Abstract_Algebra']

# Step 1: Load CSV
df = pd.read_csv(TRAIN_PATH)
df = df.dropna()
df['label'] = df['label'].astype(int)

# Step 2: Train/Dev split
train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

# Step 3: Convert to spaCy format
def convert_to_spacy(df, output_path):
    nlp = spacy.blank("en")
    db = DocBin()
    for _, row in df.iterrows():
        doc = nlp.make_doc(row["Question"])
        cats = {label: 0.0 for label in LABELS}
        cats[LABELS[row["label"]]] = 1.0
        doc.cats = cats
        db.add(doc)
    db.to_disk(output_path)

os.makedirs("data", exist_ok=True)
convert_to_spacy(train_df, "data/train.spacy")
convert_to_spacy(dev_df, "data/dev.spacy")

# Step 4: Prepare test data for prediction
def predict_on_test():
    test_df = pd.read_csv(TEST_PATH)
    test_df = test_df.dropna()
    nlp = spacy.load(MODEL_DIR)

    predictions = []
    for i, row in test_df.iterrows():
        doc = nlp(row["Question"])
        pred = max(doc.cats.items(), key=lambda x: x[1])[0]
        label_id = LABELS.index(pred)
        predictions.append((i, label_id))

    # Save predictions
    pd.DataFrame(predictions, columns=["id", "label"]).to_csv("test_predictions.csv", index=False)
    print("✅ Predictions saved to test_predictions.csv")

# Step 5: Run predictions after training
if os.path.exists(MODEL_DIR):
    predict_on_test()
else:
    print("⚠️ Model not found. Train the model first using: python -m spacy train config.cfg --cpu")
