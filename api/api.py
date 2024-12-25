import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model and vectorizer
model = joblib.load('sms_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Create the FastAPI app
app = FastAPI()

# Define the input data format
class SMSInput(BaseModel):
    text: str

# Define the prediction endpoint
@app.post("/predict")
def predict_sms(data: SMSInput):
    sms_vectorized = vectorizer.transform([data.text])
    prediction = model.predict(sms_vectorized)[0]
    return {"prediction": prediction}
