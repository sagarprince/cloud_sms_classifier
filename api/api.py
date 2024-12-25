import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Load the model and vectorizer
model = joblib.load('sms_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Create the FastAPI app
app = FastAPI()

# Define the input data format
class SMSInput(BaseModel):
    text: str
    
class SMSBatchInput(BaseModel):
    messages: List[SMSInput]

# Define the prediction endpoint
@app.post("/predict-message-type")
def predict_sms(data: SMSInput):
    sms_vectorized = vectorizer.transform([data.text])
    prediction = model.predict(sms_vectorized)[0]
    return {"prediction": prediction}

@app.post("/predict-messages-type")
def predict_sms(data: SMSBatchInput):
    sms_texts = [sms.text for sms in data.messages]
    sms_vectorized = vectorizer.transform(sms_texts)
    predictions = model.predict(sms_vectorized)
    
    result = [{"sms": sms_text, "type": pred} 
              for sms_text, pred in zip(sms_texts, predictions)]
    
    return result
