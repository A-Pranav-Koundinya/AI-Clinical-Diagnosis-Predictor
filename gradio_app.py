# Gradio App for Symptom-Based Disease Prediction

import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import joblib

# Load model, tokenizer, and label encoder
model = AutoModelForSequenceClassification.from_pretrained("./symptom_diagnosis_model")
tokenizer = AutoTokenizer.from_pretrained("./symptom_diagnosis_model")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
label_encoder = joblib.load("label_encoder.joblib")

def predict_disease(symptoms):
    result = classifier(symptoms)
    class_id = int(result[0]['label'].split('_')[-1])
    disease = label_encoder.inverse_transform([class_id])[0]
    confidence = round(result[0]['score'] * 100, 2)
    return f"Predicted Disease: {disease}\nConfidence: {confidence}%"

# Gradio interface
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Textbox(lines=2, placeholder="e.g. fever, headache, sore throat"),
    outputs="text",
    title="🩺 AI Clinical Diagnosis Predictor",
    description="Enter symptoms separated by commas and get the most likely disease prediction."
)

# Launch the app
demo.launch(share=True)
