import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_SOURCE = "SoftALL/OBSIDIAN"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
    model.eval()
    return tokenizer, model


def predict_text(text, tokenizer, model, max_length=128):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    label = model.config.id2label[pred_id]
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            model.config.id2label[i]: float(probs[i].item())
            for i in range(len(probs))
        },
    }