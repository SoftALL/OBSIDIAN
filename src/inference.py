import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_SOURCE = "SoftALL/OBSIDIAN"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
    model.eval()
    return tokenizer, model


def _get_device(model):
    """
    Returns the device where the model is currently located.
    This keeps inference compatible with both CPU and GPU.
    """
    return next(model.parameters()).device


def predict_text(text, tokenizer, model, max_length=128):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    device = _get_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

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


def predict_batch_texts(texts, tokenizer, model, max_length=128, batch_size=32):
    """
    Predict labels for multiple texts more efficiently than calling predict_text()
    one by one.

    Returns a list of dictionaries in the same general format as predict_text():
    [
        {
            "label": "Threat",
            "confidence": 0.91,
            "probabilities": {
                "Threat": 0.91,
                "Violence": 0.03,
                ...
            }
        },
        ...
    ]
    """
    if texts is None:
        return []

    texts = ["" if text is None else str(text) for text in texts]

    if len(texts) == 0:
        return []

    device = _get_device(model)
    results = []

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs_batch = torch.softmax(outputs.logits, dim=1)

        for probs in probs_batch:
            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())
            label = model.config.id2label[pred_id]

            results.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "probabilities": {
                        model.config.id2label[i]: float(probs[i].item())
                        for i in range(len(probs))
                    },
                }
            )

    return results