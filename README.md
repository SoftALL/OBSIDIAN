# OBSIDIAN Arabic Tweet Classifier

## Table of Contents

- [Project Overview](#project-overview)
- [Task / Objective](#task--objective)
- [Label Definitions](#label-definitions)
- [Folder Structure](#folder-structure)
- [Required Model Files](#required-model-files)
- [Installation Steps](#installation-steps)
- [How to Run Locally](#how-to-run-locally)
- [How to Use Single-Text Mode](#how-to-use-single-text-mode)
- [How to Use Batch Mode](#how-to-use-batch-mode)
- [Preprocessing Notes](#preprocessing-notes)
- [Current Limitations](#current-limitations)
- [Notes on the Original Notebooks](#notes-on-the-original-notebooks)
- [Suggested Git Workflow](#suggested-git-workflow)
- [Status](#status)
- [Future Improvements](#future-improvements)

## Project Overview

A Streamlit-based application for classifying Arabic tweets or short texts using a fine-tuned **AraBERT** model.

The app supports:
- **Single-text classification**
- **Batch classification** from CSV/XLSX files
- **Prediction confidence display**
- **Label distribution visualization**
- **Downloadable batch results**

This project converts Abdullah’s notebook-based OBSIDIAN prototype into a cleaner, GitHub-organized Streamlit application.

The model classifies Arabic tweets or short texts into 5 categories:
- **Threat**
- **Violence**
- **Distress**
- **Complaint**
- **Neutral**

The goal is to provide a more usable interface for testing and demonstrating the trained model without relying directly on the original notebooks.

## Task / Objective

The objective of this project is to:
- organize the original notebook-based work into a clean application structure
- provide a user-friendly Streamlit interface
- support both single-text and batch-file prediction
- make the project easier to run, test, review, and hand off

This repository is intended to be the application version of the OBSIDIAN project, while the original notebooks remain useful as research and development references.

## Label Definitions

### Threat
Text that includes direct or indirect threats, intimidation, or intent to cause harm.

**Example:**  
`سأقتلك إذا رأيتك مرة أخرى`

### Violence
Text describing physical aggression, assault, attack, or violent incidents.

**Example:**  
`قاموا بضرب الرجل في الشارع بعنف شديد`

### Distress
Text expressing fear, panic, emotional suffering, helplessness, or need for help.

**Example:**  
`أنا خائف جدًا ولا أعرف ماذا أفعل، أحتاج مساعدة`

### Complaint
Text expressing dissatisfaction, frustration, criticism, or reporting a service/product issue.

**Example:**  
`الخدمة سيئة جدًا والتطبيق يتعطل كل مرة`

### Neutral
Text that does not strongly indicate threat, violence, distress, or complaint.

**Example:**  
`الجو اليوم معتدل والناس في الحديقة`

## Folder Structure

```text
obsidian-streamlit-app/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── src/
│   ├── inference.py
│   ├── preprocess.py
│   ├── labels.py
│   ├── batch.py
│   └── utils.py
├── model/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── model.safetensors   # kept local, not pushed to GitHub
├── data_samples/
│   └── sample_test.csv
└── outputs/
    └── .gitkeep
```

## Required Model Files

The application requires the trained model files inside the `model/` folder.

Required files:
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`

Important:
- `model.safetensors` is large and is typically kept **local only**
- it is ignored in Git using `.gitignore`
- the app will not perform inference without it

## Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/obsidian-streamlit-app.git
cd obsidian-streamlit-app
```

### 2. Create and activate a virtual environment

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the model files
Place the required model files inside the `model/` folder.

## How to Run Locally

Run the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to Use Single-Text Mode

1. Open the **Single Text** tab
2. Enter one Arabic sentence or tweet
3. Click **Predict**
4. Review:
   - predicted label
   - confidence score
   - probability chart
   - class probability table

### Example inputs

**Threat**  
`سأقتلك إذا رأيتك مرة أخرى`

**Violence**  
`قاموا بضرب الرجل في الشارع بعنف شديد`

**Distress**  
`أنا خائف جدًا ولا أعرف ماذا أفعل، أحتاج مساعدة`

**Complaint**  
`الخدمة سيئة جدًا والتطبيق يتعطل كل مرة`

**Neutral**  
`الجو اليوم معتدل والناس في الحديقة`

## How to Use Batch Mode

1. Open the **Batch Upload** tab
2. Upload a CSV or XLSX file
3. Select the text column to classify
4. Click **Run Batch Prediction**
5. Review:
   - uploaded data preview
   - selected text column preview
   - progress bar during prediction
   - results preview
   - predicted label distribution chart
6. Download the full output as CSV

### Supported text column names
The app detects these names case-insensitively:
- `cleaned_text`
- `text`
- `tweet`
- `tweet_text`
- `content`

So columns like `Text` and `text` are both supported.

### Sample test file
A small sample file is included for quick testing:

```text
data_samples/sample_test.csv
```

## Preprocessing Notes

The app uses lightweight preprocessing to stay consistent with the notebook workflow and trained model assumptions.

Current preprocessing includes:
- handling missing values
- converting values to strings
- trimming whitespace
- collapsing repeated spaces

The app avoids aggressive extra cleaning so that the input remains consistent with the trained tokenizer/model setup.

Inference is aligned with the trained model by using the appropriate token length setting.

## Current Limitations

- The app depends on a local copy of `model.safetensors`
- The model file is large and is not included in the public repo by default
- Batch prediction is currently done row by row, so large files may take noticeable time
- Preview tables intentionally show only a subset of rows:
  - uploaded data preview: first 10 rows
  - selected text preview: first 5 rows
  - results preview: first 20 rows
- Some classes may overlap semantically in difficult examples, especially between **Threat** and **Distress**
- Deployment to a public Streamlit service may require a separate strategy for hosting the model weights

## Notes on the Original Notebooks

The original notebooks were used as development and reference material for:
- training workflow
- inference logic
- dataset usage
- model configuration

This Streamlit project is the cleaned application layer built on top of that work.

## Suggested Git Workflow

After making changes, use:

```bash
git add .
git commit -m "Describe your change"
git push
```

Example commit messages:
- `Add README with setup and usage instructions`
- `Improve batch upload experience`
- `Polish Streamlit UI`

## Status

Current status:
- model loading works
- single-text prediction works
- batch prediction works
- progress bar works
- sample file included
- UI has been polished for easier use

## Future Improvements

Possible next improvements:
- faster batch inference
- manual selection from all columns, not only likely text columns
- richer analytics for batch results
- deployment strategy for model hosting
- stronger README deployment section once final hosting approach is decided
