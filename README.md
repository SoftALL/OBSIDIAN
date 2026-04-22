# OBSIDIAN

## Table of Contents

- [Project Overview](#project-overview)
- [Objective](#objective)
- [Current Architecture](#current-architecture)
- [Repositories and Hosting](#repositories-and-hosting)
- [Label Definitions](#label-definitions)
- [Project Structure](#project-structure)
- [Model Hosting Details](#model-hosting-details)
- [Installation](#installation)
- [How to Run Locally](#how-to-run-locally)
- [How to Use the App](#how-to-use-the-app)
  - [Single Text Mode](#single-text-mode)
  - [Batch Upload Mode](#batch-upload-mode)
- [Sample Test File](#sample-test-file)
- [Preprocessing and Inference Notes](#preprocessing-and-inference-notes)
- [Original Notebook Context](#original-notebook-context)


## Project Overview

**OBSIDIAN** is a Streamlit-based application for classifying Arabic tweets or short texts using a fine-tuned **AraBERT** model.

The app supports:
- **Single-text classification**
- **Batch classification** from CSV/XLSX files
- **Prediction confidence display**
- **Class probability visualization**
- **Batch result download as CSV**

This project turns Abdullah's notebook-based Colab work into a cleaner, organized application that can be reviewed, tested, and deployed more easily.

The model classifies Arabic text into **5 categories**:
- **Threat**
- **Violence**
- **Distress**
- **Complaint**
- **Neutral**

## Objective

The goal of this project is to:
- convert the original Colab prototype into a clean GitHub project
- provide a user-friendly Streamlit interface
- support both single-text and batch prediction workflows
- separate code, model hosting, and deployment more clearly
- make the project easier to test, demonstrate, and hand off

## Current Architecture

The current deployed-style setup is:

- **GitHub** hosts the application code
- **Hugging Face** hosts the fine-tuned model files
- **Streamlit** runs the user interface
- The app loads the model from the Hugging Face model repo:
  - `SoftALL/OBSIDIAN`

This means the application no longer depends on a local `model.safetensors` file inside the GitHub repository.

## Repositories and Hosting

### GitHub repository
- Organization: **SoftALL**
- Repository: **OBSIDIAN**

### Hugging Face model repository
- Organization: **SoftALL**
- Model repository: **OBSIDIAN**

The Hugging Face model repo is used to host the model assets required for inference.

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
Text expressing dissatisfaction, frustration, criticism, or reporting a service or product issue.

**Example:**  
`الخدمة سيئة جدًا والتطبيق يتعطل كل مرة`

### Neutral
Text that does not strongly indicate threat, violence, distress, or complaint.

**Example:**  
`الجو اليوم معتدل والناس في الحديقة`

## Project Structure

```text
OBSIDIAN/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── obsidian_logo.png
│   └── obsidian_banner.jpeg
├── src/
│   ├── inference.py
│   ├── preprocess.py
│   ├── labels.py
│   ├── batch.py
│   └── utils.py
├── model/
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── data_samples/
│   └── sample_test.csv
└── outputs/
    └── .gitkeep
```

## Model Hosting Details

The fine-tuned model is hosted on Hugging Face in:

- `SoftALL/OBSIDIAN`

The Hugging Face model repo contains the following model assets:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

Important notes:
- the large `model.safetensors` file is **not tracked** inside the GitHub application repo
- the application loads the model from Hugging Face using Transformers `from_pretrained()`
- the local `model/` folder in this repo only keeps small configuration/tokenizer files for project organization and reference

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SoftALL/OBSIDIAN.git
cd OBSIDIAN
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

## How to Run Locally

Run the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### Important
Since the app loads the model from Hugging Face, you need:
- an internet connection on first run
- enough time for the model to download/cache the first time

After the model is cached, later runs are usually faster on the same machine.

## How to Use the App

### Single Text Mode

1. Open the **Single Text** tab
2. Enter one Arabic sentence or tweet
3. Click **Predict**
4. Review:
   - predicted label
   - confidence score
   - probability chart
   - class probability table

#### Example inputs

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

### Batch Upload Mode

1. Open the **Batch Upload** tab
2. Upload a CSV or XLSX file
3. Select the text column to classify
4. Click **Run Batch Prediction**
5. Review:
   - uploaded data preview
   - selected text column preview
   - progress indicator
   - classified result preview
   - predicted label distribution chart
6. Download the full output as CSV

#### Supported text column names

The app detects these names case-insensitively:
- `cleaned_text`
- `text`
- `tweet`
- `tweet_text`
- `content`

So columns like `Text` and `text` are both supported.

## Sample Test File

A small sample file is included for quick testing:

```text
data_samples/sample_test.csv
```

This file can be used directly in the **Batch Upload** tab.

## Preprocessing and Inference Notes

The app uses lightweight preprocessing to stay consistent with the original notebook workflow and trained model assumptions.

Current preprocessing includes:
- handling missing values
- converting values to strings
- trimming whitespace
- collapsing repeated spaces

The app avoids aggressive extra cleaning so the text stays closer to the model's expected input style.

Inference is aligned with the trained setup by using:
- the uploaded Hugging Face model repo
- `max_length=128`

## Original Notebook Context

The original notebooks were used as reference material for:
- training workflow
- dataset usage
- model configuration
- prototype inference logic

This repository is the cleaned application layer built from that earlier notebook-based work.
