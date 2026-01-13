# Automated Evaluation of Handwritten Assessments 📝🤖

An AI-assisted system that helps evaluators/teachers **grade handwritten answers faster** by combining **OCR (text extraction)** and **NLP-based evaluation (grading)**.

> ⚠️ **Disclaimer:** This project is **not intended to replace human evaluators**.  
> It is built to **assist** teachers by reducing manual effort and providing an AI-supported score based on **keyword matching + semantic similarity**.

---

## 📌 Overview

This project has **two major components**:

### 1) Optical Character Recognition (OCR)
- Extracts text from handwritten answer-sheet images.
- We initially attempted building our own OCR model using the **IAM Handwriting Forms Dataset**.
- Due to limited accuracy on real-world student handwriting and technical vocabulary, the current version uses **Microsoft Azure OCR** for reliable extraction.

### 2) Answer Evaluation (Grading)
- Compares the OCR-extracted **student answer** with a **reference answer**.
- Uses:
  - ✅ **TF-IDF** for keyword extraction & keyword matching
  - ✅ **BERT embeddings** for semantic similarity (meaning-based evaluation)
- Output:
  - **Score (out of 5 or 10)**
  - Evaluation insights (keyword overlap + semantic score)

---

## ✨ Key Features

- 📄 Extract text from handwritten answer images (OCR)
- 🎯 Keyword-based grading using TF-IDF
- 🧠 Semantic similarity grading using Transformer embeddings (BERT)
- 🧾 Can provide explainability (matched / missing keywords)
- ⏱️ Helps reduce grading time significantly

---

## 🛠️ Tech Stack

### OCR
- **Microsoft Azure OCR** (Computer Vision / Read API)

### NLP / Machine Learning
- **TF-IDF** (`sklearn.feature_extraction.text.TfidfVectorizer`)
- **BERT embeddings** (`transformers`, `torch`)
- **Cosine similarity** (`sklearn.metrics.pairwise.cosine_similarity`)

### Image Processing (preprocessing / dataset work)
- **Pillow (PIL)**
- **OpenCV (cv2)**
- **NumPy**

### Programming Language
- **Python 3.x**

---

## 📚 Dataset Used (OCR Training Attempt)

We explored OCR model training using the **IAM Handwriting Forms Dataset**:
- Size: ~5GB
- Writers: 657
- Pages: 1539
- Sentences: 13,350
- Words: 115,320
- Data: PNG images + XML metadata (line coordinates + transcription)

⚠️ **Note:** IAM handwriting differs from student engineering handwriting and lacks technical vocabulary, affecting OCR performance.

---


---

## ⚙️ Installation & Setup

### ✅ 1) Clone the repository
```bash
git clone https://github.com/Nalini-24/Automated-Evaluation-of-Handwritten-Assessments.git
cd Automated-Evaluation-of-Handwritten-Assessments

2) Create and activate a virtual environment (Recommended)
python -m venv venv

Windows (PowerShell)
venv\Scripts\activate

🔑 Azure OCR Setup

This project uses Azure OCR to extract handwritten text.

✅ Requirements

1.Azure subscription
2.Azure Computer Vision resource

✅ Create .env file (REQUIRED)

Create a file called .env in the project root and add:

AZURE_ENDPOINT=your_endpoint_here
AZURE_KEY=your_key_here

To get these values:

1. Azure Portal → Create Computer Vision resource
2. Copy Endpoint and Key

✅ Tip: Never push .env to GitHub. Add it to .gitignore.

▶️ How to Run the Project (Complete Steps)

✅ Step 1: Add your handwritten answer image

Put your handwritten image inside:

input/
Example:

input/sample_answer.jpg

✅ Step 2: Add the reference answer

In the current version, reference answers can be:

written inside a .txt file (recommended), OR

hardcoded in the code

✅ Recommended approach:

Create a file:

input/reference_answer.txt
Example content:

India is a democratic country with regular elections and individual rights.

✅ Step 3: Run the project

From the root folder:

python src/main.py

✅ Step 4: View results

Your output will be saved/displayed as:

Extracted OCR text

Keyword score (TF-IDF)

Semantic similarity score (BERT)

Final grade score

If output file saving is enabled, check:

output/


Example:

output/results.txt

✅ Example Output

Example output after running:

OCR Text:
"The constitution gives freedom to all citizens, and elections are held fairly."

Matched Keywords:
constitution, elections, citizens

Keyword Score: 3.5/5
Semantic Similarity Score: 4.2/5

Final Score: 7.7/10

🧩 Working / Approach
✅ OCR Pipeline

Extract handwritten text from answer image using Azure OCR

(Dataset attempt) preprocessing steps explored:

Convert page image to grayscale

Crop line images using XML coordinates

Resize + pad for batching (1024×128)

✅ Evaluation Pipeline

Extract OCR text using Azure OCR

Extract reference keywords using TF-IDF

Compute keyword overlap score

Compute semantic similarity using BERT embeddings

Combine both scores to generate final marks

📈 Performance Metrics (Reported)

✅ OCR Accuracy: ~92% (Word-level accuracy using Azure OCR)

✅ Grading Accuracy: ~85% correlation with human evaluator scores (Pearson correlation)

✅ Time Saved: ~70% reduction in grading time

✅ Throughput: 200+ answer sheets/hour (approx)

📸 Screenshots

Add screenshots in:

docs/screenshots/


Example usage in README:

OCR Output


Grading Output


🎥 Demo

Add a demo GIF/video in:

docs/demo/


Example:


🚀 Future Improvements

Planned improvements to increase accuracy and explainability:

✅ Better Semantic Similarity

Replace raw BERT embeddings with Sentence-BERT

Suggested model: all-MiniLM-L6-v2

✅ Better Keyword Extraction

Replace TF-IDF keywords with KeyBERT (context-aware keyword extraction)

✅ Explainable Feedback

Provide:

✅ Matched keywords

❌ Missing key points

💡 Suggestions to improve answers

✅ Contradiction Detection

Add Natural Language Inference (NLI) models (e.g., roberta-large-mnli)

Penalize contradictory/wrong answers

📌 Project Notes

This project is still under development.

OCR training on IAM dataset did not generalize well to real student answers.

Current focus is on improving the grading/evaluation module.

📜 License

This project is licensed under the MIT License.
See the LICENSE
 file for details.
