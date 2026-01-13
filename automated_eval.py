import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import tkinter as tk
from tkinter import filedialog, Text

def recognize_text_azure(image_path, subscription_key, endpoint):
    client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    with open(image_path, "rb") as image:
        response = client.read_in_stream(image, raw=True)
    operation_id = response.headers["Operation-Location"].split("/")[-1]
    while True:
        result = client.get_read_result(operation_id)
        if result.status in [OperationStatusCodes.succeeded, OperationStatusCodes.failed]:
            break
        time.sleep(1)
    if result.status == OperationStatusCodes.succeeded:
        extracted_text = []
        for page in result.analyze_result.read_results:
            for line in page.lines:
                extracted_text.append(line.text)
        return " ".join(extracted_text)  
    else:
        return "No text detected."
    

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def extract_keywords(reference_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([reference_text])
    keywords = vectorizer.get_feature_names_out()
    return set(keywords)


def compute_keyword_score(student_text, reference_keywords):
    student_words = set(student_text.split())
    matched_keywords = reference_keywords & student_words
    keyword_score = len(matched_keywords) / len(reference_keywords) if reference_keywords else 0
    return keyword_score * 5


def compute_semantic_similarity(reference_text, student_text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    inputs_reference = tokenizer(reference_text, return_tensors="pt", padding=True, truncation=True)
    inputs_student = tokenizer(student_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        reference_output = model(**inputs_reference)
        student_output = model(**inputs_student)
    reference_embedding = reference_output.last_hidden_state.mean(dim=1)
    student_embedding = student_output.last_hidden_state.mean(dim=1)
    similarity = cosine_similarity(reference_embedding, student_embedding)
    return similarity[0][0] * 5


def evaluate_answer(detected_text, reference_text):
    detected_text = preprocess_text(detected_text)
    reference_text = preprocess_text(reference_text)
    reference_keywords = extract_keywords(reference_text)
    print(reference_keywords)
    keyword_score = compute_keyword_score(detected_text, reference_keywords)
    semantic_score = compute_semantic_similarity(reference_text, detected_text)
    total_score = min(10, keyword_score + semantic_score)
    return {
        "keyword_score": keyword_score,
        "semantic_score": semantic_score,
        "total_score": total_score
    }
'''Cluster Computing Scalability: Moderate scalability, as it relies on a single location of interconnected machines. 
    Adding nodes requires significant infrastructure investment. 
    Resource Sharing: Provides tightly coupled resource sharing within a single administrative domain, 
    making it ideal for high-performance computing. Computing Power: High computing power due to homogeneous hardware, 
    low-latency interconnections, and optimized resource utilization. Applications: Common in weather modeling, seismic analysis, 
    machine learning training, and big data batch processing tasks. Limitations: Limited geographical distribution. 
    Infrastructure and operational costs increase with size. Less flexible compared to other distributed paradigms'''

def main():
    def process_inputs():
        image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        reference_text = text_entry.get("1.0", tk.END).strip()
        if not image_path or not reference_text:
            result_label.config(text="Please provide both image and reference text.")
            return
        result_label.config(text="Processing... Please wait.")
        detected_text = recognize_text_azure(image_path, subscription_key, endpoint)
        results = evaluate_answer(detected_text, reference_text)
        result_display = (
            f"Detected Text (Paragraph):\n{detected_text}\n\n"
            f"Evaluation Results:\n"
            f"Keyword Score: {results['keyword_score']:.2f}/5\n"
            f"Semantic Similarity Score: {results['semantic_score']:.2f}/5\n"
            f"Total Score: {results['total_score']:.2f}/10"
        )
        result_label.config(text=result_display)

    subscription_key = "your-key"
    endpoint = "your-endpoint"

    app = tk.Tk()
    app.title("Answer Evaluation System")

    tk.Label(app, text="Enter Reference Answer:").pack()
    text_entry = Text(app, height=10, width=50)
    text_entry.pack()

    tk.Button(app, text="Select Image and Evaluate", command=process_inputs).pack()

    result_label = tk.Label(app, text="", wraplength=400, justify="left")
    result_label.pack()

    app.mainloop()

if __name__ == "__main__":
    main()
