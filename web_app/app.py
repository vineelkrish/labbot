from flask import Flask, render_template, request, jsonify
import sys
import os
import random
from sentence_transformers import SentenceTransformer, util

# Add scripts folder to path
sys.path.append(os.path.abspath("../scripts"))

# IMPORT NEW SEMANTIC ENGINE
from semantic_engine import search, format_answer, build_vector_index

app = Flask(__name__)

# Build vector database once at startup
build_vector_index()

# Model for interview scoring
model = SentenceTransformer('all-MiniLM-L6-v2')


# -----------------------------
# Interview Question Bank
# -----------------------------
INTERVIEW_QUESTIONS = [
    "What is an operating system?",
    "Explain process and program difference.",
    "What is CPU scheduling?",
    "What is deadlock?",
    "Explain paging in memory management."
]


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Ask Question (Semantic AI)
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"answer": "Invalid request"}), 400

    question = data.get("question", "").strip()

    if question == "":
        return jsonify({"answer": "Please ask a valid question."})

    # SEMANTIC SEARCH
    result = search(question)

    if result:
        answer_text = format_answer(result)
    else:
        answer_text = "This topic is outside the current syllabus."

    return jsonify({"answer": answer_text})


# -----------------------------
# Interview Mode
# -----------------------------
@app.route("/start_interview", methods=["GET"])
def start_interview():
    question = random.choice(INTERVIEW_QUESTIONS)
    return jsonify({"question": question})


# -----------------------------
# Evaluate Answer (AI Grading)
# -----------------------------
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "score": 0,
            "feedback": "Invalid answer submission.",
            "model_answer": ""
        })

    question = data.get("question", "").strip()
    student_answer = data.get("answer", "").strip()

    if student_answer == "":
        return jsonify({
            "score": 0,
            "feedback": "You did not provide an answer.",
            "model_answer": ""
        })

    # Get semantic model answer
    result = search(question)

    if not result:
        return jsonify({
            "score": 0,
            "feedback": "Model answer not found in syllabus.",
            "model_answer": ""
        })

    model_answer = format_answer(result)

    # Semantic similarity scoring
    emb1 = model.encode(student_answer, convert_to_tensor=True)
    emb2 = model.encode(model_answer, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2).item()
    score = round(similarity * 100, 2)

    # Feedback logic
    if score > 80:
        feedback = "Excellent answer."
    elif score > 60:
        feedback = "Good understanding, add more details."
    elif score > 40:
        feedback = "Partial understanding."
    else:
        feedback = "Needs improvement. Revise the concept."

    return jsonify({
        "score": score,
        "feedback": feedback,
        "model_answer": model_answer
    })


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
