from flask import Flask, render_template, request, jsonify
import sys
import os
import random
from sentence_transformers import SentenceTransformer, util

# Add scripts folder to path
sys.path.append(os.path.abspath("../scripts"))
from search_engine import search, load_knowledge_base

app = Flask(__name__)

# -----------------------------
# Load Knowledge Base
# -----------------------------
knowledge_text = load_knowledge_base()

# Load semantic model (for interview scoring)
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
# Format Answer
# -----------------------------
def format_answer(block):
    if not block:
        return ""

    lines = block.splitlines()
    clean_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith("Definition:"):
            clean_lines.append(line.replace("Definition:", "").strip())

        elif line.startswith("-"):
            clean_lines.append(line.strip("- ").strip())

    return " ".join(clean_lines)


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Ask Question (Main Assistant)
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"answer": "Invalid request"}), 400

    question = data.get("question", "").strip()

    if question == "":
        return jsonify({"answer": "Please ask a valid question."})

    result = search(question, knowledge_text)

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
# Evaluate Answer
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

    # Get model answer
    result = search(question, knowledge_text)

    if not result:
        return jsonify({
            "score": 0,
            "feedback": "Model answer not found in syllabus.",
            "model_answer": ""
        })

    model_answer = format_answer(result)

    # Semantic Similarity Scoring
    emb1 = model.encode(student_answer, convert_to_tensor=True)
    emb2 = model.encode(model_answer, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2).item()
    score = round(similarity * 100, 2)

    # Feedback
    if score > 75:
        feedback = "Excellent answer."
    elif score > 50:
        feedback = "Good attempt. Improve explanation."
    else:
        feedback = "Needs improvement."

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
