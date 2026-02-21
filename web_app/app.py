from flask import Flask, render_template, request, jsonify
import sys
import os

# Add scripts folder to path
sys.path.append(os.path.abspath("../scripts"))

# Import engines
from semantic_engine import search, format_answer, build_vector_index
from interview_engine import start_interview, evaluate_answer, next_question, final_result

app = Flask(__name__)

# Build semantic vector DB once at startup
print("Loading knowledge base...")
build_vector_index()
print("Knowledge base loaded")


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= ASK (LEARNING MODE) =================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True)

        if not data or "question" not in data:
            return jsonify({"answer": "Invalid request."})

        question = data.get("question", "").strip()

        if question == "":
            return jsonify({"answer": "Please ask a valid question."})

        result, subject = search(question)

        if result:
            answer_text = format_answer(result)
            answer_text = f"[From {subject.upper()}]\n" + answer_text
        else:
            answer_text = "This topic is outside the current syllabus."

        return jsonify({"answer": answer_text})

    except Exception as e:
        print("ASK ERROR:", e)
        return jsonify({"answer": "Internal error occurred while answering."})


# ================= START INTERVIEW =================
@app.route("/start_interview", methods=["GET"])
def start_interview_route():
    try:
        question = start_interview()

        if question is None:
            return jsonify({"question": None})

        return jsonify({"question": question})

    except Exception as e:
        print("INTERVIEW START ERROR:", e)
        return jsonify({"question": None})


# ================= EVALUATE ANSWER =================
@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        data = request.get_json(silent=True)

        if not data or "answer" not in data:
            return jsonify({
                "score": 0,
                "feedback": "Invalid answer.",
                "next": None,
                "final_score": 0,
                "strong": [],
                "weak": []
            })

        student_answer = data.get("answer", "").strip()

        # Evaluate current answer
        score, feedback = evaluate_answer(student_answer)

        # Get next question
        next_q = next_question()

        # ================= INTERVIEW FINISHED =================
        if next_q is None:
            final = final_result()

            return jsonify({
                "score": score,
                "feedback": feedback,
                "next": None,
                "final_score": final["score"],
                "strong": final["strong"],
                "weak": final["weak"]
            })

        # ================= CONTINUE INTERVIEW =================
        return jsonify({
            "score": score,
            "feedback": feedback,
            "next": next_q
        })

    except Exception as e:
        print("EVALUATE ERROR:", e)
        return jsonify({
            "score": 0,
            "feedback": "Internal evaluation error.",
            "next": None,
            "final_score": 0,
            "strong": [],
            "weak": []
        })


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
