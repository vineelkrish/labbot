import os
import re
import random
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERVIEW_FILE = os.path.join(BASE_DIR, "../interview_data/os_interview.txt")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# SESSION STATE (REAL VIVA STATE MACHINE)
# ==========================================================
SESSION = {
    "active": False,
    "start_time": None,
    "duration": 300,  # 5 minutes
    "current_concept": None,
    "current_level": "easy",
    "current_question": None,
    "scores": defaultdict(list),
    "attempted": 0
}

QUESTION_BANK = {}


# ==========================================================
# PARSE FILE
# ==========================================================
def load_questions():
    global QUESTION_BANK
    QUESTION_BANK = {}

    with open(INTERVIEW_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    concepts = re.split(r"CONCEPT:\s*", text)[1:]

    for block in concepts:
        lines = block.strip().splitlines()
        concept = lines[0].strip()

        QUESTION_BANK[concept] = {"easy": [], "medium": [], "hard": []}

        entries = re.split(r"LEVEL:\s*", block)[1:]

        for entry in entries:
            level = entry.splitlines()[0].strip()
            q_match = re.search(r"QUESTION:\s*(.*)", entry)
            pts = re.findall(r"\*\s*(.*)", entry)

            if q_match:
                QUESTION_BANK[concept][level].append({
                    "question": q_match.group(1).strip(),
                    "points": pts
                })


# ==========================================================
# START INTERVIEW
# ==========================================================
def start_interview():
    load_questions()

    SESSION["active"] = True
    SESSION["start_time"] = time.time()
    SESSION["scores"].clear()
    SESSION["attempted"] = 0

    return pick_question()


# ==========================================================
# PICK NEXT QUESTION (ADAPTIVE)
# ==========================================================
def pick_question():

    # time over?
    if time.time() - SESSION["start_time"] > SESSION["duration"]:
        SESSION["active"] = False
        return None

    concept = random.choice(list(QUESTION_BANK.keys()))

    level = SESSION["current_level"]

    if not QUESTION_BANK[concept][level]:
        level = "easy"

    q = random.choice(QUESTION_BANK[concept][level])

    SESSION["current_concept"] = concept
    SESSION["current_question"] = q

    return f"[{concept} - {level.upper()}]\n{q['question']}"


# ==========================================================
# EVALUATE ANSWER (SEMANTIC SCORING)
# ==========================================================
def evaluate_answer(answer):

    if not SESSION["active"] or not SESSION["current_question"]:
        return 0, "Interview not active"

    answer_emb = model.encode(answer, convert_to_tensor=True)

    points = SESSION["current_question"]["points"]

    if not points:
        score = 50
    else:
        pts_emb = model.encode(points, convert_to_tensor=True)
        sims = util.cos_sim(answer_emb, pts_emb)[0]

        matched = sum(1 for s in sims if s > 0.45)
        score = int((matched / len(points)) * 100)

    SESSION["scores"][SESSION["current_concept"]].append(score)
    SESSION["attempted"] += 1

    # Adaptive difficulty
    if score > 75:
        SESSION["current_level"] = "hard"
        feedback = "Strong answer"
    elif score > 40:
        SESSION["current_level"] = "medium"
        feedback = "Okay answer"
    else:
        SESSION["current_level"] = "easy"
        feedback = "Weak answer"

    return score, feedback


# ==========================================================
# NEXT QUESTION
# ==========================================================
def next_question():
    if not SESSION["active"]:
        return None
    return pick_question()


# ==========================================================
# FINAL RESULT
# ==========================================================
def final_result():

    if SESSION["attempted"] == 0:
        return {"score": 0, "strong": [], "weak": []}

    topic_avg = {c: sum(v)/len(v) for c, v in SESSION["scores"].items()}

    strong = [c for c, s in topic_avg.items() if s >= 70]
    weak = [c for c, s in topic_avg.items() if s < 40]

    overall = int(sum(topic_avg.values()) / len(topic_avg))

    return {
        "score": overall,
        "strong": strong,
        "weak": weak
    }
