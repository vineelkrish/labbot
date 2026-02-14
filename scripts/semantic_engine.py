import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_FILE = os.path.join(BASE_DIR, "../syllabus_text/cleaned/os_knowledge_base.txt")

model = SentenceTransformer("all-MiniLM-L6-v2")

CONCEPTS = []
NAME_EMB = None
DESC_EMB = None


# ---------------- LOAD ----------------
def load_knowledge_base():
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        return f.read()


def split_into_concepts(text):
    blocks = re.split(r"\n--- CONCEPT:", text)
    concepts = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        concepts.append("--- CONCEPT:" + block)

    return concepts


def extract_name(block):
    match = re.search(r"--- CONCEPT:\s*(.*?)\s*---", block)
    return match.group(1).strip() if match else ""


def extract_description(block):
    clean = re.sub(r"--- CONCEPT:.*?---", "", block)
    clean = re.sub(r"Definition:|Key Points:", "", clean)
    clean = clean.replace("-", "")
    return clean.strip()


# ---------------- BUILD INDEX ----------------
def build_vector_index():
    global CONCEPTS, NAME_EMB, DESC_EMB

    text = load_knowledge_base()
    CONCEPTS = split_into_concepts(text)

    names = [extract_name(c) for c in CONCEPTS]
    descs = [extract_description(c) for c in CONCEPTS]

    NAME_EMB = model.encode(names, convert_to_tensor=True)
    DESC_EMB = model.encode(descs, convert_to_tensor=True)


# ---------------- SEARCH ----------------
def search(query):
    if NAME_EMB is None:
        build_vector_index()

    q_emb = model.encode(query, convert_to_tensor=True)

    name_scores = util.cos_sim(q_emb, NAME_EMB)[0]
    desc_scores = util.cos_sim(q_emb, DESC_EMB)[0]

    # weighted scoring (important!)
    final_scores = (0.65 * name_scores) + (0.35 * desc_scores)

    best_idx = int(np.argmax(final_scores))
    best_score = float(final_scores[best_idx])

    if best_score < 0.40:
        return None

    return CONCEPTS[best_idx]


# ---------------- FORMAT ----------------
def format_answer(block):
    lines = block.splitlines()
    formatted = []

    for line in lines:
        line = line.strip()

        if line.startswith("--- CONCEPT:"):
            formatted.append(line.replace("--- CONCEPT:", "").strip())

        elif line.startswith("Definition:"):
            formatted.append(line.replace("Definition:", "").strip())

        elif line.startswith("-"):
            formatted.append("• " + line.strip("- ").strip())

    return "\n".join(formatted)
