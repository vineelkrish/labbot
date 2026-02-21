import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- MULTI SUBJECT KB ----------
KB_FILES = {
    "os": os.path.join(BASE_DIR, "../syllabus_text/cleaned/os_knowledge_base.txt"),
    "dbms": os.path.join(BASE_DIR, "../syllabus_text/cleaned/dbms_knowledge_base.txt"),
    "cn": os.path.join(BASE_DIR, "../syllabus_text/cleaned/cn_knowledge_base.txt")
}

model = SentenceTransformer("all-MiniLM-L6-v2")

# Each subject stores its own vectors  
SUBJECT_DATA = {}

# ---------- SUBJECT DETECTION ----------
subject_keywords = {
    "os": "process scheduling deadlock paging semaphore cpu thread synchronization memory",
    "dbms": "sql database normalization transaction schema er diagram table key relation",
    "cn": "network osi tcp ip routing packet dns protocol topology"
}

subject_vectors = {s: model.encode(text, convert_to_tensor=True) for s, text in subject_keywords.items()}


# ---------------- LOAD & PARSE ----------------
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
    """
    Build a rich semantic description instead of deleting sections.
    This keeps definition + explanation + key points together.
    """

    text = block

    # remove concept header
    text = re.sub(r"--- CONCEPT:.*?---", "", text, flags=re.DOTALL)

    lines = text.splitlines()
    collected = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # remove section labels but KEEP content
        line = re.sub(r'(?i)^(definition|explanation|key points|examples?)\s*:\s*', '', line)

        # convert bullets to sentence
        if line.startswith("-"):
            line = line.strip("- ").strip()

        collected.append(line)

    # join into semantic paragraph
    return " ".join(collected)


# ---------------- BUILD INDEX ----------------
def build_vector_index():
    global SUBJECT_DATA

    for subject, path in KB_FILES.items():

        if not os.path.exists(path):
            print(f"Missing KB: {subject}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        concepts = split_into_concepts(text)

        semantic_texts = []
        for c in concepts:
            title = extract_name(c)
            desc = extract_description(c)

            # combine meaning (VERY IMPORTANT)
            semantic_texts.append(f"{title}. {desc}")

        embeddings = model.encode(semantic_texts, convert_to_tensor=True)

        SUBJECT_DATA[subject] = {
            "concepts": concepts,
            "embeddings": embeddings
        }

        print(f"{subject.upper()} loaded → {len(concepts)} concepts")


# ---------------- SUBJECT DETECTION ----------------
def detect_subject(query):
    q_emb = model.encode(query, convert_to_tensor=True)

    best_subject = None
    best_score = -1

    for subject, vec in subject_vectors.items():
        score = util.cos_sim(q_emb, vec).item()

        if score > best_score:
            best_score = score
            best_subject = subject

    return best_subject, best_score


# ---------------- SEARCH ----------------
def search(query):

    if not SUBJECT_DATA:
        build_vector_index()

    subject, confidence = detect_subject(query)
    data = SUBJECT_DATA.get(subject)

    if not data:
        return None, None

    q_emb = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, data["embeddings"])[0]

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < 0.35:
        return None, subject

    return data["concepts"][best_idx], subject

# ---------------- FORMAT ----------------
# ---------------- FORMAT ----------------
def format_answer(block):
    lines = block.splitlines()
    formatted = []
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # ---------- TITLE ----------
        if line.lower().startswith("--- concept:"):
            title = line.split(":", 1)[1]
            title = re.sub(r'-{2,}', '', title).strip()   # remove trailing ---
            formatted.append(title)
            current_section = None
            continue

        # ---------- SECTION HEADERS ----------
        header = re.match(r'(?i)^(definition|explanation|example|examples|key points)\s*:\s*(.*)', line)
        if header:
            current_section = header.group(1).lower()

            # Do NOT print "Key Points:" label
            if current_section != "key points":
                content = header.group(2).strip()
                if content:
                    formatted.append(content)
            continue

        # ---------- BULLETS ----------
        if re.match(r'^[-*•]\s+', line):
            formatted.append(line[1:].strip())   # clean text only (UI adds bullet)
            continue

        # ---------- NUMBERED LIST ----------
        if re.match(r'^\d+\.\s+', line):
            formatted.append(re.sub(r'^\d+\.\s+', '', line))
            continue

        # ---------- MULTI-LINE TEXT ----------
        if current_section in ("definition", "explanation", "example", "examples"):
            formatted.append(line)

    return "\n".join(formatted)
