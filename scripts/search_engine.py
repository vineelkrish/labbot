import os
import re

# Path to the single knowledge base file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_FILE = os.path.join(BASE_DIR, "../syllabus_text/cleaned/os_knowledge_base.txt")

# Stopwords to ignore
STOPWORDS = {
    "what", "is", "an", "the", "a", "explain", "define",
    "how", "why", "does", "do", "of", "to", "in", "on",
    "and", "for"
}

# Minimum keyword hits required to accept an answer
MIN_KEYWORD_HITS = 1


def load_knowledge_base():
    """Load the full knowledge base file"""
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        raise FileNotFoundError("Knowledge base file not found")

    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        return f.read()


def extract_keywords(query):
    """Extract meaningful keywords from the question"""
    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    return [w for w in words if w not in STOPWORDS]


def split_into_concepts(text):
    """Split the knowledge base into individual concept blocks"""
    blocks = re.split(r"\n--- CONCEPT:", text)
    concepts = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Add back the marker for consistency
        concepts.append("--- CONCEPT:" + block)

    return concepts


def extract_concept_name(block):
    """Extract the concept name from a block"""
    match = re.search(r"--- CONCEPT:\s*(.+?)\s*---", block)
    return match.group(1).strip() if match else "Unknown Concept"


def search(query, knowledge_text):
    keywords = extract_keywords(query)

    if not keywords:
        return None

    concept_blocks = split_into_concepts(knowledge_text)

    best_block = None
    best_hits = 0

    for block in concept_blocks:
        block_lower = block.lower()
        hits = sum(1 for k in keywords if k in block_lower)

        if hits > best_hits:
            best_hits = hits
            best_block = block

    if best_hits < MIN_KEYWORD_HITS:
        return None

    return best_block.strip()


def format_answer(block):
    """Format the answer cleanly for display / voice output"""
    lines = block.splitlines()
    formatted = []

    for line in lines:
        if line.startswith("--- CONCEPT:"):
            formatted.append(line.replace("--- CONCEPT:", "Concept:").strip())
        elif line.startswith("Definition:"):
            formatted.append(line.strip())
        elif line.startswith("Key Points:"):
            formatted.append("Key Points:")
        elif line.startswith("-"):
            formatted.append(line.strip())

    return "\n".join(formatted)


if __name__ == "__main__":
    knowledge_text = load_knowledge_base()

    print("Syllabus-Aware OS Assistant")
    print("Ask a question (type 'exit' to quit)\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break

        result = search(question, knowledge_text)

        if result:
            print("\nAnswer:")
            print(format_answer(result))
            print()
        else:
            print("\nThis topic is outside the current syllabus.\n")
