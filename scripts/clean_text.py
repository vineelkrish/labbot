import os
import re

TEXT_DIR = "../syllabus_text"
CLEAN_DIR = "../syllabus_text/cleaned"

os.makedirs(CLEAN_DIR, exist_ok=True)

def is_garbage_line(line):
    # very short lines with mostly symbols/numbers
    if len(line) < 4 and not line.isalpha():
        return True
    if re.search(r"[|•]", line):
        return True
    if re.fullmatch(r"[0-9\s]+", line):
        return True
    if re.fullmatch(r"(user\s*){2,}", line.lower()):
        return True
    return False

def clean_text(text):
    lines = text.splitlines()
    cleaned_paragraphs = []
    buffer = ""

    for line in lines:
        line = line.strip()

        if not line:
            if buffer:
                cleaned_paragraphs.append(buffer)
                buffer = ""
            continue

        if is_garbage_line(line):
            continue

        # Join broken lines
        if buffer:
            buffer += " " + line
        else:
            buffer = line

        if line.endswith((".", "?", "!")):
            cleaned_paragraphs.append(buffer)
            buffer = ""

    if buffer:
        cleaned_paragraphs.append(buffer)

    return "\n\n".join(cleaned_paragraphs)

for file in os.listdir(TEXT_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(TEXT_DIR, file), "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = clean_text(raw_text)

        with open(os.path.join(CLEAN_DIR, file), "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"[OK] Cleaned {file}")
