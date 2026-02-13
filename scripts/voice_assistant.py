import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
from search_engine import search, load_knowledge_base


# -------------------------------
# TEXT TO SPEECH (gTTS VERSION)
# -------------------------------

def speak(text):
    if not text or not text.strip():
        return

    print("\nAssistant:", text)

    tts = gTTS(text=text, lang='en')
    filename = "response.mp3"
    tts.save(filename)

    playsound(filename)

    os.remove(filename)


# -------------------------------
# SPEECH RECOGNITION
# -------------------------------

recognizer = sr.Recognizer()


def listen():
    with sr.Microphone() as source:
        print("\n🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You:", text)
        return text

    except sr.UnknownValueError:
        return None

    except sr.RequestError:
        speak("Speech service is not available.")
        return None


# -------------------------------
# FORMAT ANSWER
# -------------------------------

def format_answer(block):
    lines = block.splitlines()
    clean_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith("Definition:"):
            clean_lines.append(line.replace("Definition:", "").strip())

        elif line.startswith("-"):
            clean_lines.append(line.strip("- ").strip())

    return ". ".join(clean_lines)


# -------------------------------
# MAIN
# -------------------------------

def main():
    knowledge_text = load_knowledge_base()

    speak("Operating Systems voice assistant started. Ask your question.")

    while True:
        query = listen()

        if not query:
            speak("Sorry, I did not understand. Please repeat.")
            continue

        if query.lower() == "exit":
            speak("Goodbye.")
            break

        result = search(query, knowledge_text)

        if result:
            answer_text = format_answer(result)
            speak(answer_text)
        else:
            speak("This topic is outside the current syllabus.")


if __name__ == "__main__":
    main()
