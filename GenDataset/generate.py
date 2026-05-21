import requests
import json
import time
import os
import random

# Config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"  # or gemma3:4b
OUTPUT_FILE = "gym_personas.json"
TOTAL_PERSONAS = 50
MAX_RETRIES = 3

FITNESS_GOALS = ["Weight Loss", "Muscle Gain", "Stress Relief", "Social/Community", "Rehabilitation", "General Health"]
LIFE_CONTEXTS = ["Busy Professional", "College Student", "Stay-at-home Parent", "Retiree", "Fitness Enthusiast", "Shift Worker", "Corporate Desk Worker"]

# NEW: Control the distribution to reflect reality (most people struggle with consistency)
MOTIVATION_LEVELS = [
    "Very Low (Easily intimidated, procrastinates, hates sweating)",
    "Low (Struggles with consistency, easily makes excuses)",
    "Medium (Tries their best, goes 1-2 times a week)",
    "High (Dedicated, enjoys working out)",
    "Very High (Fitness is a core part of their identity)"
]
# 65% of people will have low/very low motivation (realistic gym stats)
MOTIVATION_WEIGHTS = [0.25, 0.40, 0.20, 0.10, 0.05] 

def stream_generate(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.9,
            "num_predict": 1000
        }
    }

    print("[Streaming output]\n", end="", flush=True)
    full_text = ""
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        full_text += token
                        print(token, end="", flush=True)
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        pass
        print("\n")
    except requests.exceptions.RequestException as e:
        print(f"\nStreaming error: {e}")
        return ""
    return full_text

def parse_json_object(text):
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return None

def generate_single_persona(index):
    # Force Python to decide the baseline, not the LLM (which is biased towards positivity)
    chosen_motivation = random.choices(MOTIVATION_LEVELS, weights=MOTIVATION_WEIGHTS, k=1)[0]
    chosen_context = random.choice(LIFE_CONTEXTS)
    chosen_goal = random.choice(FITNESS_GOALS)

    prompt = f"""You are a synthetic data generator creating profiles for gym members. 
Generate exactly 1 diverse gym member persona based on these assigned traits:
- Life Context: {chosen_context}
- Fitness Goal: {chosen_goal}
- Motivation Level: {chosen_motivation}

Crucial: LLMs tend to make everyone highly motivated. YOU MUST respect the assigned Motivation Level. If they have Low or Very Low motivation, their story must reflect laziness, intimidation, or excuses (e.g., paying for a membership but only going to sit in the sauna, or quitting after 2 weeks).

The persona must be a JSON object (not a list) with the following keys:
- "name": a fictional full name
- "age": a realistic integer between 18 and 75
- "fitness_goal": "{chosen_goal}"
- "life_context": "{chosen_context}"
- "motivation": "{chosen_motivation}"
- "personality": a short descriptor (e.g., "Anxious", "Lazy but hopeful", "Extroverted")
- "story": a short, realistic 2-3 sentence backstory explaining why they signed up for the gym and what their daily life is like. Ensure their behavior matches their assigned motivation level.

Return ONLY a valid JSON object, no other text, no markdown. Persona id: {index}."""

    for attempt in range(MAX_RETRIES):
        text = stream_generate(prompt)
        if text:
            persona = parse_json_object(text)
            if persona and "age" in persona:
                return persona
            print(f"Attempt {attempt+1}: could not parse JSON. Retrying...")
        time.sleep(1)
    return None

def main():
    all_personas = []

    if os.path.exists(OUTPUT_FILE):
        ans = input(f"'{OUTPUT_FILE}' already exists. Overwrite? (y/n) ").strip().lower()
        if ans != 'y': return

    print("\nGenerating gym persona #1 with real-time streaming...\n")
    first = generate_single_persona(1)
    if first is None:
        print("Failed to generate. Exiting.")
        return

    all_personas.append(first)
    print("\n--- Persona #1 ---")
    print(json.dumps(first, indent=2))

    try:
        answer = input(f"Generate the remaining {TOTAL_PERSONAS - 1} personas? (y/n) ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        answer = 'n'

    if answer == 'y':
        for i in range(2, TOTAL_PERSONAS + 1):
            print(f"\n--- Generating persona {i}/{TOTAL_PERSONAS} ---")
            p = generate_single_persona(i)
            if p: all_personas.append(p)
            
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_personas, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_personas)} personas to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()