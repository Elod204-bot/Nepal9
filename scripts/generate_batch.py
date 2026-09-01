import os
import sys
import time
import io
import random
from datetime import datetime

from PIL import Image
from google import genai
from google.genai import types

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "durable-student-507318-t0")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
DURATION_HOURS = float(os.environ.get("DURATION_HOURS", 5.8))
INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", 5))
OUTPUT_DIR = "output"
MODEL_NAME = "gemini-3.1-flash-image"

def generate_dynamic_prompt():
    first_names_f = ["Aarti", "Priya", "Sunita", "Maya", "Sita", "Nisha", "Bina", "Gita"]
    first_names_m = ["Arjun", "Bikash", "Ramesh", "Sanjay", "Kamal", "Raj", "Sunil", "Nabin"]
    last_names = ["Gurung", "Thapa", "Shrestha", "Tamang", "Lama", "Magar", "Rai", "Karki"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    places = ["POKHARA", "KATHMANDU", "LALITPUR", "BHAKTAPUR", "DHARAN", "BUTWAL"]

    is_female = random.choice([True, False])
    
    if is_female:
        name = f"{random.choice(first_names_f)} {random.choice(last_names)}"
        desc = random.choice([
            "South Asian woman in her late 20s with glasses and a friendly expression",
            "South Asian woman in her early 30s with long dark hair and a neutral expression",
            "South Asian woman in her 40s wearing traditional small earrings and a warm smile",
            "South Asian woman in her 20s with short hair and a serious expression"
        ])
    else:
        name = f"{random.choice(first_names_m)} {random.choice(last_names)}"
        desc = random.choice([
            "South Asian man in his early 30s with a neat mustache and a serious expression",
            "South Asian man in his 20s with short hair and a slight smile",
            "South Asian man in his 40s with a short beard and glasses",
            "South Asian man in his late 20s with a clean-shaven face and professional look"
        ])

    dob_year = random.randint(1975, 2003)
    dob_day = random.randint(1, 28)
    dob = f"{dob_day:02d} {random.choice(months)} {dob_year}"
    
    passport_num = f"NP{random.randint(1000000, 9999999)}"
    birth_place = random.choice(places)

    prompt = (
        f"A hyper-realistic, close-up photograph of an open Nepalese passport biographical data "
        f"page and adjacent entry/exit page, resting on a rustic dark wood grain table. "
        f"The bio page features a professional headshot of a {desc}. Document text is sharp, clear, and aligned in OCR-B font, "
        f"including text 'NEPALESE PASSPORT', document number {passport_num}, name {name.upper()}, DOB {dob}, "
        f"place of birth {birth_place}, issue date 01 Feb 2024, expiry date 31 Jan 2034. "
        f"Background features a light-colored Nepalese coat of arms and intricate guilloché line patterns. "
        f"The Machine Readable Zone (MRZ) is visible at the bottom. The adjacent page has realistic red and "
        f"blue entry/exit ink stamps from Tribhuvan International Airport, Kathmandu. "
        f"Natural soft overhead lighting, 3/4 angled view from above."
    )
    
    return prompt, name.upper(), passport_num

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to {PROJECT_ID} in {LOCATION}...", flush=True)

    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client initialized successfully.", flush=True)
    except Exception as e:
        print(f"Initialization error: {e}", flush=True)
        sys.exit(1)

    start_time = time.time()
    max_duration_seconds = DURATION_HOURS * 3600
    iteration = 1

    print(f"Starting loop using {MODEL_NAME}...", flush=True)

    while (time.time() - start_time) < max_duration_seconds:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"passport_{timestamp}_{iteration}.png")
        
        # Generate a new personality for this specific iteration
        current_prompt, current_name, current_passport = generate_dynamic_prompt()

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Requesting generation #{iteration} ({current_name} - {current_passport})...", flush=True)

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )

            image_saved = False

            if response.candidates:
                for candidate in response.candidates:
                    if not candidate.content:
                        continue

                    for part in candidate.content.parts:
                        if getattr(part, "inline_data", None):
                            image_bytes = part.inline_data.data
                            image = Image.open(io.BytesIO(image_bytes))
                            image.save(filename)

                            print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Saved {filename}", flush=True)
                            image_saved = True
                            break

                    if image_saved:
                        break

            if not image_saved:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No image returned.", flush=True)
                if response.text:
                    print("Model response:", response.text, flush=True)

        except Exception as err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API ERROR: {err}", flush=True)

        iteration += 1
        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining <= 0:
            print("Session completed.", flush=True)
            break

        sleep_time = min(INTERVAL_SECONDS, remaining)

        if sleep_time > 0:
            print(f"Sleeping {int(sleep_time)}s...", flush=True)
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
