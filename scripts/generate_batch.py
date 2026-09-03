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
INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", 61))
OUTPUT_DIR = "output"
MODEL_NAME = "gemini-3.1-flash-image"

def generate_dynamic_prompt():
    first_names = ["Aarti", "Priya", "Sunita", "Maya", "Sita", "Nisha", "Bina", "Gita", "Anjali", "Puja"]
    last_names = ["Gurung", "Thapa", "Shrestha", "Tamang", "Lama", "Magar", "Rai", "Karki", "Adhikari", "Joshi"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    places = ["POKHARA", "KATHMANDU", "LALITPUR", "BHAKTAPUR", "DHARAN", "BUTWAL", "BIRATNAGAR"]

    # Variable 1: Name
    first = random.choice(first_names)
    last = random.choice(last_names)
    name = f"{first} {last}".upper()

    # Variable 2: Document Number
    passport_num = f"NP{random.randint(1000000, 9999999)}"

    # Variable 3: DOB
    dob_year = random.randint(1985, 2003)
    dob_day = random.randint(1, 28)
    dob_month = random.choice(months)
    dob = f"{dob_day:02d} {dob_month} {dob_year}"

    # Variable 4 & 5: Issue & Expiry Dates
    issue_year = 2024
    issue_day = random.randint(1, 28)
    issue_month = random.choice(months)
    issue_date = f"{issue_day:02d} {issue_month} {issue_year}"
    expiry_date = f"{issue_day:02d} {issue_month} {issue_year + 10}"

    # Variable 6: Place of Birth
    birth_place = random.choice(places)

    # Simulated MRZ generation based on variables
    mrz_name = f"{last}<<{first}<<<<<<<<<<<<<".upper()
    mrz_line = f"P<NPL{mrz_name}{passport_num}<8NPL{dob_year%100:02d}{months.index(dob_month)+1:02d}{dob_day:02d}F34{issue_year%100:02d}{months.index(issue_month)+1:02d}{issue_day:02d}<<<<<<<<<<<<<<04"

    prompt = (
        "A hyper-realistic, close-up photograph of an open Nepalese passport biographical data "
        "page and adjacent entry/exit page, resting on a rustic dark wood grain table. "
        "Document Details: The bio page text must be sharp, clear, and perfectly aligned, using standard passport OCR-B fonts. "
        "Document Type: 'NEPALESE PASSPORT' and 'PASSPORT' printed at the top in dark ink. "
        "Passport photo: A high-quality, professional headshot of a South Asian woman with long black hair, "
        "wearing clear-framed glasses and a black blazer over a light shirt, with a natural, pleasant expression. "
        "Field Labels (in Nepali and English): 'Passport', 'Document Number', 'Name', 'Date of Birth (DOB)', "
        "'Gender', 'Place of Birth', 'Date of Issue', 'Date of Expiry', 'Issuing Authority'. "
        f"Specific Data Points: [VARIABLE 1 - NAME]: {name}, [VARIABLE 2 - DOCUMENT NUMBER]: {passport_num}, "
        f"[VARIABLE 3 - DOB]: {dob}, [VARIABLE 4 - ISSUE DATE]: {issue_date}, [VARIABLE 5 - EXPIRY DATE]: {expiry_date}, "
        f"[VARIABLE 6 - PLACE OF BIRTH]: {birth_place}. "
        "Aesthetics and Security Features: The background of the bio page must feature the intricate, light-colored "
        "Nepalese coat of arms and the distinct 'Complex Guilloché Line Pattern'. "
        f"The Machine Readable Zone (MRZ) at the bottom must be accurate and legible using the structure: {mrz_line}. "
        "The adjacent page has realistic red and blue entry/exit ink stamps from TIA Tribhuvan International Airport, "
        "Kathmandu, Nepal, with clear dates, fiber-filled paper texture, and natural soft overhead lighting viewed from a 3/4 right angle."
    )

    return prompt, name, passport_num

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
