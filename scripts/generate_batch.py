import io
import os
import random
import sys
import time
from datetime import datetime

from google import genai
from google.genai import types
from PIL import Image

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "durable-student-507318-t0")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
DURATION_HOURS = float(os.environ.get("DURATION_HOURS", 5.8))
INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", 61))
OUTPUT_DIR = "output"
MODEL_NAME = "gemini-3.1-flash-image"


def generate_dynamic_prompt():
    first_names = [
        "Aarav", "Bikash", "Dipendra", "Roshan", "Sandeep",
        "Suman", "Pradeep", "Manish", "Bibek", "Rajesh"
    ]
    last_names = [
        "Gurung", "Thapa", "Shrestha", "Tamang", "Lama",
        "Magar", "Rai", "Karki", "Adhikari", "Joshi"
    ]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    places = ["POKHARA", "KATHMANDU", "LALITPUR", "BHAKTAPUR", "DHARAN", "BUTWAL", "BIRATNAGAR"]

    # Variable 1: Name
    first = random.choice(first_names)
    last = random.choice(last_names)
    name = f"{first} {last}".upper()

    # Variable 2: Document Number
    passport_num = f"NP{random.randint(1000000, 9999999)}"

    # Variable 3: DOB
    dob_year = random.randint(1990, 2004)
    dob_day = random.randint(1, 28)
    dob_month = random.choice(months)
    dob = f"{dob_day:02d} {dob_month} {dob_year}"

    # Variables 4 & 5: Issue & Expiry Dates
    issue_year = 2024
    issue_day = random.randint(1, 28)
    issue_month = random.choice(months)
    issue_date = f"{issue_day:02d} {issue_month} {issue_year}"
    expiry_date = f"{issue_day:02d} {issue_month} {issue_year + 10}"

    # Variable 6: Place of Birth
    birth_place = random.choice(places)

    # Machine Readable Zone (MRZ) formatted with male marker 'M'
    mrz_name = f"{last}<<{first}<<<<<<<<<<<<<".upper()
    mrz_line = (
        f"P<NPL{mrz_name}{passport_num}<8NPL"
        f"{dob_year%100:02d}{months.index(dob_month)+1:02d}{dob_day:02d}"
        f"M34{issue_year%100:02d}{months.index(issue_month)+1:02d}{issue_day:02d}"
        f"<<<<<<<<<<<<<<04"
    )

    prompt = (
        "A realistic, high-angle close-up photograph of an open Nepalese passport resting on a rustic "
        "dark wooden table. The bio-data identity page features an official government-style layout with "
        "sharp OCR-B typography, official security ink stamps, and the national emblem of Nepal visible as a subtle "
        "watermark pattern in the center. "
        "Passport Photo: A clear, professional passport portrait of a young South Asian man wearing a dark blue jacket "
        "over a neutral shirt, facing forward with a neutral, formal expression against a plain off-white studio background. "
        "Document Fields: "
        f"Name: {name}, Document No: {passport_num}, Date of Birth: {dob}, Sex: M, "
        f"Place of Birth: {birth_place}, Date of Issue: {issue_date}, Date of Expiry: {expiry_date}, "
        "Authority: Department of Passports Kathmandu. "
        f"Bottom Line MRZ: Fully legible standard machine-readable zone: {mrz_line}. "
        "Lighting & Optics: Natural side lighting accentuating fine paper fiber texture, crisp focus across the text fields, "
        "and a gentle shallow depth of field softening the table edges."
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

        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Requesting generation #{iteration} "
            f"({current_name} - {current_passport})...",
            flush=True,
        )

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
