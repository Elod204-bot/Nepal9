import os
import sys
import time
import io
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

# Your exact template prompt
PROMPT = (
    "A hyper-realistic, close-up photograph of an open Nepalese passport biographical data page and "
    "adjacent entry/exit page, resting on a rustic dark wood grain table. The bio page text is sharp, "
    "clear, and perfectly aligned in standard passport OCR-B fonts. At the top of the bio page, the text "
    "'NEPALESE PASSPORT' and 'PASSPORT' are printed in dark ink. The page includes a high-quality, "
    "professional headshot photo of a South Asian woman with long black hair, wearing clear-framed glasses "
    "and a dark blazer over a light shirt, with a natural, pleasant expression. Field labels are in both "
    "Nepali and English (Passport, Document Number, Name, Date of Birth (DOB), Gender, Place of Birth, "
    "Date of Issue, Date of Expiry, Issuing Authority). The data fields are populated with clear structural "
    "placeholders with brackets: Name: '[SURNAME] [GIVEN NAME]', Document Number: '[NPXXXXXXX]', "
    "DOB: '[DATE OF BIRTH]', Gender: '[GENDER]', Place of Birth: '[PLACE OF BIRTH]', Date of Issue: "
    "'[DATE OF ISSUE]', Date of Expiry: '[DATE OF EXPIRY]', Issuing Authority: '[ISSUING AUTHORITY]'. "
    "The background of the bio page features the intricate, light-colored Nepalese coat of arms and the "
    "distinct 'Compex Guilloché Line Pattern.' The Machine Readable Zone (MRZ) at the bottom is accurate, "
    "legible, and incorporates structural placeholders: 'P<NPL[SURNAME]<<[GIVEN \"ENTRY \"EXIT \"KATHMANDU, "
    "\"TRIBHUVAN '[YYYY.MM.DD]'. 3/4 AIRPORT\", INTERNATIONAL KATHMANDU\" NAME]<<<<<<<<<<<<<NPXXXXXXX<8NPL[YYMMDD][G][YYYYDD]G<<<<<<<<<<<<<<XX', "
    "NEPAL\", Nepalese TIA The There Various [ ] a above. across adjacent also and angled another are blue "
    "border composition control date denotes each entry exit fibers field. filled from generic genuine-looking "
    "ink is lighting. like natural, on overall overhead page page. paper passport placed placeholder placeholder. "
    "reading realistic realistically red right right-hand security shows slight soft stamp stamps stamps. "
    "texture, the under variable view visa wear, where with>"
)

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
        filename = os.path.join(OUTPUT_DIR, f"passport_template_{timestamp}_{iteration}.png")

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Requesting generation #{iteration} (Template Mode)...", flush=True)

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=PROMPT,
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
