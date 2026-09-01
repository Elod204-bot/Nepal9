import os
import sys
import time
import io
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "durable-student-507318-t0")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
DURATION_HOURS = 5.8
INTERVAL_SECONDS = 120
OUTPUT_DIR = "output"

PROMPT = (
    "A hyper-realistic, close-up photograph of an open Nepalese passport biographical data "
    "page and adjacent entry/exit page, resting on a rustic dark wood grain table. "
    "The bio page features a professional headshot of a South Asian woman in her late 20s with "
    "glasses and a friendly expression. Document text is sharp, clear, and aligned in OCR-B font, "
    "including text 'NEPALESE PASSPORT', document number NP9876543, name AARTI GURUNG, DOB 12 Nov 1995, "
    "place of birth POKHARA, issue date 01 Feb 2024, expiry date 31 Jan 2034. "
    "Background features a light-colored Nepalese coat of arms and intricate guilloché line patterns. "
    "The Machine Readable Zone (MRZ) is visible at the bottom. The adjacent page has realistic red and "
    "blue entry/exit ink stamps from Tribhuvan International Airport, Kathmandu. "
    "Natural soft overhead lighting, 3/4 angled view from above."
)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting via modern google-genai SDK...", flush=True)

    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    except Exception as e:
        print(f"Connection failed: {e}", flush=True)
        sys.exit(1)

    start_time = time.time()
    max_duration = DURATION_HOURS * 3600
    iteration = 1

    while (time.time() - start_time) < max_duration:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/passport_{timestamp}_{iteration}.png"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Requesting image #{iteration}...", flush=True)

        try:
            result = client.models.generate_images(
                model="imagen-3.0-generate-001",
                prompt=PROMPT,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="4:3",
                    output_mime_type="image/png"
                )
            )

            if result.generated_images:
                img_data = result.generated_images[0].image.image_bytes
                image = Image.open(io.BytesIO(img_data))
                image.save(filename)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS -> {filename}", flush=True)
            else:
                print("Blocked by safety filters.", flush=True)

        except Exception as err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API ERROR: {err}", flush=True)

        iteration += 1
        elapsed = time.time() - start_time
        remaining = max_duration - elapsed

        if remaining <= 0:
            break

        time.sleep(min(INTERVAL_SECONDS, remaining))

if __name__ == "__main__":
    main()
