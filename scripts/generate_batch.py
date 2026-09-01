import os
import sys
import time
import io
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "durable-student-507318-t0")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
DURATION_HOURS = float(os.environ.get("DURATION_HOURS", 5.8))
INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", 120))
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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to Vertex AI via google-genai...", flush=True)
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client connected successfully.", flush=True)
    except Exception as e:
        print(f"Initialization error: {e}", flush=True)
        sys.exit(1)

    # Preferred models: fast/flash tier first, then standard 002
    target_models = ["imagen-3.0-fast-generate-001", "imagen-3.0-generate-002"]

    start_time = time.time()
    max_duration_seconds = DURATION_HOURS * 3600
    iteration = 1

    print(f"Starting loop for ~{DURATION_HOURS} hours...", flush=True)

    while (time.time() - start_time) < max_duration_seconds:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/passport_{timestamp}_{iteration}.png"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating image #{iteration}...", flush=True)

        image_saved = False
        for model_name in target_models:
            try:
                print(f"Calling model '{model_name}'...", flush=True)
                result = client.models.generate_images(
                    model=model_name,
                    prompt=PROMPT,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="4:3",
                        output_mime_type="image/png",
                        person_generation="allow_adult"
                    )
                )

                if result.generated_images:
                    img_data = result.generated_images[0].image.image_bytes
                    image = Image.open(io.BytesIO(img_data))
                    image.save(filename)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Saved {filename}", flush=True)
                    image_saved = True
                    break
                else:
                    print(f"Model {model_name} returned no images (filtered).", flush=True)

            except Exception as err:
                print(f"Error with {model_name}: {err}", flush=True)

        if not image_saved:
            print("Iteration finished without saving an image.", flush=True)

        iteration += 1
        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining <= 0:
            print("Session time completed.", flush=True)
            break

        sleep_time = min(INTERVAL_SECONDS, remaining)
        print(f"Sleeping for {int(sleep_time)}s ({int(remaining / 60)} mins remaining)...", flush=True)
        time.sleep(sleep_time)

    print("\nSession complete.", flush=True)

if __name__ == "__main__":
    main()
