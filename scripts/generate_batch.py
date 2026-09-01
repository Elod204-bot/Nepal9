import os
import sys
import time
import io
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "durable-student-507318-t0")
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

def find_working_endpoint():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning Google Cloud regions for unlocked Imagen access...", flush=True)
    regions = ["us-central1", "us-east1", "europe-west1", "europe-west4", "asia-southeast1"]
    models = ["imagen-3.0-generate-001", "imagen-3.0-generate-002", "imagen-3.0-fast-generate-001"]

    for region in regions:
        try:
            client = genai.Client(vertexai=True, project=PROJECT_ID, location=region)
        except Exception:
            continue

        for model_name in models:
            print(f"Testing {model_name} in {region}...", flush=True)
            try:
                # Test the connection with a harmless prompt to bypass safety filters during the check
                client.models.generate_images(
                    model=model_name,
                    prompt="A simple blue square.",
                    config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1")
                )
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Locked onto {model_name} in {region}", flush=True)
                return client, model_name
            except Exception as e:
                error_msg = str(e).lower()
                # If it throws a safety error on a blue square, the model exists and is working.
                if "safety" in error_msg or "blocked" in error_msg:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Locked onto {model_name} in {region}", flush=True)
                    return client, model_name
                # 404 means not in this region, move to the next combination
                continue
                
    return None, None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    client, active_model = find_working_endpoint()
    
    if not client:
        print("CRITICAL FAILURE: Google Cloud is returning 404 for Imagen across all major regions in this project.", flush=True)
        sys.exit(1)

    start_time = time.time()
    max_duration = DURATION_HOURS * 3600
    iteration = 1

    print(f"\nStarting ~{DURATION_HOURS} hour generation loop...", flush=True)

    while (time.time() - start_time) < max_duration:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/passport_{timestamp}_{iteration}.png"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Requesting image #{iteration}...", flush=True)

        try:
            result = client.models.generate_images(
                model=active_model,
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
                print(f"[{datetime.now().strftime('%H:%M:%S')}] BLOCKED: Vertex AI safety filters flagged the passport prompt.", flush=True)

        except Exception as err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API ERROR: {err}", flush=True)

        iteration += 1
        elapsed = time.time() - start_time
        remaining = max_duration - elapsed

        if remaining <= 0:
            print("Session time completed.", flush=True)
            break

        time.sleep(min(INTERVAL_SECONDS, remaining))

if __name__ == "__main__":
    main()
