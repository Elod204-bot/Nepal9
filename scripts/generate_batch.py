import os
import sys
import time
from datetime import datetime
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "durable-student-507318-t0")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Vertex AI...", flush=True)
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        # Using the standard Imagen-3 production endpoint
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded successfully.", flush=True)
    except Exception as e:
        print(f"Initialization error: {e}", flush=True)
        sys.exit(1)

    start_time = time.time()
    max_duration_seconds = DURATION_HOURS * 3600
    iteration = 1

    print(f"Starting loop for ~{DURATION_HOURS} hours...", flush=True)

    while (time.time() - start_time) < max_duration_seconds:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/passport_{timestamp}_{iteration}.png"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Requesting image #{iteration} from Imagen 3...", flush=True)

        try:
            response = model.generate_images(
                prompt=PROMPT,
                number_of_images=1,
                aspect_ratio="4:3"
            )

            if response and response.images:
                response.images[0].save(location=filename, include_generation_parameters=False)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Image saved to {filename}", flush=True)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No image returned (safety filter).", flush=True)

        except Exception as err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API Error: {err}", flush=True)

        iteration += 1
        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining <= 0:
            print("Session time completed.", flush=True)
            break

        sleep_time = min(INTERVAL_SECONDS, remaining)
        print(f"Sleeping for {int(sleep_time)} seconds ({int(remaining / 60)} mins remaining)...", flush=True)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
