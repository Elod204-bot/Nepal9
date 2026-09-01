import os
import sys
import time
from datetime import datetime
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- Configuration ---
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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Booting up Vertex AI in {LOCATION}...", flush=True)

    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        # Forcing the most stable, guaranteed-to-exist endpoint for Vertex Vision
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Locked onto verified model: imagegeneration@006", flush=True)
    except Exception as e:
        print(f"Fatal SDK Error: {e}", flush=True)
        sys.exit(1)

    start_time = time.time()
    max_duration = DURATION_HOURS * 3600
    iteration = 1

    print(f"Starting ~{DURATION_HOURS} hour loop. Stand by for generations...", flush=True)

    while (time.time() - start_time) < max_duration:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/passport_{timestamp}_{iteration}.png"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Firing API Request #{iteration}...", flush=True)

        try:
            response = model.generate_images(
                prompt=PROMPT,
                number_of_images=1,
                aspect_ratio="4:3"
            )

            if response and response.images:
                response.images[0].save(location=filename, include_generation_parameters=False)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS -> {filename}", flush=True)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] BLOCKED: Google Safety Filter caught the passport prompt.", flush=True)

        except Exception as err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API ERROR: {err}", flush=True)

        iteration += 1
        elapsed = time.time() - start_time
        remaining = max_duration - elapsed

        if remaining <= 0:
            print("Time limit reached. Shutting down.", flush=True)
            break

        print(f"Sleeping 120s to prevent rate limits ({int(remaining/60)} mins left)...", flush=True)
        time.sleep(min(INTERVAL_SECONDS, remaining))

if __name__ == "__main__":
    main()
