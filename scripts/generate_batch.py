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

def load_image_model():
    # Try the most common stable model identifiers in order
    model_candidates = [
        "imagen-3.0-generate-001",
        "imagegeneration@006",
        "imagegeneration@005"
    ]
    for name in model_candidates:
        try:
            print(f"Trying to load model: {name}...")
            model = ImageGenerationModel.from_pretrained(name)
            print(f"Successfully loaded model: {name}")
            return model
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            
    raise RuntimeError("Could not load any valid ImageGenerationModel in Vertex AI.")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("==================================================")
    print(f"Initializing Vertex AI (Project: {PROJECT_ID}, Location: {LOCATION})")
    print("==================================================")

    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = load_image_model()
    except Exception as e:
        print(f"Failed to initialize Vertex AI / Model: {e}")
        sys.exit(1)

    start_time = time.time()
    max_duration_seconds = DURATION_HOURS * 3600
    iteration = 1

    print(f"Starting execution loop for ~{DURATION_HOURS} hours...")

    while (time.time() - start_time) < max_duration_seconds:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/passport_{timestamp}_{iteration}.png"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Attempting generation #{iteration}...")

        try:
            response = model.generate_images(
                prompt=PROMPT,
                number_of_images=1,
                aspect_ratio="4:3"
            )

            if response and response.images:
                response.images[0].save(location=filename, include_generation_parameters=False)
                print(f"Successfully saved image to: {filename}")
            else:
                print("API returned no images.")

        except Exception as err:
            print(f"API Error during iteration #{iteration}: {err}")

        iteration += 1
        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining <= 0:
            print("Time limit reached.")
            break

        sleep_time = min(INTERVAL_SECONDS, remaining)
        print(f"Waiting {int(sleep_time)}s before next attempt ({int(remaining / 60)}m remaining)...")
        time.sleep(sleep_time)

    print("\nSession complete.")

if __name__ == "__main__":
    main()
