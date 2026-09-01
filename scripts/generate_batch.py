import os
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
    
    print(f"Initializing Vertex AI (Project: {PROJECT_ID}, Location: {LOCATION})...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    print("Loading Imagen 3 model...")
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

    start_time = time.time()
    max_duration_seconds = DURATION_HOURS * 3600
    iteration = 1

    print(f"Starting continuous loop for {DURATION_HOURS} hours...")

    while (time.time() - start_time) < max_duration_seconds:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/image_{timestamp}_idx{iteration}.png"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating image #{iteration}...")
        try:
            response = model.generate_images(
                prompt=PROMPT,
                number_of_images=1,
                aspect_ratio="4:3",
                safety_filter_level="block_medium_and_above",
                person_generation="allow_adult"
            )

            if response.images:
                response.images[0].save(location=filename, include_generation_parameters=False)
                print(f"Saved: {filename}")
            else:
                print("No image returned from API.")

        except Exception as e:
            print(f"Generation error on iteration #{iteration}: {e}")

        iteration += 1
        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining <= 0:
            break

        sleep_time = min(INTERVAL_SECONDS, remaining)
        print(f"Sleeping for {int(sleep_time)}s. ({int(remaining / 60)} mins remaining)...")
        time.sleep(sleep_time)

    print("\nTarget duration reached. Exiting script.")

if __name__ == "__main__":
    main()
