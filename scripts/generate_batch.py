import os
import sys
import time
import io
from datetime import datetime

from PIL import Image
from google import genai
from google.genai import types

PROJECT_ID = os.environ.get(
    "GOOGLE_CLOUD_PROJECT",
    "durable-student-507318-t0"
)
LOCATION = os.environ.get(
    "GOOGLE_CLOUD_LOCATION",
    "global"
)

DURATION_HOURS = float(os.environ.get("DURATION_HOURS", 5.8))
# Changed from 120 to 5 seconds to run almost continuously
INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", 5))
OUTPUT_DIR = "output"

MODEL_NAME = "gemini-3.1-flash-image"

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

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"Connecting to {PROJECT_ID} in {LOCATION}...",
        flush=True
    )

    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "Client initialized successfully.",
            flush=True
        )

    except Exception as e:
        print(f"Initialization error: {e}", flush=True)
        sys.exit(1)

    start_time = time.time()
    max_duration_seconds = DURATION_HOURS * 3600
    iteration = 1

    print(
        f"Starting loop using {MODEL_NAME}...",
        flush=True
    )

    while (time.time() - start_time) < max_duration_seconds:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            OUTPUT_DIR,
            f"passport_{timestamp}_{iteration}.png"
        )

        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
            f"Requesting generation #{iteration}...",
            flush=True
        )

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

                            image = Image.open(
                                io.BytesIO(image_bytes)
                            )

                            image.save(filename)

                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"SUCCESS: Saved {filename}",
                                flush=True
                            )

                            image_saved = True
                            break

                    if image_saved:
                        break

            if not image_saved:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "No image returned.",
                    flush=True
                )

                if response.text:
                    print("Model response:", response.text, flush=True)

        except Exception as err:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"API ERROR: {err}",
                flush=True
            )

        iteration += 1

        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining <= 0:
            print("Session completed.", flush=True)
            break

        sleep_time = min(INTERVAL_SECONDS, remaining)

        # Skip the sleeping print statement if the wait time is 0 to keep logs clean
        if sleep_time > 0:
            print(
                f"Sleeping {int(sleep_time)}s...",
                flush=True
            )
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
