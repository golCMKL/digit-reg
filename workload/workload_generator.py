import argparse
import io
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, ImageDraw

API_URL = "http://localhost:5000/predict"
_thread_local = threading.local()


def get_session():
    """Use one HTTP session per worker thread."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def generate_image():
    """Create a synthetic 28x28 grayscale image."""
    image = Image.new("L", (28, 28), color=255)
    draw = ImageDraw.Draw(image)

    # Draw a rough digit-like shape using random connected lines.
    points = [
        (random.randint(5, 22), random.randint(3, 24))
        for _ in range(random.randint(3, 7))
    ]
    draw.line(points, fill=random.randint(0, 100), width=2)

    # Add occasional extra strokes to vary the input.
    if random.random() < 0.35:
        x, y = random.randint(4, 23), random.randint(4, 23)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=0)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def send_request(index, timeout):
    """Submit one image and record the outcome."""
    image_bytes = generate_image()
    start = time.perf_counter()

    try:
        response = get_session().post(
            API_URL,
            files={
                "image": (
                    f"synthetic_{index}.png",
                    image_bytes,
                    "image/png",
                )
            },
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        try:
            body = response.json()
        except ValueError:
            body = {}

        success = response.ok and "prediction" in body

        return {
            "success": success,
            "status": response.status_code,
            "latency_ms": latency_ms,
            "cached": body.get("cached", False),
            "error": body.get("error", "")
            if not success else "",
        }

    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "success": False,
            "status": "connection/timeout",
            "latency_ms": latency_ms,
            "cached": False,
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate workload for the digit recognition API"
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    if args.requests < 1 or args.workers < 1:
        parser.error("--requests and --workers must be positive")

    print(f"API: {API_URL}")
    print(f"Requests: {args.requests}")
    print(f"Concurrent workers: {args.workers}")
    print("Generating workload...")

    results = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(send_request, i, args.timeout)
            for i in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - start
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = sorted(r["latency_ms"] for r in results)
    cache_hits = sum(r["cached"] is True for r in successful)

    def percentile(values, p):
        if not values:
            return 0
        index = max(0, int((p / 100) * len(values) + 0.999) - 1)
        return values[min(index, len(values) - 1)]

    print("\n--- Workload Summary ---")
    print(f"Total requests:    {len(results)}")
    print(f"Successful:        {len(successful)}")
    print(f"Failed:            {len(failed)}")
    print(f"Error rate:        {len(failed) / len(results) * 100:.2f}%")
    print(f"Elapsed time:      {elapsed:.2f} s")
    print(f"Throughput:        {len(results) / elapsed:.2f} requests/s")
    print(f"Average latency:   {sum(latencies) / len(latencies):.2f} ms")
    print(f"P95 latency:       {percentile(latencies, 95):.2f} ms")
    print(f"Cache hits:        {cache_hits}")
    print(
        f"Cache hit rate:    "
        f"{cache_hits / len(successful) * 100:.2f}%"
        if successful else "Cache hit rate:    N/A"
    )

    if failed:
        print("\nFailure details:")
        for status, count in Counter(str(r["status"]) for r in failed).items():
            print(f"  {status}: {count}")


if __name__ == "__main__":
    main()