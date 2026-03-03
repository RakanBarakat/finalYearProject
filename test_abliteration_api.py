import os
import sys
import json
import textwrap

import requests


def main() -> None:
    """
    Minimal Abliteration API smoke test.

    Usage (PowerShell on Windows):

        $env:ABLIT_KEY = "ak_..."   # your real key
        python test_abliteration_api.py

    Optional arguments:
        python test_abliteration_api.py "your test prompt"
    """

    api_key = os.environ.get("ABLIT_KEY")
    if not api_key:
        print("ERROR: Please set the ABLIT_KEY environment variable to your Abliteration API key.")
        print('Example (PowerShell):  $env:ABLIT_KEY = "ak_...your_key_here..."')
        sys.exit(1)

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello. Respond in one short sentence."

    url = "https://api.abliteration.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "abliterated-model",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        # Keep this tiny so it barely uses any credits
        "max_tokens": 32,
        "temperature": 0.0,
    }

    print(f"Sending test request to {url} ...")
    print(f"Prompt: {prompt!r}")

    try:
        res = requests.post(url, headers=headers, json=body, timeout=30)
    except Exception as e:
        print(f"\nERROR: Request failed: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"\nHTTP status: {res.status_code}")

    try:
        data = res.json()
    except Exception:
        print("Raw response:")
        print(res.text[:500])
        sys.exit(0)

    print("\nRaw JSON (truncated):")
    print(textwrap.shorten(json.dumps(data, indent=2), width=800, placeholder=" ..."))

    if res.status_code == 200:
        try:
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            print("\nModel reply:")
            print(content)
        except Exception:
            pass
    else:
        print("\nNon-200 from Abliteration; above JSON usually includes an 'error' field.")


if __name__ == "__main__":
    main()

