#testing script
#supporting script

import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/recommend"

def test_case(payload, label):
    print(f"\n=== {label} ===")
    print("Request:", json.dumps(payload, indent=2))
    resp = requests.post(BASE_URL, json=payload)
    print("Status code:", resp.status_code)
    try:
        data = resp.json()
        print("Response JSON keys:", list(data.keys()))
        if "recommendations" in data:
            print("Number of recommendations:", len(data["recommendations"]))
            if data["recommendations"]:
                print("First song:", data["recommendations"][0].get("title"), "-", data["recommendations"][0].get("artist"))
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Raw response text:", resp.text)

if __name__ == "__main__":
    # Case 1: basic
    test_case(
        {
            "text": "I feel stressed but hopeful",
            "top_k": 5,
            "same_emotion_only": True,
            "artist": None,
            "sort_by": "similarity",
        },
        "Case 1: basic"
    )

    # Case 2: artist filter
    test_case(
        {
            "text": "I feel stressed but hopeful",
            "top_k": 5,
            "same_emotion_only": True,
            "artist": "Taylor Swift",
            "sort_by": "similarity",
        },
        "Case 2: Taylor Swift"
    )

    # Case 3: same_emotion_only = false
    test_case(
        {
            "text": "I feel stressed but hopeful",
            "top_k": 5,
            "same_emotion_only": False,
            "artist": None,
            "sort_by": "similarity",
        },
        "Case 3: same_emotion_only = False"
    )
