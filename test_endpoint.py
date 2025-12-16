import requests
import base64

def test_endpoint():
    url = "http://127.0.0.1:5000/v1/chat/completions"
    
    # Test case 1: Image URL
    print("Testing with Image URL...")
    payload_url = {
        "model": "joytag",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What tags are in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/fancyfeast/joytag/resolve/main/assets/screenshot_20231220a.jpg"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload_url)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_endpoint()
