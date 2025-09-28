import requests
import json
from PIL import Image

# Test the API with the Kolkata coordinates
test_data = {
    "latitude": 22.640482011240668,
    "longitude": 88.41506251385172,
    "start_date": "2024-01-01",
    "end_date": "2024-09-28",
    "parameters": {
        "red_band": "B4",
        "nir_band": "B8",
        "rgb_b1": "B4",
        "rgb_b2": "B3",
        "rgb_b3": "B2",
        "grid_width": 1,
        "grid_height": 1,
        "output_format": "png",
        "opacity": 0.8,
        "cloud_percentage": 20,
        "buffer_size": 500,
        "image_scale": 5,
        "image_quality": 100,
        "enhance_contrast": True,
        "upscale_factor": 2.2,
        "detect_farmland": True,
        "farmland_confidence": 0.6
    }
}

try:
    print("Testing API with Kolkata coordinates...")
    response = requests.post(
        'https://vegetation-stress-mapping-production.up.railway.app/classify-location',
        json=test_data,
        timeout=120  # 2 minute timeout
    )
    
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        # Save the image
        with open('test_output.png', 'wb') as f:
            f.write(response.content)
        print(f"✅ Success! Image saved as test_output.png")
        print(f"Image size: {len(response.content)} bytes")
        
        # Try to analyze the image
        try:
            img = Image.open('test_output.png')
            print(f"Image dimensions: {img.size}")
            print(f"Image mode: {img.mode}")
            print(f"Image format: {img.format}")
            
            # Check if it's mostly transparent/empty
            if hasattr(img, 'getbands'):
                bands = img.getbands()
                print(f"Image bands: {bands}")
                
        except Exception as img_error:
            print(f"Error analyzing image: {img_error}")
            
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response content type: {response.headers.get('content-type')}")
        try:
            error_text = response.text
            print(f"Error response: {error_text}")
        except:
            print("Could not decode error response")
        
except Exception as e:
    print(f"❌ Request failed: {e}")