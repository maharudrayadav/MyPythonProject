import os
import sys
import base64

# ✅ Check if name is provided
if len(sys.argv) < 2:
    print("❌ Error: Name is required")
    sys.exit(1)

person_name = sys.argv[1]  # Read name from arguments
dataset_path = "dataset"
person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

# ✅ Read Base64 image from stdin (for API communication)
image_data = sys.stdin.read().strip()
if not image_data:
    print("❌ Error: No image received")
    sys.exit(1)

# ✅ Decode Base64 and save the image
image_data = image_data.split(",")[1]  # Remove "data:image/jpeg;base64,"
image_bytes = base64.b64decode(image_data)

image_path = os.path.join(person_folder, f"{len(os.listdir(person_folder))}.jpg")
with open(image_path, "wb") as f:
    f.write(image_bytes)

print(f"✅ Image saved: {image_path}")
