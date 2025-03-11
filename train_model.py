import cv2
import numpy as np
import os

dataset_path = "dataset"
LBPH_model = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels():
    images, labels = [], []
    label_dict = {}
    label_id = 0

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        label_dict[label_id] = person
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            images.append(img)
            labels.append(label_id)

        label_id += 1

    return images, labels, label_dict

print("🔄 Training LBPH model...")
images, labels, label_dict = get_images_and_labels()

if len(images) == 0:
    print("❌ Error: No images found. Please add training data.")
else:
    LBPH_model.train(images, np.array(labels))
    LBPH_model.save("lbph_model.xml")
    print("✅ Model trained and saved as lbph_model.xml")
