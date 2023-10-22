import os
import zipfile
from tqdm import tqdm

import cv2
import numpy as np
from uuid import uuid1
import albumentations as alb

augmentor = alb.Compose(
    [
        alb.RandomCrop(width=224, height=224),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
        alb.VerticalFlip(p=0.5),
        alb.Rotate(limit=75, p=0.5),
        alb.ISONoise(intensity=(0.05, 0.15)),
        alb.ZoomBlur(),
    ]
)


with zipfile.ZipFile("archive.zip", "r") as zip_ref:
    zip_ref.extractall("dataset")

print("extraction done")

print("data preprocessing started")

image_names = [
    os.path.join(root, fl)
    for root, dirs, files in os.walk("dataset/colored_images/colored_images")
    for fl in files
]

for img_name in tqdm(image_names, desc="processing :"):
    image = cv2.imread(img_name)
    image = image[
        int(image.shape[0] * 0.05) : int(image.shape[0] * 0.95),
        int(image.shape[1] * 0.08) : int(image.shape[1] * 0.9),
    ]
    cv2.imwrite(img_name, image)


os.makedirs("augmented_dataset/normal", exist_ok=True)
os.makedirs("augmented_dataset/infected", exist_ok=True)
for img_name in tqdm(image_names, desc="preprocessing images"):
    for i in range(
        np.random.randint(1, 3) if img_name.split("/")[-2] == "No_DR" else 2
    ):
        image = cv2.resize(cv2.imread(img_name), (224, 224))
        augmented = augmentor(image=image)["image"]
        img_path = f"augmented_dataset/{'normal' if img_name.split('/')[-2]=='No_DR' else 'infected' }/{uuid1()}.jpeg"
        cv2.imwrite(img_path, augmented)

image_names = [
    os.path.join(root, fl)
    for root, dirs, files in os.walk("dataset/colored_images/colored_images")
    for fl in files
    if "No_DR" not in root
]


class_count = {}
classes = os.listdir("dataset/colored_images/colored_images")
classes.remove("No_DR")
for i in classes:
    class_count[i] = len(os.listdir(f"dataset/colored_images/colored_images/{i}"))
    os.makedirs(f"severity_augmented_dataset/{i}", exist_ok=True)

for img_name in tqdm(image_names, desc="preprocessing images"):
    aug_count = int(image_names.__len__() / class_count[img_name.split("/")[-2]])
    for i in range(np.random.randint(1, 3) if aug_count == 1 else aug_count):
        image = cv2.resize(cv2.imread(img_name), (224, 224))
        augmented = augmentor(image=image)["image"]
        img_path = (
            f"severity_augmented_dataset/{img_name.split('/')[-2]}/{uuid1()}.jpeg"
        )
        cv2.imwrite(img_path, augmented)
