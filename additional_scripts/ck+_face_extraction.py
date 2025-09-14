import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import shutil

ckplus_dir = 'Datasets/CK+'
output_dir = 'Datasets/CK+_extracted_faces'

def extract():
    image_size = (128, 128)
    use_example_frame_only = False

    detector = MTCNN()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for emotion in tqdm(os.listdir(ckplus_dir), desc='Emotions'):
        emotion_path = os.path.join(ckplus_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue

        images = sorted([f for f in os.listdir(emotion_path) if f.endswith('.png')])
        if not images:
            continue

        selected_images = [images[0]] if use_example_frame_only else images

        for img_name in selected_images:
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            detections = detector.detect_faces(img_rgb)

            if len(detections) == 0:
                continue

            largest_face = max(detections, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = largest_face['box']
            x, y = max(0, x), max(0, y)
            face_img = img_rgb[y:y+h, x:x+w]

            face_resized = cv2.resize(face_img, image_size)
            save_dir = os.path.join(output_dir, emotion)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))

def delete_duplicates():
    for emotion in os.listdir(ckplus_dir):
        images = set()
        emotion_path = os.path.join(ckplus_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            if not os.path.isfile(img_path):
                continue

            img_hash = img_name[:4]
            if img_hash in images:
                os.remove(img_path)
            else:
                images.add(img_hash)

def split_ds(train_to_test_ratio=0.8):
    for emotion in os.listdir(output_dir):
        emotion_path = os.path.join(output_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue

        images = [f for f in os.listdir(emotion_path) if f.endswith('.png')]
        if not images:
            continue
        np.random.shuffle(images)

        split_idx = int(len(images) * train_to_test_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        train_dir = os.path.join(output_dir, 'train', emotion)
        test_dir = os.path.join(output_dir, 'test', emotion)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        for img_name in train_images:
            shutil.move(os.path.join(emotion_path, img_name), os.path.join(train_dir, img_name))
        for img_name in test_images:
            shutil.move(os.path.join(emotion_path, img_name), os.path.join(test_dir, img_name))
        
        if not os.listdir(emotion_path):
            os.rmdir(emotion_path)

if __name__ == "__main__":
    delete_duplicates()
    extract()
    split_ds()