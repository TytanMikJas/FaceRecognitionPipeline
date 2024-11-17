import os
import shutil
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    source_dir = "D:/Projects/FaceRecognitionPipeline/datasets"
    output_dir = "D:/Projects/FaceRecognitionPipeline/datasets"
    categories = ["hand", "low_qual", "hand_low_qual", "normal"]

    train_ratio = 0.8

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in categories:
        print("Processing category:", category)
        category_dir = os.path.join(source_dir, category)
        images = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)
        
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)
        
        for img in train_images:
            shutil.copy(os.path.join(category_dir, img), os.path.join(train_category_dir, img))
        
        for img in test_images:
            shutil.copy(os.path.join(category_dir, img), os.path.join(test_category_dir, img))

    print("Data successfully split into train and test datasets!")
