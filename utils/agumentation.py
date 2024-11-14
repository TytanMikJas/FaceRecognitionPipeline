import os
import matplotlib.pyplot as plt
import albumentations as A
import cv2

"""
    Customly defined augmentations for creation of our own dataset.
    The agumentation helps to create images of low quality, with different conditions
    that can occur while scanning your face for recignition.
"""
custom_agumentations = [
            A.RandomBrightnessContrast(brightness_limit=(-0.42, -0.27), contrast_limit=(-0.1, 0), brightness_by_max=True , p=1),
            A.RandomBrightnessContrast(brightness_limit=(0.28, 0.4), contrast_limit=(0, -0.1), brightness_by_max=True , p=1),
            A.RandomSunFlare(num_flare_circles_range=(14, 17), flare_roi=(0.2, 0, 0.8, 0.2), src_radius=800, method='physics_based', p=1),
            A.RandomRain(p=1, rain_type='default', slant_range=(-7, 7), drop_width=1, drop_length=5, brightness_coefficient=0.85, blur_value=6),
            A.GaussianBlur(blur_limit=(7, 11), sigma_limit=(1, 10), p=1),
            A.Downscale(scale_range=(0.1, 0.2), p=1),
            A.GaussNoise(var_limit=(300, 500), mean=0, per_channel=True, p=1)
        ]

basic_agumentations = [
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5)
    ]

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def apply_basic_augmentations(image):
    """
    Apply a set of basic augmentations suitable for CNN training.

    Parameters:
    - image: The input image in RGB format.

    Returns:
    - aug_image: Augmented image.
    """
    augmentation_pipeline = A.Compose(basic_agumentations)
    return augmentation_pipeline(image=image)["image"]

def apply_custom_augmentations(image):
    """
    Apply exactly two random augmentations from a set of augmentations to the image.

    Parameters:
    - image_path: Path to the input image.

    Returns:
    - aug_image: Augmented image.
    """    
    augmentation_pipeline = A.Compose([
        A.OneOf(custom_agumentations, p=1),
        A.OneOf(custom_agumentations, p=1)
    ])
    return augmentation_pipeline(image=image)["image"]


def augment_directory(source_dir, target_dir, augmentation_func, num_augmentations=1):
    """
    Apply an augmentation function to all images in a directory and save the results.

    Parameters:
    - source_dir: Directory containing the original images.
    - target_dir: Directory to save the augmented images.
    - augmentation_func: Function to apply for augmentations (e.g., apply_basic_augmentations).
    - num_augmentations: Number of augmented copies to generate per image.

    Returns:
    - None
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_dir, filename)
            try:
                image = read_image(image_path)
                for i in range(num_augmentations):
                    aug_image = augmentation_func(image)
                    aug_filename = f"{os.path.splitext(filename)[0]}_aug_{i + 1}.jpg"
                    aug_image_path = os.path.join(target_dir, aug_filename)
                    cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# CAN DELETE LATER, JUST FOR SHOWCASE
if (__name__ =="__main__"):
    image_path = "../assets/model.png"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    basic_augmented_image = apply_basic_augmentations(image_rgb)
    custom_augmented_image = apply_custom_augmentations(image_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Original, Basic Augmentation, and Custom Augmentation")

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(basic_augmented_image)
    axes[1].set_title("Basic Augmentation")
    axes[1].axis("off")

    axes[2].imshow(custom_augmented_image)
    axes[2].set_title("Custom Augmentation")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
