from .tools import *
import tensorflow as tf
from tqdm import tqdm
from glob import glob

def apply_mask(image, mask, mode="multiply"):
    """Applies the mask to the image based on the specified mode."""
    if mode == "multiply":
        return image * mask
    elif mode == "concatenate":
        return np.concatenate([image, mask], axis=1)
    elif mode == "concatenate_multiplied":
        multiplied = image * mask
        mask = np.repeat(mask, 3, axis=2) 
        return np.concatenate([image, mask, multiplied], axis=1)
    else:
        return mask  # Default: Return only the mask

def present_results_on_models(dir_images, save_dir, models, cfg: DictConfig, output_mode="multiply"):
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)

    for path in tqdm(glob(os.path.join(dir_images, "*"))):
        name = os.path.basename(path).split(".")[0]
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        images = [image]
        for model in models:
            y = predict(image, model, cfg)
            masked_img = apply_mask(image, y, output_mode)
            images.append(masked_img)

        result_img = np.concatenate(images, axis=1)
        save_dir_name = os.path.join(save_dir, f"{name}_all_models.png")
        save(result_img, save_dir_name)



def show_pred_image(image, model):
    if image is not None:
        y = predict(image, model)
        h, _, _ = image.shape
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1)
        cv2.imshow(cat_images)
    else:
        print("Изображение не загружено.")
