from tools import *

def present_results_on_models(dir_images, save_dir, models, cfg: DictConfig):
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)

    for path in tqdm(glob(os.path.join(dir_images, "*"))):
        name = os.path.basename(path).split(".")[0]

        """ Считывание изображения как 1 пакета"""
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        images = [image]

        for model in models:
            y = predict(image, model, cfg)
            masked_img = masked_image(image, y)
            images.append(masked_img)

        # Объединение всех изображений в одно
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
