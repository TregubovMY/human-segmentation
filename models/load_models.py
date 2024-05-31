import gdown
import os
from tqdm import tqdm

def download_model(url: str, output_path: str):
    """Загружает файл модели с Google Drive с прогрессбаром."""

    if os.path.exists(output_path):
        print(f"Модель уже существует: {output_path}")
    else:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=f"Загрузка {output_path}") as t:
            def download_callback(downloaded: int, total_size: int):
                if total_size:
                    t.total = total_size
                    t.update(downloaded - t.n)

            gdown.download(url, output_path, quiet=False, use_cookies=False, callback=download_callback)
        print(f"Модель {output_path} успешно загружена")

if __name__ == "__main__":
    model_urls = {
        "u2_net.h5": "https://drive.google.com/uc?id=14AnilEmPdmbZqTlM2YxjTkKFZwFI7Hy1&export=download",
        "u_net.h5":  "https://drive.google.com/uc?id=1HqHml3gYYESKG31QjHTqDCmHXT28Qqy0&export=download",
        "deepLabV3_plus.h5": "https://drive.google.com/uc?id=1b2xX-OqOVWyLFW-wvFejov25F0Acns1h&export=download",
    }

    for model_name, url in model_urls.items():
        output_path = os.path.join("./models", model_name)
        download_model(url, output_path)