import argparse
import gdown
import zipfile
import os
from tqdm import tqdm

def download_from_google_drive(url: str, output_path: str):
    """Загружает файл с Google Drive с прогрессбаром и распаковывает, если это архив."""
    if os.path.exists(output_path):
        print(f"Архив уже существует: {output_path}")
    else:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc="Загрузка набора данных person") as t:
            # Определяем функцию обратного вызова для обновления tqdm
            def download_callback(downloaded: int, total_size: int):
                # Обновляем tqdm только если total_size известен
                if total_size:
                    t.total = total_size
                    t.update(downloaded - t.n)

            # Загружаем файл с помощью gdown, передавая функцию обратного вызова
            gdown.download(url, output_path, quiet=False, use_cookies=False)

    if output_path.endswith('.zip') and not os.path.exists(output_path[:-4]):
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc="Распаковка", total=len(zip_ref.infolist())):
                zip_ref.extract(member, './data/raw')
    else:
        print(f"Данные уже распакованы: {output_path[:-4]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Загрузка датасета')
    parser.add_argument('--source', choices=['people_segmentation', 'P3M-10k'], required=True,
                        help='Какой набор данных: people_segmentation или P3M-10k')
    args = parser.parse_args()

    person_url = "https://drive.google.com/uc?id=1Ax8IqUPC0SH5UwgrGSyJIlCVJCU6pdYr&export=download"
    P3M_url = "https://drive.google.com/uc?id=1BPPLckL5O9EbM9k5zJyUTKHQjpyUrzdf&export=download"  
    output_path = f'./data/raw/{args.source}.zip'

    if args.source == 'people_segmentation':
        download_from_google_drive(person_url, output_path)
        print("Датасет person успешно загружен и распакован в ./data/raw/P3M-10k")
    elif args.source == 'P3M-10k':
        download_from_google_drive(P3M_url, output_path)
        print("Датасет P3M успешно загружен и распакован в ./data/raw/people_segmentation")
