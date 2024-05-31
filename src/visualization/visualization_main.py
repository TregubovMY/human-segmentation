import sys
import os
import image
import utils
from src.models.deeplabv3plus import model_deepLabV3_plus
from src.models.u2_net import model_U2_Net
from src.models.u_net import model_U_Net
import hydra
from omegaconf import DictConfig
from src.utils.utils import folder_path


@hydra.main(config_path="./../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    base_dir = folder_path()
    input_dir = os.path.join(base_dir, 'data', 'test', 'konf')
    output_dir = os.path.join(base_dir, 'data', 'test', 'results')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = [
        model_U_Net(),
        model_deepLabV3_plus(),
        model_U2_Net(),
    ]

    image.present_results_on_models(input_dir, output_dir, models, cfg)

if __name__ == "__main__":
    main()
