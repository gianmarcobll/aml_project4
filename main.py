import logging
import argparse
import sys
import numpy as np
from hyperparameters_tuning import run_hpo
from evaluate_policy import evaluate_policy

logger = None
seed = 0
np.random.seed(seed)

if __name__ == '__main__':
    print("Hello")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", help="Selects between training and evaluation mode.", default=False,
        choices=["train", "evaluate", "train-vision-based"], required=True)
    parser.add_argument(
        "--train_domain", help="Domain Hopper environment. Select between source and target.",
        choices=["source", "target"], required=False)
    parser.add_argument("--render", help="Render the environment", default=False, required=False, type=bool)
    parser.add_argument(
        "--evaluate_domain", help="Domain Hopper environment. Select between source and target.", default=False,
        choices=["source", "target"], required=False)
    parser.add_argument("--model_path", help="Path to the RL model", default=False, type=str, required=False)
    parser.add_argument("--udr_enable", help="Enable UDR", default=False, type=bool, required=False)
    parser.add_argument("--feature_extractor",  help="Select feature extractor for vision based approach",
        choices=["resnet", "nature-cnn", "shufflenet"], required=False)
    parser.add_argument("--combined_extractor", help="Enable features extraction using feature_extractor and multi-layer-perceptron", default=False,
        type=bool, required=False)
    parser.add_argument("--udr_extended_enable", help="Enable UDR extended", default=False, type=bool, required=False)
    parser.add_argument("--pretrained_extractor",  help="Enable for transfer learning", default=False, type=bool, required=False)
    parser.add_argument("--vision_based", help="Enable if you wanna evaluate a vision based model", default=False, type=bool,
                        required=False)

    args = parser.parse_args()

    match args.mode:
        case 'train':
            params_hpo = {
                'train_domain': args.train_domain,
                'evaluate_domain': args.evaluate_domain,
                'study_name': f"{args.train_domain}_{args.evaluate_domain}" if not args.udr_enable else f"{args.train_domain}_{args.evaluate_domain}_udr",
                'udr_enable': args.udr_enable
            }
            run_hpo(params_hpo)
        case 'evaluate':
            params = {
                'evaluate_domain': args.evaluate_domain,
                'model_path': args.model_path,
                'vision_based': args.vision_based
            }
            evaluate_policy(params)

        case 'train-vision-based':
            print("Not implemented!")
