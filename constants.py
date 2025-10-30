NAME_TO_CLASS = {"antiproton": 0, "proton": 1, "pion+": 2}

CLASS_TO_NAME = {v: k for k, v in NAME_TO_CLASS.items()}

MODEL_CONFIG_FILE = "model_config.yaml"
OPTIMIZER_CONFIG_FILE = "optimizer_config.yaml"

OPTUNA_FILE = "optuna_config.yaml"

TARGET_SHAPE = (120, 30)  # W H for cv2.resize

TRAIN_DATASET_LEN_80_360 = 2079400
VAL_DATASET_LEN_80_360 = 594100
TEST_DATASET_LEN_80_360 = 296939
