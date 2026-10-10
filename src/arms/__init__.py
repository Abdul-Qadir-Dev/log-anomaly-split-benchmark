from .deeplog import DeepLogArm
from .iforest import IForestArm
from .llm_zeroshot import LLMZeroShotArm
from .ocsvm import OCSVMArm
from .pca import PCAArm
from .semantic_knn import SemanticKNNArm
from .supervised_lgbm import SupervisedLGBMArm

ARMS = {
    "pca": PCAArm,
    "iforest": IForestArm,
    "ocsvm": OCSVMArm,
    "deeplog": DeepLogArm,
    "semantic_knn": SemanticKNNArm,
    "supervised_lgbm": SupervisedLGBMArm,
    "llm_zeroshot": LLMZeroShotArm,
}

UNSUPERVISED_ARMS = ["pca", "iforest", "ocsvm", "deeplog", "semantic_knn"]
SUPERVISED_ARMS = ["supervised_lgbm"]
SAMPLE_BASED_ARMS = ["llm_zeroshot"]
NON_LLM_ARMS = UNSUPERVISED_ARMS + SUPERVISED_ARMS
