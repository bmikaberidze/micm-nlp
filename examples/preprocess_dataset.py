"""
Example: preprocess a dataset using the micm-nlp pipeline.

This example loads an FTP-reframed XStoryCloze dataset directly from the
HuggingFace Hub (`mikaberidze/xstory-cloze-ftp`), tokenizes it for a
decoder-only LM (BLOOM-560M), and saves the tokenized output locally.

Usage:
    python examples/preprocess_dataset.py --config examples/configs/xsc_preprocess.yml
"""

import micm_nlp.utils as utils
from micm_nlp.config import CONFIG
from micm_nlp.pipeline import preprocess_dataset

if __name__ == '__main__':
    config_path = utils.parse_script_args()
    config = CONFIG.from_yaml(config_path)
    dataset = preprocess_dataset(config)
