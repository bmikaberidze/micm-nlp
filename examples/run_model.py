"""
Example: fine-tune and evaluate a model using the micm-nlp pipeline.

This example fine-tunes BLOOM-560M with the Cross-Prompt Encoder (XPE) PEFT
method on the FTP-reframed XStoryCloze dataset (Arabic split) loaded from
the HuggingFace Hub (`mikaberidze/xstory-cloze-ftp`), and evaluates after
training.

Usage:
    python examples/run_model.py --config examples/configs/xsc_finetune.yml
"""

import micm_nlp.utils as utils
from micm_nlp.config import CONFIG
from micm_nlp.pipeline import run

if __name__ == '__main__':
    config_path = utils.parse_script_args()
    config = CONFIG.from_yaml(config_path)
    model, test_output = run(config)
