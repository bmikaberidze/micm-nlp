"""
Example: fine-tune and evaluate a model using the micm-nlp pipeline.

This example fine-tunes BLOOM-560M with the Cross-Prompt Encoder (XPE) PEFT
method on the FTP-reframed XStoryCloze dataset (Arabic split) loaded from
the HuggingFace Hub (`mikaberidze/xstory-cloze-ftp`), and evaluates after
training.

Usage:
    micm-nlp init-examples
    python examples/run_model.py --config micm-nlp-examples/xsc_finetune.yml
"""

import micm_nlp
import micm_nlp.utils as utils
from micm_nlp.config import CONFIG
from micm_nlp.pipeline import run

if __name__ == '__main__':
    # Sets the workspace root, so artefacts/ lands somewhere deliberate. Reads
    # PROJECT_ROOT_PATH from the environment or .env when called with no argument.
    micm_nlp.init()
    config_path = utils.parse_script_args()
    config = CONFIG.from_yaml(config_path)
    model, test_output = run(config)
