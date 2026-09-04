"""
Example: preprocess a dataset using the micm-nlp pipeline.

This example loads an FTP-reframed XStoryCloze dataset directly from the
HuggingFace Hub (`mikaberidze/xstory-cloze-ftp`), tokenizes it for a
decoder-only LM (BLOOM-560M), and saves the tokenized output locally.

Usage:
    micm-nlp init-examples
    python examples/preprocess_dataset.py --config micm-nlp-examples/xsc_preprocess.yml
"""

import micm_nlp
import micm_nlp.utils as utils
from micm_nlp.config import CONFIG
from micm_nlp.pipeline import preprocess_dataset

if __name__ == '__main__':
    # Sets the workspace root, so artefacts/ lands somewhere deliberate. Reads
    # PROJECT_ROOT_PATH from the environment or .env when called with no argument.
    micm_nlp.init()
    config_path = utils.parse_script_args()
    config = CONFIG.from_yaml(config_path)
    dataset = preprocess_dataset(config)
