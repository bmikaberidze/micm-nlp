"""``python -m micm_nlp`` must reach the same CLI as the ``micm-nlp`` script.
Runs the interpreter as a subprocess so the module-as-main path is what is
exercised, not an import."""

import subprocess
import sys


def test_module_main_shows_help():
    proc = subprocess.run(
        [sys.executable, '-m', 'micm_nlp', '--help'],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'init-examples' in proc.stdout


def test_module_main_without_command_fails():
    proc = subprocess.run(
        [sys.executable, '-m', 'micm_nlp'],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
