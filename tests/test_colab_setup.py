"""Exercise the actual setup cell from root, notebooks and a fresh Colab-like folder."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/phases_1_2_preprocessing_eda.ipynb'


class ColabSetupTests(unittest.TestCase):
    def setUp(self):
        self.original_cwd = Path.cwd()
        self.original_path = sys.path.copy()
        self.nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.source = ''.join(self.nb['cells'][2]['source'])

    def tearDown(self):
        os.chdir(self.original_cwd)
        sys.path[:] = self.original_path

    def run_setup(self):
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(compile(self.source, str(NOTEBOOK), 'exec'), namespace)
        return namespace

    def test_local_root_and_notebooks_reuse_checkout(self):
        for directory in (ROOT, ROOT / 'notebooks'):
            with self.subTest(directory=directory):
                os.chdir(directory)
                with patch('subprocess.run') as clone, patch('subprocess.check_call') as pip:
                    state = self.run_setup()
                    clone.assert_not_called()
                    self.assertEqual(state['REPO_ROOT'], ROOT)
                    self.assertEqual(pip.call_args.args[0][-1], str(ROOT / 'requirements.txt'))

    def test_fresh_session_clones_before_install_and_reuses_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            os.chdir(directory)
            calls = []

            def clone(args, **kwargs):
                calls.append('clone')
                destination = Path(args[-1])
                shutil.copytree(ROOT / 'src', destination / 'src', ignore=shutil.ignore_patterns('__pycache__'))
                shutil.copy2(ROOT / 'requirements.txt', destination / 'requirements.txt')

            with patch('subprocess.run', side_effect=clone), patch('subprocess.check_call', side_effect=lambda args: calls.append('pip')):
                state = self.run_setup()
                self.assertEqual(calls, ['clone', 'pip'])
                local_file = state['REPO_ROOT'] / 'src/data_preprocessing.py'
                local_file.write_text('# session edit', encoding='utf-8')
                self.run_setup()
                self.assertEqual(calls, ['clone', 'pip', 'pip'])
                self.assertEqual(local_file.read_text(encoding='utf-8'), '# session edit')
            os.chdir(self.original_cwd)

    def test_notebook_defers_function_implementation_to_src(self):
        import ast
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code':
                tree = ast.parse(''.join(cell['source']))
                self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
