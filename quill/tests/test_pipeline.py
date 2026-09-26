"""Offline checks for the public task paths after the package restructure."""

import json
import re
import tempfile
import unittest

from benchmarks.dependeval import run_dependeval
from benchmarks.dependeval.python_deps import build_python_deps_plugins
from benchmarks.repoqa import RepoQAFunctionExtractor, RepoQASearchPrompt
from quill import TaskSpec, V5Pipeline
from quill.cache import PixelMemCache
from quill.plugins import PluginSet


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskSpec(domain="test", description="test", input_schema={}, query={})

    def test_python_dependency_card(self):
        documents = {
            "base.py": "class Base: pass\n",
            "main.py": "from base import Base\nclass Main(Base): pass\n",
        }

        def confirm_order(prompt):
            order = re.search(r"Computed dependency order: (\[[^\n]+\])", prompt)
            self.assertIsNotNone(order)
            return order.group(1), 0, 0

        pipeline = V5Pipeline(build_python_deps_plugins(), self.task, confirm_order)
        answer, stats = pipeline.run(
            {"files": list(documents), "file_contents": documents}, documents=documents
        )
        self.assertEqual(answer, ["base.py", "main.py"])
        self.assertGreater(stats.n_strong_edges, 0)

    def test_repoqa_candidate_card(self):
        documents = {
            "module.py": (
                'def first():\n    """read a file"""\n    pass\n\n'
                'def second():\n    """send a message"""\n    pass\n'
            )
        }

        def select_candidate(prompt):
            self.assertIn("second", prompt)
            self.assertIn("send a message", prompt)
            return "second", 0, 0

        plugins = PluginSet(
            name="repoqa",
            extractor=RepoQAFunctionExtractor(),
            prompt_template=RepoQASearchPrompt(),
            derivation_rules=[],
        )
        answer, stats = V5Pipeline(plugins, self.task, select_candidate).run(
            {"description": "send a message"}, documents=documents
        )
        self.assertEqual(answer, "second")
        self.assertEqual(stats.n_primitives, 2)

    def test_pixelmem_cache_round_trip(self):
        documents = {"module.py": 'def hello():\n    """greet"""\n    pass\n'}
        with tempfile.TemporaryDirectory() as root:
            cache = PixelMemCache(root)
            extractor = cache.wrap(RepoQAFunctionExtractor())
            first = extractor.extract(documents)
            self.assertFalse(extractor.last_hit)
            second = extractor.extract(documents)
            self.assertTrue(extractor.last_hit)
            self.assertEqual(first, second)
            self.assertEqual((cache.stats.misses, cache.stats.hits), (1, 1))

    def test_retained_dependeval_interface(self):
        documents = {
            "base.py": "class Base: pass\n",
            "main.py": "from base import Base\nclass Main(Base): pass\n",
        }
        answer, _ = run_dependeval(
            list(documents),
            documents,
            lambda _: '["base.py", "main.py"]',
            enable_pairwise=False,
        )
        self.assertEqual(json.loads(answer), ["base.py", "main.py"])


if __name__ == "__main__":
    unittest.main()
