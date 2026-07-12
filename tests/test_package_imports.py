import importlib.metadata
import unittest


class PackageImportTests(unittest.TestCase):
    def test_public_package_imports(self):
        import heuriboost_rag
        import heuriboost_rag.backends
        import heuriboost_rag.reckless

        self.assertEqual(heuriboost_rag.__version__, "0.2.0")
        self.assertEqual(
            importlib.metadata.version("heuriboost-rag"),
            heuriboost_rag.__version__,
        )
