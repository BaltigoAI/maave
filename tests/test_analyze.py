import unittest
import maave
from test_models import *


class AnalyzeTests(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_functional_tensorflow(self):
        pass

    def test_class_tensorflow(self):
        pass

    def test_functional_torch(self):
        pass

    def test_class_torch(self):
        maave.analyze(LeNet)
        pass


if __name__ == '__main__':
    unittest.main()
