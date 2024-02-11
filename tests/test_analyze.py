import unittest
import maave
from test_models import *


class AnalyzeTests(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_functional_tensorflow(self):
        pass

    def test_class_tensorflow(self):
        maave.analyze(MyModel, name="MyModel v1")

    def test_functional_torch(self):
        pass

    def test_class_torch(self):
        maave.analyze(LeNet, name="LeNet v1")
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
