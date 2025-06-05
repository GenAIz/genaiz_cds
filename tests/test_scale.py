import unittest
import os

import numpy as np

from genaiz_clinical_data_standard import gcds, gcds_utils
from genaiz_clinical_data_standard.ml_utils import numericize, scale


TEST_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(TEST_PATH, "data")


class TestScale(unittest.TestCase):

    def test_scale(self):
        errors = set()
        with open(os.path.join(TEST_DATA_PATH, "megadata.csv")) as in_stream:
            metadata, table = gcds.read_mega_data(gcds.DEMOGRAPHIC, in_stream, errors)
        self.assertEqual(len(errors), 0)
        new_metadata, new_table = numericize.numericize(metadata, table)
        ss, new_table = scale.scale(np.array(new_table))
        self.assertEqual(len(new_metadata), 32)
        self.assertEqual(gcds_utils.num_rows(new_table), 4)
        self.assertEqual(gcds_utils.num_cols(new_table), 32)


if __name__ == '__main__':
    unittest.main()
