import unittest
import os

import numpy
import numpy as np
import scipy
from genaiz_clinical_data_standard import gcds, gcds_utils
from genaiz_clinical_data_standard.ml_utils import numericize


TEST_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(TEST_PATH, "data")


class TestNumericize(unittest.TestCase):

    def test_drop_columns(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="identifier", dtype=gcds.T_IDENTIFIER, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name="date", dtype=gcds.T_DATE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=3, name="ignore", dtype=gcds.T_IGNORE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=4, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=5, name="keeper", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1)
        ]
        table = [[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 0]]
        metadata, table = numericize.drop_columns(metadata, table)
        self.assertEqual(len(metadata), 2)
        self.assertEqual(metadata[0].name, gcds.COL_PARTICIPANT_ID)
        self.assertEqual(metadata[1].name, "keeper")

    def test_has_duplicate_pairs(self):
        table = [[0, 1], [1, 0], [4, 5], [5, 6], [7, 8], [7, 7]]
        self.assertFalse(numericize._has_duplicate_pairs(table, 0, 1))
        table = [[0, 1], [1, 0], [4, 5], [5, 6], [7, 8], [7, 8]]
        self.assertTrue(numericize._has_duplicate_pairs(table, 0, 1))

    def test_encode_boolean_values(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="bool", dtype=gcds.T_BOOLEAN, state="x", unknown_value="unknown"),
        ]
        table = [[0, "a"], [5, "b"], [7, "unknown"]]
        numericize._encode_boolean_values(metadata, table)
        self.assertNotEqual(table[0][1], table[1][1])
        self.assertEqual(table[2][1], 0)
        self.assertEqual({table[0][1], table[1][1]}, {-1, 1})
        self.assertEqual(metadata[1].mapping, {'unknown': 0, 'a': 1, 'b': -1})
        self.assertEqual(metadata[1].dtype, gcds.T_DISCRETE)
        self.assertEqual(metadata[1].unknown_value, 0)
        metadata[1].dtype = gcds.T_BOOLEAN
        metadata[1].unknown_value = "unknown"
        table = [[0, "a"], [5, "a"], [7, "unknown"]]
        numericize._encode_boolean_values(metadata, table)
        self.assertEqual(table[0][1], table[1][1])
        self.assertEqual(table[2][1], 0)
        self.assertIn(table[0][1], {-1, 1})
        self.assertIn(table[1][1], {-1, 1})
        metadata[1].dtype = gcds.T_BOOLEAN
        metadata[1].unknown_value = -1
        table = np.array([[0, 10], [5, 12], [7, -1]])
        numericize._encode_boolean_values(metadata, table)
        self.assertNotEqual(table[0, 1], table[1, 1])
        self.assertEqual(table[2, 1], 0)
        self.assertEqual({table[0, 1], table[1, 1]}, {-1, 1})
        self.assertEqual(metadata[1].dtype, gcds.T_DISCRETE)
        self.assertEqual(metadata[1].unknown_value, 0)

    def test_encode_target_values(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="target", dtype=gcds.T_TARGET, state="x", unknown_value="unknown"),
        ]
        table = [[0, "a"], [5, "b"], [7, "unknown"]]

        mapping, Y = numericize.encode_target_as_array(metadata[1], table)
        self.assertEqual(Y[0], 0)
        self.assertEqual(Y[1], 1)
        self.assertEqual(Y[2], 2)
        _, Y = numericize.encode_target_as_array(metadata[1], table, mapping=mapping)
        self.assertEqual(Y[0], 0)
        self.assertEqual(Y[1], 1)
        self.assertEqual(Y[2], 2)

        numericize._encode_target_values(metadata, table)
        self.assertEqual(table[0][1], 0)
        self.assertEqual(table[1][1], 1)
        self.assertEqual(table[2][1], 2)
        self.assertEqual(metadata[1].dtype, gcds.T_TARGET)
        self.assertEqual(metadata[1].unknown_value, 2)
        self.assertEqual(metadata[1].mapping, {"a": 0, "b": 1, "unknown": 2})
        metadata[1].unknown_value = -1
        table = np.array([[0, 32], [5, -1], [7, 22]])
        numericize._encode_target_values(metadata, table)
        self.assertEqual(table[0][1], 2)
        self.assertEqual(table[1][1], 0)
        self.assertEqual(table[2][1], 1)
        self.assertEqual(metadata[1].dtype, gcds.T_TARGET)
        self.assertEqual(metadata[1].unknown_value, 0)
        self.assertEqual(metadata[1].mapping, {32: 2, -1: 0, 22: 1})

    def test_encode_target_values2(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="target", dtype=gcds.T_TARGET, state="x", unknown_value="unknown"),
        ]
        table = [[0, "a"], [5, "b"], [7, "b"]]
        numericize._encode_target_values(metadata, table)
        self.assertEqual(table[0][1], 0)
        self.assertEqual(table[1][1], 1)
        self.assertEqual(table[2][1], 1)
        self.assertEqual(metadata[1].dtype, gcds.T_TARGET)
        self.assertEqual(metadata[1].unknown_value, 2)
        self.assertEqual(metadata[1].mapping, {"a": 0, "b": 1, "unknown": 2})

    def test_encode_categorical_values(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="general_cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="set", dtype=gcds.T_SET, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=3, name="range", dtype=gcds.T_RANGE, state="x", unknown_value="unknown"),
        ]
        table = [[0, 45, "a,b", "23 - 45"], [1, 21, "b,c", "unknown"], [2, 70, "unknown", "23 - 45"]]
        new_metadata, new_table = numericize._encode_categorical_values(metadata, table)
        # general cat
        self.assertEqual(new_metadata[1].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[1].unknown_value, -1)
        self.assertEqual(new_metadata[1].fill_value, 0)
        self.assertIn("21", new_metadata[1].name)
        self.assertEqual(new_metadata[2].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[2].fill_value, 0)
        self.assertIn("45", new_metadata[2].name)
        self.assertEqual(new_metadata[3].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[3].fill_value, 1)
        self.assertIn("70", new_metadata[3].name)
        # set
        self.assertEqual(new_metadata[4].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[4].fill_value, 0)
        self.assertIn("a", new_metadata[4].name)
        self.assertEqual(new_metadata[5].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[5].fill_value, 0)
        self.assertIn("b", new_metadata[5].name)
        self.assertEqual(new_metadata[6].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[6].fill_value, 0)
        self.assertIn("c", new_metadata[6].name)
        self.assertEqual(new_metadata[7].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[7].fill_value, 1)
        self.assertIn("unknown", new_metadata[7].name)
        # range
        self.assertEqual(new_metadata[8].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[8].fill_value, 0)
        self.assertIn("23 - 45", new_metadata[8].name)
        self.assertEqual(new_metadata[9].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[9].fill_value, 1)
        self.assertIn("unknown", new_metadata[9].name)
        correct_table = [
            [0, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [2, 0, 0, 1, 0, 0, 0, 1, 1, 0]
        ]
        self.assertEqual(new_table, correct_table)

    def test_encode_categorical_values_suppress(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="general_cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="set", dtype=gcds.T_SET, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=3, name="range", dtype=gcds.T_RANGE, state="x", unknown_value="unknown"),
        ]
        table = [[0, 45, "a,b", "23 - 45"], [1, 21, "b,c", "unknown"], [2, 70, "unknown", "23 - 45"]]
        new_metadata, new_table = numericize._encode_categorical_values(metadata, table, suppress_unknown_columns=True)
        # general cat
        self.assertEqual(new_metadata[1].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[1].unknown_value, -1)
        self.assertEqual(new_metadata[1].fill_value, 0)
        self.assertIn("21", new_metadata[1].name)
        self.assertEqual(new_metadata[2].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[2].fill_value, 0)
        self.assertIn("45", new_metadata[2].name)
        # set
        self.assertEqual(new_metadata[3].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[3].fill_value, 0)
        self.assertIn("a", new_metadata[3].name)
        self.assertEqual(new_metadata[4].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[4].fill_value, 0)
        self.assertIn("b", new_metadata[4].name)
        self.assertEqual(new_metadata[5].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[5].fill_value, 0)
        self.assertIn("c", new_metadata[5].name)
        # range
        self.assertEqual(new_metadata[6].dtype, gcds.T_DISCRETE)
        self.assertEqual(new_metadata[6].fill_value, 0)
        self.assertIn("23 - 45", new_metadata[6].name)
        correct_table = [
            [0, 0, 1, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [2, 0, 0, 0, 0, 0, 1]
        ]
        self.assertEqual(new_table, correct_table)

    def test_encode_categorical_values_noop(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="col1", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="col2", dtype=gcds.T_CONTINUOUS, state="x", unknown_value="unknown"),
        ]
        table = [[0, 45, 12], [1, 21, "unknown"], [2, 70, "unknown"]]
        new_metadata, new_table = numericize._encode_categorical_values(metadata, table)
        self.assertEqual(len(new_metadata), 3)
        correct_table = [[0, 45, 12], [1, 21, "unknown"], [2, 70, "unknown"]]
        self.assertEqual(new_table, correct_table)

    def test_make_one_row_csv(self):
        metadata = [
            gcds.GCDSMetadata(index=1, name="general_cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="set", dtype=gcds.T_SET, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=3, name="range", dtype=gcds.T_RANGE, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=4, name=gcds.COL_EXPERIMENTAL_CONDITION, dtype=gcds.T_CATEGORICAL, state="x",
                              unknown_value="unknown"),
            # test that metadata order does not matter
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
        ]
        table = [
            [0, 45, "a,b", "23 - 45", "exp1"], [0, 21, "b,c", "unknown", "exp2"], [1, 70, "unknown", "23 - 45", "exp1"]
        ]
        new_metadata, new_table = numericize.make_one_row(
            gcds_utils.has_metadata(metadata, gcds.COL_EXPERIMENTAL_CONDITION),
            metadata, table
        )
        truth_table = [
            [0, 45, 'a,b', '23 - 45', 21, 'b,c', 'unknown'],
            [1, 70, 'unknown', '23 - 45', 70, 'unknown', 'unknown']
        ]
        truth_metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="exp1 general_cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="exp1 set", dtype=gcds.T_SET, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=3, name="exp1 range", dtype=gcds.T_RANGE, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=4, name="exp2 general_cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=5, name="exp2 set", dtype=gcds.T_SET, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=6, name="exp2 range", dtype=gcds.T_RANGE, state="x", unknown_value="unknown"),
        ]
        self.assertEqual(new_metadata, truth_metadata)
        self.assertEqual(new_table, truth_table)

    def test_make_one_row_numpy(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="num", dtype=gcds.T_NUMERIC, state="x", unknown_value=0),
            gcds.GCDSMetadata(index=2, name="float", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=3, name=gcds.COL_EXPERIMENTAL_CONDITION, dtype=gcds.T_CATEGORICAL, state="x",
                              unknown_value="unknown"),
        ]
        table = numpy.array([[0, 45, 2.3, 1], [0, 21, 2.5, 2], [1, 70, 6.7, 1]])
        new_metadata, new_table = numericize.make_one_row(
            gcds_utils.has_metadata(metadata, gcds.COL_EXPERIMENTAL_CONDITION),
            metadata, table
        )
        truth_table = np.array([
            [0, 45, 2.3, 21, 2.5],
            [1, 70, 6.7, 0, -1]
        ])
        truth_metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="1.0 num", dtype=gcds.T_NUMERIC, state="x", unknown_value=0),
            gcds.GCDSMetadata(index=2, name="1.0 float", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=3, name="2.0 num", dtype=gcds.T_NUMERIC, state="x", unknown_value=0),
            gcds.GCDSMetadata(index=4, name="2.0 float", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
        ]
        self.assertEqual(new_metadata, truth_metadata)
        self.assertTrue(np.all(new_table == truth_table))

    def test_make_one_row_sparse(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="num", dtype=gcds.T_NUMERIC, state="x", unknown_value=0),
            gcds.GCDSMetadata(index=2, name="float", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=3, name=gcds.COL_EXPERIMENTAL_CONDITION, dtype=gcds.T_CATEGORICAL, state="x",
                              unknown_value="unknown"),
        ]
        table = scipy.sparse.csc_matrix(([1, 1, 2, 1, 3, 1], ([0, 1, 1, 2, 2, 2], [3, 1, 3, 0, 2, 3])), (3, 4))
        new_metadata, new_table = numericize.make_one_row(
            gcds_utils.has_metadata(metadata, gcds.COL_EXPERIMENTAL_CONDITION),
            metadata, table
        )
        truth_table = np.array([
            [0, 0, 0, 1, 0],
            [1, 0, 3, 0, -1]
        ])
        truth_metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="1 num", dtype=gcds.T_NUMERIC, state="x", unknown_value=0),
            gcds.GCDSMetadata(index=2, name="1 float", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=3, name="2 num", dtype=gcds.T_NUMERIC, state="x", unknown_value=0),
            gcds.GCDSMetadata(index=4, name="2 float", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
        ]
        self.assertEqual(new_metadata, truth_metadata)
        self.assertTrue(np.all(new_table.todense() == truth_table))

    def test_encode_numerical_unknowns(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="n", dtype=gcds.T_NUMERIC, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=3, name="d", dtype=gcds.T_DISCRETE, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=4, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=5, name="cu", dtype=gcds.T_CURRENCY, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=6, name="du", dtype=gcds.T_DURATION, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=7, name="duX", dtype=gcds.T_DURATION, state="x", unknown_value="unknown"),
        ]
        table = [
            [0, "c1", 12, 12, 1.2, 9.99, 3, 3],
            [1, "c2", 2.3, 3, 2.3, 8.99, 4.5, 4.5],
            [2, "c2", "unknown", "unknown", -1, "unknown", "unknown", 4]
        ]
        metadata, table = numericize.encode_numerical_unknowns(metadata, table)

        truth_metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="n", dtype=gcds.T_NUMERIC, state="x", fill_value=0),
            gcds.GCDSMetadata(index=3, name="d", dtype=gcds.T_DISCRETE, state="x", fill_value=0),
            gcds.GCDSMetadata(index=4, name="c", dtype=gcds.T_CONTINUOUS, state="x", fill_value=0),
            gcds.GCDSMetadata(index=5, name="cu", dtype=gcds.T_CURRENCY, state="x", fill_value=0),
            gcds.GCDSMetadata(index=6, name="du", dtype=gcds.T_DURATION, state="x", fill_value=0),
            gcds.GCDSMetadata(index=7, name="duX", dtype=gcds.T_DURATION, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=8, name="n unk:unknown", dtype=gcds.T_NUMERIC, state="x", unknown_value=-1,
                              fill_value=1),
            gcds.GCDSMetadata(index=9, name="d unk:unknown", dtype=gcds.T_DISCRETE, state="x", unknown_value=-1,
                              fill_value=1),
            gcds.GCDSMetadata(index=10, name="c unk:-1", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1,
                              fill_value=1),
            gcds.GCDSMetadata(index=11, name="cu unk:unknown", dtype=gcds.T_CURRENCY, state="x", unknown_value=-1,
                              fill_value=1),
            gcds.GCDSMetadata(index=12, name="du unk:unknown", dtype=gcds.T_DURATION, state="x", unknown_value=-1,
                              fill_value=1),
        ]
        self.assertEqual(metadata, truth_metadata)

        truth_table = [
            [0, 'c1', 12, 12, 1.2, 9.99, 3, 3, 0, 0, 0, 0, 0],
            [1, 'c2', 2.3, 3, 2.3, 8.99, 4.5, 4.5, 0, 0, 0, 0, 0],
            [2, 'c2', 0, 0, 0, 0, 0, 4, 1, 1, 1, 1, 1]
        ]
        self.assertEqual(table, truth_table)

    def test_encode_numerical_unknowns_suppress(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="n", dtype=gcds.T_NUMERIC, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=3, name="d", dtype=gcds.T_DISCRETE, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=4, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=5, name="cu", dtype=gcds.T_CURRENCY, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=6, name="du", dtype=gcds.T_DURATION, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=7, name="duX", dtype=gcds.T_DURATION, state="x", unknown_value="unknown"),
        ]
        table = [
            [0, "c1", 12, 12, 1.2, 9.99, 3, 3],
            [1, "c2", 2.3, 3, 2.3, 8.99, 4.5, 4.5],
            [2, "c2", "unknown", "unknown", -1, "unknown", "unknown", 4]
        ]
        metadata, table = numericize.encode_numerical_unknowns(metadata, table, suppress_unknown_columns=True)

        truth_metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="n", dtype=gcds.T_NUMERIC, state="x", fill_value=0),
            gcds.GCDSMetadata(index=3, name="d", dtype=gcds.T_DISCRETE, state="x", fill_value=0),
            gcds.GCDSMetadata(index=4, name="c", dtype=gcds.T_CONTINUOUS, state="x", fill_value=0),
            gcds.GCDSMetadata(index=5, name="cu", dtype=gcds.T_CURRENCY, state="x", fill_value=0),
            gcds.GCDSMetadata(index=6, name="du", dtype=gcds.T_DURATION, state="x", fill_value=0),
            gcds.GCDSMetadata(index=7, name="duX", dtype=gcds.T_DURATION, state="x", unknown_value="unknown"),
        ]
        self.assertEqual(metadata, truth_metadata)

        truth_table = [
            [0, 'c1', 12, 12, 1.2, 9.99, 3, 3],
            [1, 'c2', 2.3, 3, 2.3, 8.99, 4.5, 4.5],
            [2, 'c2', 0, 0, 0, 0, 0, 4]
        ]
        self.assertEqual(table, truth_table)

    def test_encode_identifiers(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70)
        ]
        table = [["0", "c1"], ["1", "c2"], ["2", "c2"]]
        numericize.encode_identifiers(metadata, table)
        self.assertEqual(metadata, metadata)
        truth_table = [[0, "c1"], [1, "c2"], [2, "c2"]]
        self.assertEqual(table, truth_table)
        self.assertEqual(metadata[0].mapping, {"0": 0, "1": 1, "2": 2})

    def test_numericize(self):
        errors = set()
        with open(os.path.join(TEST_DATA_PATH, "megadata.csv")) as in_stream:
            metadata, table = gcds.read_mega_data(gcds.DEMOGRAPHIC, in_stream, errors)
        self.assertEqual(len(errors), 0)
        new_metadata, new_table = numericize.numericize(metadata, table)
        # other unit tests check correctness.
        self.assertEqual(len(new_metadata), 32)
        self.assertEqual(gcds_utils.num_rows(new_table), 4)
        self.assertEqual(gcds_utils.num_cols(new_table), 32)


if __name__ == '__main__':
    unittest.main()
