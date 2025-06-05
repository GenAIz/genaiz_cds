import unittest
from collections import OrderedDict

import numpy as np
import scipy

from genaiz_clinical_data_standard import gcds, gcds_utils
from genaiz_clinical_data_standard.ml_utils import merge


class TestMerge(unittest.TestCase):

    def test_identifier_mapping(self):
        md = gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER, state="x",
                               unknown_value=gcds.DEFAULT_UNKNOWN)
        table = [[10, "c1"], [11, "c2"], [12, "c2"]]
        mapping = merge._identifier_mapping(md, table)
        self.assertEqual(mapping, OrderedDict([(0, 10), (1, 11), (2, 12)]))
        md.mapping = {"0": 10, "1": 11, "2": 12}
        mapping = merge._identifier_mapping(md, table)
        self.assertEqual(mapping, OrderedDict([(0, "0"), (1, "1"), (2, "2")]))

    def test_create_identifier_to_row(self):
        r1 = OrderedDict([(0, "0"), (1, "1"), (2, "2")])
        r2 = OrderedDict([(0, "0"), (2, "2"), (3, "3")])
        id_to_row = merge._create_identifier_to_row(r1, r2, True)
        self.assertEqual(id_to_row, OrderedDict([("0", 0), ("2", 1)]))
        id_to_row = merge._create_identifier_to_row(r1, r2, False)
        self.assertEqual(id_to_row, OrderedDict([("0", 0), ("1", 1), ("2", 2), ("3", 3)]))

    def test_merge_metadata(self):
        m1 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=3, name="cu", dtype=gcds.T_CURRENCY, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=4, name="du", dtype=gcds.T_DURATION, state="x", unknown_value="unknown"),
        ]
        m2 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat two", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="c two", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=3, name="cu", dtype=gcds.T_CURRENCY, state="x", unknown_value="unknown"),
            gcds.GCDSMetadata(index=4, name="du", dtype=gcds.T_DURATION, state="x", unknown_value="unknown"),
        ]
        md, map1, map2 = merge._merge_metadata(m1, m2)
        self.assertEqual(len(md), 9)
        names = {
            0: gcds.COL_PARTICIPANT_ID, 1: "cat", 2: "c", 3: "cu", 4: "du",
            5: "cat two", 6: "c two", 7: "cu 2", 8: "du 2"
        }
        for i in range(len(md)):
            self.assertEqual(md[i].name, names[i])
            self.assertEqual(md[i].index, i)
        self.assertEqual(map1, {1: 1, 2: 2, 3: 3, 4: 4})
        self.assertEqual(map2, {1: 5, 2: 6, 3: 7, 4: 8})

    def test_new_table(self):
        t1 = [[10, 2, 5.2], [11, 2, 7.2], [12, 2, 11.2]]
        t2 = [[6, 2, 5.4], [11, 2, 7.4], [12, 2, 11.4]]
        nt = merge._new_table(3, t1, t2)
        self.assertIsInstance(nt, list)
        self.assertEqual(gcds_utils.num_cols(nt), 5)
        t1 = np.array(t1)
        t2 = np.array(t2)
        nt = merge._new_table(3, t1, t2)
        self.assertIsInstance(nt, np.ndarray)
        self.assertEqual(gcds_utils.num_cols(nt), 5)
        s = scipy.sparse.csc_matrix((3, 5))
        nt = merge._new_table(3, t1, s)
        self.assertTrue(scipy.sparse.issparse(nt))
        self.assertEqual(gcds_utils.num_cols(nt), 7)
        nt = merge._new_table(3, s, t2)
        self.assertTrue(scipy.sparse.issparse(nt))
        self.assertEqual(gcds_utils.num_cols(nt), 7)

    def test_copy_table_data(self):
        m1 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
        ]
        t1 = [[10, "c1", 5.2], [11, "c2", 7.2], [12, "c2", 11.2]]
        m2 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=44),
            gcds.GCDSMetadata(index=2, name="c two", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-13),
        ]
        t2 = [[6, "c3", 5.4], [11, "c1", 7.4], [12, "c1", 11.4]]
        id_map1 = merge._identifier_mapping(m1[0], t1)
        id_map2 = merge._identifier_mapping(m2[0], t2)
        id_to_row = merge._create_identifier_to_row(id_map1, id_map2, False)
        md, col_map1, col_map2 = merge._merge_metadata(m1, m2)
        new_table = merge._new_table(len(id_to_row), t1, t2)
        merge._copy_table_data(id_to_row, id_map1, col_map1, md, t1, new_table)
        truth_table = [
            [None, 70, -1, None, None], [None, 'c1', 5.2, None, None],
            [None, 'c2', 7.2, None, None], [None, 'c2', 11.2, None, None]
        ]
        self.assertEqual(new_table, truth_table)
        merge._copy_table_data(id_to_row, id_map2, col_map2, md, t2, new_table)
        truth_table = [
            [None, 70, -1, 'c3', 5.4], [None, 'c1', 5.2, 44, -13],
            [None, 'c2', 7.2, 'c1', 7.4], [None, 'c2', 11.2, 'c1', 11.4]
         ]
        self.assertEqual(new_table, truth_table)

    def test_copy_table_data_dense(self):
        m1 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
        ]
        t1 = np.array([[10, 2, 5.2], [11, 2, 7.2], [12, 2, 11.2]])
        m2 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=44),
            gcds.GCDSMetadata(index=2, name="c two", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-13),
        ]
        t2 = np.array([[6, 3, 5.4], [11, 3, 7.4], [12, 3, 11.4]])
        id_map1 = merge._identifier_mapping(m1[0], t1)
        id_map2 = merge._identifier_mapping(m2[0], t2)
        id_to_row = merge._create_identifier_to_row(id_map1, id_map2, False)
        md, col_map1, col_map2 = merge._merge_metadata(m1, m2)
        new_table = merge._new_table(len(id_to_row), t1, t2)
        merge._copy_table_data(id_to_row, id_map1, col_map1, md, t1, new_table)
        merge._copy_table_data(id_to_row, id_map2, col_map2, md, t2, new_table)
        truth_table = [
            [0., 70., -1., 3., 5.4],
            [0., 2., 5.2, 44., -13.],
            [0., 2., 7.2, 3., 7.4],
            [0., 2., 11.2, 3., 11.4]
        ]
        self.assertTrue(np.all(truth_table == new_table))

    def test_copy_table_data_sparse(self):
        m1 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
        ]
        t1 = np.array([[10, 2, 5.2], [11, 2, 7.2], [12, 2, 11.2]])
        m2 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=44),
            gcds.GCDSMetadata(index=2, name="c two", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-13),
        ]
        t2 = scipy.sparse.csc_matrix(([6, 11, 12, 3, 7.4], ([0, 1, 2, 0, 1], [0, 0, 0, 1, 2])), (3, 3))
        id_map1 = merge._identifier_mapping(m1[0], t1)
        id_map2 = merge._identifier_mapping(m2[0], t2)
        id_to_row = merge._create_identifier_to_row(id_map1, id_map2, False)
        md, col_map1, col_map2 = merge._merge_metadata(m1, m2)
        new_table = merge._new_table(len(id_to_row), t1, t2)
        merge._copy_table_data(id_to_row, id_map1, col_map1, md, t1, new_table)
        merge._copy_table_data(id_to_row, id_map2, col_map2, md, t2, new_table)
        new_table = new_table.todense()
        truth_table = [
            [0., 70., -1., 3., 0],
            [0., 2., 5.2, 44., -13.],
            [0., 2., 7.2, 0, 7.4],
            [0., 2., 11.2, 0, 0]
        ]
        self.assertTrue(np.all(truth_table == new_table))

    def test_merge(self):
        m1 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
        ]
        t1 = [[10, "c1", 5.2], [11, "c2", 7.2], [12, "c2", 11.2]]
        m2 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=44),
            gcds.GCDSMetadata(index=2, name="c two", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-13),
        ]
        t2 = [[6, "c3", 5.4], [11, "c1", 7.4], [12, "c1", 11.4]]
        new_md, new_t = merge.merge(m1, t1, m2, t2)
        mt = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER, state="merge"),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=3, name="cat 2", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=44),
            gcds.GCDSMetadata(index=4, name="c two", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-13)
        ]
        self.assertEqual(new_md, mt)
        tt = [
            [6, 70, -1, 'c3', 5.4], [10, 'c1', 5.2, 44, -13],
            [11, 'c2', 7.2, 'c1', 7.4], [12, 'c2', 11.2, 'c1', 11.4]
        ]
        self.assertEqual(new_t, tt)

    def test_merge_dense(self):
        m1 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=70),
            gcds.GCDSMetadata(index=2, name="c", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
        ]
        t1 = np.array([[10, 7, 5.2], [11, 7, 7.2], [12, 7, 11.2]])
        m2 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="cat", dtype=gcds.T_CATEGORICAL, state="x", unknown_value=44),
            gcds.GCDSMetadata(index=2, name="c two", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-13),
        ]
        t2 = np.array([[6, 9, 5.4], [11, 9, 7.4], [12, 9, 11.4]])
        new_md, new_t = merge.merge(m1, t1, m2, t2)
        tt = np.array([
            [0, 70, -1, 9., 5.4], [1, 7., 5.2, 44, -13],
            [2, 7., 7.2, 9., 7.4], [3, 7., 11.2, 9., 11.4]
        ])
        self.assertTrue(np.all(new_t == tt))
        self.assertEqual(new_md[0].mapping, OrderedDict([(6, 0), (10, 1), (11, 2), (12, 3)]))


if __name__ == '__main__':
    unittest.main()
