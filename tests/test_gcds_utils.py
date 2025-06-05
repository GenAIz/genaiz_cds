import unittest

import numpy as np
import scipy

from genaiz_clinical_data_standard import gcds, gcds_utils


class TestGCDSUtils(unittest.TestCase):

    def test_table_like_list(self):
        table = []
        new_table = gcds_utils.table_like(table)
        self.assertEqual(new_table, [])
        table = [[1, 2, 3],[3, 4, 5]]
        new_table = gcds_utils.table_like(table)
        self.assertEqual(new_table, [[None, None, None], [None, None, None]])
        new_table = gcds_utils.table_like(table, rows=4, cols=2)
        self.assertEqual(new_table, [[None, None], [None, None], [None, None], [None, None]])
        new_table = gcds_utils.table_like(table, rows=1, cols=2, default_list_value=0)
        self.assertEqual(new_table, [[0, 0]])

    def test_table_like_ndarray(self):
        table = np.zeros((2, 6), dtype=np.int8)
        new_table = gcds_utils.table_like(table)
        self.assertTrue(isinstance(new_table, np.ndarray))
        self.assertEqual(new_table.shape, table.shape)
        self.assertEqual(new_table.dtype, table.dtype)
        new_table = gcds_utils.table_like(table, rows=4, cols=2, dtype=np.float64)
        self.assertEqual(new_table.shape, (4, 2))
        self.assertEqual(new_table.dtype, np.float64)

    def test_table_like_spmatrix(self):
        table = scipy.sparse.csc_matrix((2, 6), dtype=np.int8)
        new_table = gcds_utils.table_like(table)
        self.assertTrue(scipy.sparse.issparse(new_table))
        self.assertEqual(new_table.shape, table.shape)
        self.assertEqual(new_table.dtype, table.dtype)
        new_table = gcds_utils.table_like(table, rows=4, cols=2, dtype=np.float64)
        self.assertEqual(new_table.shape, (4, 2))
        self.assertEqual(new_table.dtype, np.float64)

    def test_num_rows(self):
        table = []
        self.assertEqual(gcds_utils.num_rows(table), 0)
        table = [[1, 2]]
        self.assertEqual(gcds_utils.num_rows(table), 1)
        table = np.array(table)
        self.assertEqual(gcds_utils.num_rows(table), 1)

    def test_num_cols(self):
        table = []
        self.assertEqual(gcds_utils.num_cols(table), 0)
        table = [[]]
        self.assertEqual(gcds_utils.num_cols(table), 0)
        table = [[1, 2]]
        self.assertEqual(gcds_utils.num_cols(table), 2)
        table = np.array(table)
        self.assertEqual(gcds_utils.num_cols(table), 2)

    def test_shape(self):
        table = [[1, 2]]
        self.assertEqual(gcds_utils.shape(table), (1, 2))
        table = np.array(table)
        self.assertEqual(gcds_utils.shape(table), (1, 2))

    def test_gv_sv_(self):
        table = [[1, 2], [3, 4]]
        self.assertEqual(gcds_utils.gv_(table, 0, 0), 1)
        self.assertEqual(gcds_utils.gv_(table, 1, 1), 4)
        gcds_utils.sv_(table, 0, 0, 5)
        self.assertEqual(gcds_utils.gv_(table, 0, 0), 5)
        table = np.array([[1, 2], [3, 4]])
        self.assertEqual(gcds_utils.gv_(table, 0, 0), 1)
        self.assertEqual(gcds_utils.gv_(table, 1, 1), 4)
        gcds_utils.sv_(table, 0, 0, 5)
        self.assertEqual(gcds_utils.gv_(table, 0, 0), 5)
        table = [[1, "foo, bar"]]
        value = gcds_utils.gv_(table, 0, 1)
        self.assertEqual(value, "foo, bar")
        value = gcds_utils.gv_(table, 0, 1, separate_set_items=True)
        self.assertEqual(value, ["foo", "bar"])

    def test_filter_metadata(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        kept = metadata.copy()
        gcds_utils.filter_metadata(kept, [])
        self.assertEqual(len(kept), 0)

        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        kept = metadata.copy()
        gcds_utils.filter_metadata(kept, [0, 1, 2])
        self.assertEqual(len(kept), 3)

        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        kept = metadata.copy()
        gcds_utils.filter_metadata(kept, [0, 2])
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].name, metadata[0].name)
        self.assertEqual(kept[1].name, metadata[2].name)
        self.assertEqual(kept[1].index, 1)

        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        kept = gcds_utils.filter_metadata(metadata, [1], make_copy=True)
        self.assertNotEqual(kept, metadata)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].name, metadata[1].name)
        self.assertEqual(kept[0].index, 0)

    def test_drop_metadata_columns(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        kept = metadata.copy()
        gcds_utils.drop_metadata_columns(metadata, [])
        self.assertEqual(len(kept), 3)

        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        kept = metadata.copy()
        gcds_utils.drop_metadata_columns(kept, [1])
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].name, metadata[0].name)
        self.assertEqual(kept[1].name, metadata[2].name)
        self.assertEqual(kept[1].index, 1)

        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        kept = gcds_utils.drop_metadata_columns(metadata, [0, 2], make_copy=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].name, metadata[1].name)
        self.assertEqual(kept[0].index, 0)

    def test_unique_table_column(self):
        table = [[0, 0, 1], [0, 4, 0], [0, 2, 0], [0, 4, 0]]
        col = gcds_utils.unique_column_values(table, 1)
        self.assertEqual(col, [0, 4, 2])
        table_np = np.array(table)
        col = gcds_utils.unique_column_values(table_np, 1)
        self.assertEqual(col, [0, 4, 2])
        table_sp = scipy.sparse.csc_matrix(([1, 4, 2, 4], ([0, 1, 2, 3], [2, 1, 1, 1])), (4, 3))
        col = gcds_utils.unique_column_values(table_sp, 1)
        self.assertEqual(col, [0, 4, 2])
        table = [[0, "foo,bar"], [0, "bar,banana"], [0, "foo,bar"], [0, "bingo,foo"]]
        col = gcds_utils.unique_column_values(table, 1, separate_set_items=True)
        self.assertEqual(col, ["foo", "bar", "banana", "bingo"])

    def test_copy_table_column(self):
        table = [[0, 0, 1], [0, 4, 0], [0, 2, 0], [0, 4, 0]]
        col = gcds_utils.copy_table_column(table, 1)
        self.assertEqual(col, [0, 4, 2, 4])
        table_np = np.array(table)
        col = gcds_utils.copy_table_column(table_np, 1)
        self.assertTrue(np.all(col == np.array([0, 4, 2, 4])))
        table_sp = scipy.sparse.csc_matrix(([1, 4, 2, 4], ([0, 1, 2, 3], [2, 1, 1, 1])), (4, 3))
        col = gcds_utils.copy_table_column(table_sp, 1)
        self.assertTrue(np.all(col.todense().flatten() == np.array([0, 4, 2, 4])))

    def test_filter_table_columns(self):
        table = [[0, 0, 1], [0, 4, 0], [0, 2, 0], [0, 4, 0]]
        ftable = gcds_utils.filter_table_columns(table, [1, 2])
        self.assertEqual(ftable, [[0, 1], [4, 0], [2, 0], [4, 0]])
        table_np = np.array(table)
        ftable = gcds_utils.filter_table_columns(table_np, [1, 2])
        self.assertTrue(np.all(ftable == np.array([[0, 1], [4, 0], [2, 0], [4, 0]])))
        table_sp = scipy.sparse.csc_matrix(([1, 4, 2, 4], ([0, 1, 2, 3], [2, 1, 1, 1])), (4, 3))
        ftable = gcds_utils.filter_table_columns(table_sp, [1, 2])
        self.assertTrue(np.all(ftable.todense() == np.array([[0, 1], [4, 0], [2, 0], [4, 0]])))

    def test_drop_table_columns(self):
        table = [[0, 0, 1], [0, 4, 0], [0, 2, 0], [0, 4, 0]]
        ftable = gcds_utils.drop_table_columns(table, [1])
        self.assertEqual(ftable, [[0, 1], [0, 0], [0, 0], [0, 0]])
        table_np = np.array(table)
        ftable = gcds_utils.drop_table_columns(table_np, [1])
        self.assertTrue(np.all(ftable == np.array([[0, 1], [0, 0], [0, 0], [0, 0]])))
        table_sp = scipy.sparse.csc_matrix(([1, 4, 2, 4], ([0, 1, 2, 3], [2, 1, 1, 1])), (4, 3))
        ftable = gcds_utils.drop_table_columns(table_sp, [1])
        self.assertTrue(np.all(ftable.todense() == np.array([[0, 1], [0, 0], [0, 0], [0, 0]])))

    def test_concat_table_columns(self):
        table1 = [[0, 0, 1], [0, 4, 0]]
        table2 = [[0, 2, 0], [0, 4, 0]]
        new_table = gcds_utils.concat_table_columns(table1, table2)
        self.assertEqual(new_table, [[0, 0, 1, 0, 2, 0], [0, 4, 0, 0, 4, 0]])
        table1_np = np.array(table1)
        table2_np = np.array(table2)
        new_table = gcds_utils.concat_table_columns(table1_np, table2_np)
        self.assertTrue(np.all(new_table == np.array([[0, 0, 1, 0, 2, 0], [0, 4, 0, 0, 4, 0]])))

    def test_has_metadata(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker col", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        self.assertIs(gcds_utils.has_metadata(metadata, gcds.COL_CLINICAL_GROUP), metadata[2])
        self.assertIs(gcds_utils.has_metadata(metadata, "biomarker col"), metadata[1])
        self.assertIs(gcds_utils.has_metadata(metadata, "biomarker_col"), metadata[1])
        self.assertIs(gcds_utils.has_metadata(metadata, "missing"), None)

    def test_has_metadata_type(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        self.assertIs(gcds_utils.has_metadata_type(metadata, gcds.T_TARGET), metadata[2])
        self.assertIs(gcds_utils.has_metadata_type(metadata, gcds.T_DISCRETE), metadata[1])
        self.assertIs(gcds_utils.has_metadata_type(metadata, "missing"), None)

    def test_concat_metadata(self):
        metadata1 = [
            gcds.GCDSMetadata(index=0, name=gcds.COL_PARTICIPANT_ID, dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value=gcds.DEFAULT_UNKNOWN),
            gcds.GCDSMetadata(index=1, name="biomarker col 1", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name=gcds.COL_CLINICAL_GROUP, dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        metadata2 = [
            gcds.GCDSMetadata(index=0, name="biomarker col 2", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="biomarker col 3", dtype=gcds.T_CONTINUOUS, state="x", unknown_value="u"),
        ]
        mx = gcds_utils.concat_metadata(metadata1, metadata2)
        self.assertEqual(len(mx), 5)
        self.assertEqual(mx[-1].name, "biomarker col 3")
        self.assertEqual(mx[-1].index, 4)
        self.assertEqual(mx[-2].name, "biomarker col 2")
        self.assertEqual(mx[-2].index, 3)
        self.assertEqual(mx[0].name, gcds.COL_PARTICIPANT_ID)
        self.assertEqual(mx[0].index, 0)

    def test_to_list_table(self):
        table = [[0, 0, 1], [0, 4, 0], [0, 2, 0], [0, 4, 0]]
        table_np = np.array(table)
        self.assertEqual(table, gcds_utils.to_list_table(table_np))




if __name__ == '__main__':
    unittest.main()
