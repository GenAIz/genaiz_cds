import unittest
import math
import os

import numpy as np
import scipy

from genaiz_clinical_data_standard import gcds


TEST_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(TEST_PATH, "data")


class TestGCDMetadata(unittest.TestCase):

    def test_equal(self):
        x = gcds.GCDSMetadata()
        self.assertEqual(x, x)
        y = gcds.GCDSMetadata()
        self.assertEqual(x, y)
        y.index = 0
        self.assertNotEqual(x, y)
        y.index = x.index
        y.name = "name"
        self.assertNotEqual(x, y)
        y.name = x.name
        y.semantics = "semantics"
        self.assertNotEqual(x, y)
        y.semantics = x.semantics
        y.dtype = "int"
        self.assertNotEqual(x, y)
        y.dtype = x.dtype
        y.state = "state"
        self.assertNotEqual(x, y)
        y.state = x.state
        y.unknown_value = "?"
        self.assertNotEqual(x, y)

    def test_str(self):
        s = ("GCDSMetadata(index=-1, name='None', semantics='None', dtype='None', state='None', unknown_value='None', "
             "fill_value='None', mapping='None')")
        self.assertEqual(str(gcds.GCDSMetadata()), s)

    def test_to_from_dict(self):
        x = gcds.GCDSMetadata(1, "test", "semantics", "string", "source", "?")
        x.mapping = {"a": 0, "b": 1}
        d = x.to_dictionary()
        d2 = dict()
        x.to_dictionary(d2)
        self.assertEqual(d, d2)
        y = gcds.GCDSMetadata()
        y.from_dictionary(d)
        self.assertEqual(x, y)

    def test_normalize_name(self):
        md = gcds.GCDSMetadata(name="col")
        self.assertEqual(md.normalized_name(), "col")
        md = gcds.GCDSMetadata(name="Col  foobar ")
        self.assertEqual(md.normalized_name(), "col_foobar")

    def test_initial_value(self):
        md = gcds.GCDSMetadata(name="col", unknown_value="X")
        self.assertEqual(md.initial_value(), "X")
        md.fill_value = "Y"
        self.assertEqual(md.initial_value(), "Y")


class TestInfer(unittest.TestCase):

    def test_infer_metadata_from_matrix(self):
        m = np.zeros((4, 5), dtype=np.float64)
        metadata = gcds.infer_metadata_from_matrix(m)
        self.assertEqual(len(metadata), 5)
        for i in range(len(metadata)):
            self.assertEqual(metadata[i].index, i)
            self.assertEqual(metadata[i].name, "col_{}".format(i))
            self.assertEqual(metadata[i].dtype, gcds.T_CONTINUOUS)

        m = np.zeros((4, 2), dtype=np.int8)
        metadata = gcds.infer_metadata_from_matrix(m)
        self.assertEqual(len(metadata), 2)
        for i in range(len(metadata)):
            self.assertEqual(metadata[i].index, i)
            self.assertEqual(metadata[i].name, "col_{}".format(i))
            self.assertEqual(metadata[i].dtype, gcds.T_DISCRETE)

        m = np.zeros((3, 10), dtype=np.bool)
        metadata = gcds.infer_metadata_from_matrix(m)
        self.assertEqual(len(metadata), 10)
        for i in range(len(metadata)):
            self.assertEqual(metadata[i].index, i)
            self.assertEqual(metadata[i].name, "col_{}".format(i))
            self.assertEqual(metadata[i].dtype, gcds.T_BOOLEAN)

    def test_infer_metadata_from_csv(self):
        table = [
            ["col1", "col2"],
            ["s1", "s2"],
            ["discrete", "set"],
            ["source", "source"],
            [-1, "!unknown"]
        ]
        metadata = gcds.infer_metadata_from_csv(table)
        self.assertEqual(len(metadata), 2)
        self.assertEqual(metadata[0], gcds.GCDSMetadata(0, "col1", "s1", "discrete", "source", -1))
        self.assertEqual(metadata[1], gcds.GCDSMetadata(1, "col2", "s2", "set", "source", "!unknown"))

    def test_infer_metadata(self):
        table = [
            ["col1", "col2"],
            ["s1", "s2"],
            ["discrete", "set"],
            ["source", "source"],
            [-1, "!unknown"]
        ]
        metadata = gcds.infer_metadata(table)
        self.assertEqual(len(metadata), 2)
        self.assertEqual(metadata[0], gcds.GCDSMetadata(0, "col1", "s1", "discrete", "source", -1))
        self.assertEqual(metadata[1], gcds.GCDSMetadata(1, "col2", "s2", "set", "source", "!unknown"))

        m = np.zeros((3, 10), dtype=np.bool)
        metadata = gcds.infer_metadata(m)
        self.assertEqual(len(metadata), 10)
        for i in range(len(metadata)):
            self.assertEqual(metadata[i].index, i)
            self.assertEqual(metadata[i].name, "col_{}".format(i))
            self.assertEqual(metadata[i].dtype, gcds.T_BOOLEAN)

    def test_infer_missing_data_types(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name="0", dtype="", state="x", unknown_value="y"),
            gcds.GCDSMetadata(index=0, name="race", dtype="", state="x", unknown_value="y")
        ]
        gcds.infer_missing_data_types(gcds.DEMOGRAPHIC, metadata)
        self.assertEqual(metadata[0].dtype, "")
        self.assertEqual(metadata[1].dtype, gcds.T_CATEGORICAL)


class TestInspect(unittest.TestCase):

    def test_identifier_regex(self):
        self.assertIsNotNone(gcds.IDENTIFIER_REGEX.fullmatch("foo.bar:123-baz_bang"))
        self.assertIsNone(gcds.IDENTIFIER_REGEX.fullmatch("%yum"))

    def test_col_name_regex(self):
        self.assertIsNotNone(gcds.COL_NAME_REGEX.fullmatch("123456qwerty _-1234"))
        self.assertIsNone(gcds.COL_NAME_REGEX.fullmatch("!"))
        self.assertIsNotNone(gcds.COL_NAME_REGEX.fullmatch("foo 123 bar-_"))
        self.assertIsNone(gcds.COL_NAME_REGEX.fullmatch("foobar!"))

    def test_numeric_regex(self):
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("inf"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("-inf"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("123"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("-123"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("123."))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("-123."))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("0.123"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("-0.123"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("1.23e10"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("1.23e-10"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("-1.23e10"))
        self.assertIsNotNone(gcds.NUMERIC_REGEX.fullmatch("-1.23e-10"))
        self.assertIsNone(gcds.COL_NAME_REGEX.fullmatch("foobar!"))
        self.assertIsNone(gcds.COL_NAME_REGEX.fullmatch("0.123e10"))
        self.assertIsNone(gcds.COL_NAME_REGEX.fullmatch("12.123e10"))
        self.assertIsNone(gcds.COL_NAME_REGEX.fullmatch(".123"))

    def test_range_regex(self):
        self.assertIsNone(gcds.RANGE_REGEX.fullmatch("inf"))
        self.assertIsNone(gcds.RANGE_REGEX.fullmatch("123"))
        self.assertIsNone(gcds.RANGE_REGEX.fullmatch("bar - ping"))
        mo = gcds.RANGE_REGEX.fullmatch("123 - 123")
        self.assertIsNotNone(mo)
        self.assertEqual(mo.group(1), "123")
        self.assertEqual(mo.group(2), "123")
        mo = gcds.RANGE_REGEX.fullmatch("123 - -123")
        self.assertIsNotNone(mo)
        self.assertEqual(mo.group(1), "123")
        self.assertEqual(mo.group(2), "-123")
        mo = gcds.RANGE_REGEX.fullmatch("-123 - -123")
        self.assertIsNotNone(mo)
        self.assertEqual(mo.group(1), "-123")
        self.assertEqual(mo.group(2), "-123")
        mo = gcds.RANGE_REGEX.fullmatch("-123--123")
        self.assertIsNotNone(mo)
        self.assertEqual(mo.group(1), "-123")
        self.assertEqual(mo.group(2), "-123")

    def test_inspect_metadata(self):
        metadata = gcds.GCDSMetadata(
            index=0, name="test", dtype=gcds.T_NUMERIC, state=gcds.STATE_SOURCE, unknown_value="u"
        )
        self.assertTrue(metadata.is_coherent())
        metadata = gcds.GCDSMetadata(
            index=-1, name="test", dtype=gcds.T_NUMERIC, state=gcds.STATE_SOURCE, unknown_value="u"
        )
        self.assertFalse(metadata.is_coherent())
        metadata = gcds.GCDSMetadata(
            index=0, name="!@ _-%^&", dtype=gcds.T_NUMERIC, state=gcds.STATE_SOURCE, unknown_value="u"
        )
        self.assertFalse(metadata.is_coherent())
        metadata = gcds.GCDSMetadata(
            index=0, name="test", dtype="random", state=gcds.STATE_SOURCE, unknown_value="u"
        )
        self.assertFalse(metadata.is_coherent())
        metadata = gcds.GCDSMetadata(
            index=0, name="test", dtype=gcds.T_NUMERIC, state=None, unknown_value="u"
        )
        self.assertFalse(metadata.is_coherent())
        metadata = gcds.GCDSMetadata(
            index=0, name="test", dtype=gcds.T_NUMERIC, state=gcds.STATE_SOURCE, unknown_value=None
        )
        self.assertFalse(metadata.is_coherent())
        metadata = gcds.GCDSMetadata(
            index=0, name="test", dtype=gcds.T_NUMERIC, state=gcds.STATE_SOURCE, unknown_value="  "
        )
        self.assertFalse(metadata.is_coherent())

    def test_inspect_empty_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_CATEGORICAL
        gcds.inspect_value(md, None, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "", errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, gcds.DEFAULT_UNKNOWN, errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_ignore_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_IGNORE
        gcds.inspect_value(md, None, errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_categorical_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_CATEGORICAL
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "foo, bar", errors)
        self.assertEqual(len(errors), 0)
        errors.clear()
        gcds.inspect_value(md, "foobar", errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_identifier_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_IDENTIFIER
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "foobar", errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_target_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_TARGET
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "foo, bar", errors)
        self.assertEqual(len(errors), 0)
        errors.clear()
        gcds.inspect_value(md, "foobar", errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_boolean_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_TARGET
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "foo, bar", errors)
        self.assertEqual(len(errors), 0)
        errors.clear()
        gcds.inspect_value(md, "foobar", errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_set_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_SET
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "foobar, barbar", errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_numeric_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_NUMERIC
        gcds.inspect_value(md, "boo", errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, 3.5, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, math.nan, errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_discrete_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_DISCRETE
        gcds.inspect_value(md, "boo", errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, 3.2, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, math.nan, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, 35, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, 35.0, errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_continuous_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_NUMERIC
        gcds.inspect_value(md, "boo", errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, 3.5, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, math.nan, errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_currency_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_NUMERIC
        gcds.inspect_value(md, "boo", errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, 3.5, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, math.nan, errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_range_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_RANGE
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "235", errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "23 - 12", errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "3 - 5", errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, "-3 - 5", errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, "-3 - -1", errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, "3-5", errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_duration_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_DURATION
        gcds.inspect_value(md, "boo", errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, -1, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 0)
        gcds.inspect_value(md, 4.5, errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_date_value(self):
        errors = set()
        md = gcds.GCDSMetadata(index=0, name="col", state="source", unknown_value=gcds.DEFAULT_UNKNOWN)
        md.dtype = gcds.T_DATE
        gcds.inspect_value(md, 32, errors)
        self.assertEqual(len(errors), 1)
        errors.clear()
        gcds.inspect_value(md, "2015-10-03", errors)
        self.assertEqual(len(errors), 0)

    def test_inspect_gcds_metadata(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name="col0", dtype=gcds.T_BOOLEAN, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=0, name="race", dtype=gcds.T_DISCRETE, state="x", unknown_value="u")
        ]
        errors = set()
        gcds.inspect_gcds_metadata(gcds.DEMOGRAPHIC, metadata, errors)
        self.assertGreaterEqual(len(errors), 4)
        metadata = [
            gcds.GCDSMetadata(index=0, name="participant_identifier", dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name="clinical_group", dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        errors = set()
        gcds.inspect_gcds_metadata(gcds.DEMOGRAPHIC, metadata, errors)
        self.assertGreaterEqual(len(errors), 0)
        metadata = [
            gcds.GCDSMetadata(index=0, name="participant_identifier", dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="t1", dtype=gcds.T_TARGET, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name="clinical_group", dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        errors = set()
        gcds.inspect_gcds_metadata(gcds.DEMOGRAPHIC, metadata, errors)
        self.assertGreaterEqual(len(errors), 1)

    def inspect_data_column(self):
        md = gcds.GCDSMetadata(
            index=0, name="participant_identifier", dtype=gcds.T_IDENTIFIER, state="x", unknown_value="u"
        )
        column = ["id1", "id2", "id3"]
        self.assertFalse(gcds.inspect_data_column(md, column, set()))
        md = gcds.GCDSMetadata(
            index=4, name="aboolean", dtype=gcds.T_BOOLEAN, state="x", unknown_value="u"
        )
        column = ["v1", "v2", "v3"]
        self.assertFalse(gcds.inspect_data_column(md, column, set()))
        column = ["v1", "v2"]
        self.assertTrue(gcds.inspect_data_column(md, column, set()))
        md = gcds.GCDSMetadata(
            index=1, name="biomarker1", dtype=gcds.T_DISCRETE, state="x", unknown_value=-1
        )
        column = [1, 2, 3]
        self.assertFalse(gcds.inspect_data_column(md, column, set()))
        md = gcds.GCDSMetadata(
            index=2, name="biomarker2", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1
        )
        column = [1.2, 3.4, 5.6]
        self.assertFalse(gcds.inspect_data_column(md, column, set()))

        md = gcds.GCDSMetadata(
            index=1, name="biomarker1", dtype=gcds.T_DISCRETE, state="x", unknown_value=-1
        )
        column = np.array([1, 2, 3])
        self.assertFalse(gcds.inspect_data_column(md, column, set()))
        column = np.array([1.2, 2.3, 3.4])
        self.assertTrue(gcds.inspect_data_column(md, column, set()))

        md = gcds.GCDSMetadata(
            index=2, name="biomarker2", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1
        )
        column = np.array([1, 2, 3])
        self.assertFalse(gcds.inspect_data_column(md, column, set()))
        column = np.array([1.2, 2.3, 3.4])
        self.assertFalse(gcds.inspect_data_column(md, column, set()))

    def inspect_data(self):
        # list
        # sparse
        # dense
        metadata = [
            gcds.GCDSMetadata(index=0, name="participant_identifier", dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="biomarker", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name="biomarker2", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1)
        ]
        table = [
            ["id1", 2, 2.3],
            ["id2", 3, 3],
            ["id3", -1, 7.6]
        ]
        self.assertFalse(gcds.inspect_data(metadata, table, set()))
        table = np.arange(9).reshape((3, 3))
        self.assertFalse(gcds.inspect_data(metadata, table, set()))
        table = np.arange(9).reshape((3, 3)).astype(np.float64)
        table[:, 2] = 1.234
        self.assertFalse(gcds.inspect_data(metadata, table, set()))

        table = scipy.sparse.csc_matrix((3, 4), dtype=np.int8)
        table[0, 0] = 1
        table[1, 0] = 2
        table[2, 0] = 3
        table[3, 0] = 4
        table[2, 2] = 10
        self.assertFalse(gcds.inspect_data(metadata, table, set()))
        table = table.astype(np.float64)
        table[2, 2] = 10.34
        self.assertFalse(gcds.inspect_data(metadata, table, set()))

    def test_inspect_gcds(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name="participant_identifier", dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="biomarker1", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=2, name="biomarker2", dtype=gcds.T_CONTINUOUS, state="x", unknown_value=-1),
            gcds.GCDSMetadata(index=3, name="clinical_group", dtype=gcds.T_TARGET, state="x", unknown_value=-1)
        ]
        table = [
            ["id1", 2, 2.3, "t1"],
            ["id2", 3, 3, "t2"],
            ["id3", -1, 7.6, "t3"]
        ]
        self.assertFalse(gcds.inspect_gcds(gcds.DEMOGRAPHIC, metadata, table, set()))
        metadata = [
            gcds.GCDSMetadata(index=0, name="participant_identifier", dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="participant_identifier", dtype=gcds.T_IDENTIFIER,
                              state="x", unknown_value="u"),
        ]
        self.assertTrue(gcds.inspect_gcds(gcds.DEMOGRAPHIC, metadata, table, set()))


    def test_load_mega_data(self):
        errors = set()
        with open(os.path.join(TEST_DATA_PATH, "megadata.csv")) as in_stream:
            metadata, table = gcds.read_mega_data(gcds.DEMOGRAPHIC, in_stream, errors)
        self.assertEqual(str(errors), str(set()))
        self.assertEqual(len(metadata), 15)
        self.assertEqual(len(table), 4)
        self.assertEqual(len(table[0]), len(metadata))

    def test_load_write(self):
        errors = set()
        with open(os.path.join(TEST_DATA_PATH, "readwrite.csv")) as in_stream:
            metadata, table = gcds.read_mega_data(gcds.FLOW_CYTOMETRY, in_stream, errors)
        # str use below allows errors to be dumped during assertion failure
        self.assertEqual(str(errors), str(set()))
        self.assertEqual(len(metadata), 7)

        rw_md_filename = os.path.join(TEST_DATA_PATH, "rw.json")
        rw_data_csv_filename = os.path.join(TEST_DATA_PATH, "data.csv")
        rw_data_mm_filename = os.path.join(TEST_DATA_PATH, "data.mm")
        if os.path.exists(rw_md_filename):
            os.remove(rw_md_filename)
        if os.path.exists(rw_data_csv_filename):
            os.remove(rw_data_csv_filename)
        if os.path.exists(rw_data_mm_filename):
            os.remove(rw_data_mm_filename)

        with open(rw_md_filename, 'w') as md_out_stream:
            with open(rw_data_csv_filename, 'w') as data_out_stream:
                gcds.write(metadata, table, md_out_stream, data_out_stream)
        with open(rw_md_filename) as md_in_stream:
            with open(rw_data_csv_filename, 'r') as data_in_stream:
                metadata_disk, table_disk = gcds.read(md_in_stream, False, data_in_stream)
        metadata.sort(key=lambda m: m.index)
        metadata_disk.sort(key=lambda m: m.index)
        self.assertEqual(metadata, metadata_disk)
        self.assertTrue(table == table_disk)

        os.remove(rw_md_filename)

        np_table = np.array(table, dtype=np.float64)
        with open(rw_md_filename, 'w') as md_out_stream:
            with open(rw_data_mm_filename, 'wb') as data_out_stream:
                gcds.write(metadata, np_table, md_out_stream, data_out_stream)
        with open(rw_md_filename) as md_in_stream:
            with open(rw_data_mm_filename, 'rb') as data_in_stream:
                metadata_disk, table_disk = gcds.read(md_in_stream, True, data_in_stream)
        metadata.sort(key=lambda m: m.index)
        metadata_disk.sort(key=lambda m: m.index)
        self.assertEqual(metadata, metadata_disk)
        self.assertTrue(np.allclose(np_table, table_disk))

        if os.path.exists(rw_md_filename):
            os.remove(rw_md_filename)
        if os.path.exists(rw_data_csv_filename):
            os.remove(rw_data_csv_filename)
        if os.path.exists(rw_data_mm_filename):
            os.remove(rw_data_mm_filename)

    def test_load_write_sparse(self):
        errors = set()
        with open(os.path.join(TEST_DATA_PATH, "readwrite_sparse.csv")) as in_stream:
            metadata, table = gcds.read_mega_data(gcds.FLOW_CYTOMETRY, in_stream, errors)
        # str use below allows errors to be dumped during assertion failure
        self.assertEqual(str(errors), str(set()))
        self.assertEqual(len(metadata), 7)
        table_sp = scipy.sparse.csc_matrix((len(table), len(metadata)), dtype=np.float64)
        for i in range(len(table)):
            for j in range(len(metadata)):
                if table[i][j] != 0:
                    table_sp[i, j] = table[i][j]

        rw_md_filename = os.path.join(TEST_DATA_PATH, "rw_sp.json")
        rw_data_mm_filename = os.path.join(TEST_DATA_PATH, "data_sp.mm")
        if os.path.exists(rw_md_filename):
            os.remove(rw_md_filename)
        if os.path.exists(rw_data_mm_filename):
            os.remove(rw_data_mm_filename)

        with open(rw_md_filename, 'w') as md_out_stream:
            with open(rw_data_mm_filename, 'wb') as data_out_stream:
                gcds.write(metadata, table_sp, md_out_stream, data_out_stream)
        with open(rw_md_filename) as md_in_stream:
            with open(rw_data_mm_filename, 'rb') as data_in_stream:
                metadata_disk, table_disk = gcds.read(md_in_stream, True, data_in_stream)
        metadata.sort(key=lambda m: m.index)
        metadata_disk.sort(key=lambda m: m.index)
        self.assertEqual(metadata, metadata_disk)
        self.assertTrue(np.allclose(table_sp.todense(), table_disk.todense()))

        if os.path.exists(rw_md_filename):
            os.remove(rw_md_filename)
        if os.path.exists(rw_data_mm_filename):
            os.remove(rw_data_mm_filename)

    def test_read_write_path(self):
        md, table = gcds.read_from_path(
            os.path.join(TEST_DATA_PATH, "readwrite_path.json"),
            os.path.join(TEST_DATA_PATH, "readwrite_path.csv")
        )
        test_write_prefix = os.path.join(TEST_DATA_PATH, "test_readwrite_path")
        gcds.write_to_path(md, table, test_write_prefix)
        try:
            gcds.write_to_path(md, table, test_write_prefix)
            self.fail("IO error on write was not generated.")
        except IOError:
            pass
        gcds.write_to_path(md, table, test_write_prefix, cannot_exist=False)
        md_path = test_write_prefix + ".json"
        table_path = test_write_prefix + ".csv"
        mdX, tableX = gcds.read_from_path(md_path, table_path)
        self.assertEqual(md, mdX)
        self.assertEqual(table, tableX)
        if os.path.exists(md_path):
            os.remove(md_path)
        if os.path.exists(table_path):
            os.remove(table_path)


class TestOther(unittest.TestCase):

    def test_coerce_csv_data(self):
        metadata = [
            gcds.GCDSMetadata(index=0, name="col0", dtype=gcds.T_BOOLEAN, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="col1", dtype=gcds.T_DISCRETE, state="x", unknown_value="u")
        ]
        table = [
            [1, "u"],
            [0, "32"]
        ]
        gcds.coerce_csv_data(metadata, table)
        self.assertEqual(table, [["1", "u"], ["0", 32]])
        metadata = [
            gcds.GCDSMetadata(index=0, name="col0", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="col1", dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        table = [
            [1, 0],
            [0, 1]
        ]
        gcds.coerce_csv_data(metadata, table)
        self.assertEqual(table, [[1, "0"], [0, "1"]])
        metadata = [
            gcds.GCDSMetadata(index=0, name="col0", dtype=gcds.T_DISCRETE, state="x", unknown_value="u"),
            gcds.GCDSMetadata(index=1, name="col1", dtype=gcds.T_TARGET, state="x", unknown_value="u")
        ]
        metadata[1].mapping = dict()
        table = [
            [1, 0],
            [0, 1]
        ]
        gcds.coerce_csv_data(metadata, table)
        self.assertEqual(table, [[1, 0], [0, 1]])


if __name__ == '__main__':
    unittest.main()
