"""
Code that can encode a GCDS (non-numerical table) to a purely numerical table.
For example, this code can encode categorical values to numerical equivalents.
"""

import copy
from typing import Any
import collections

import numpy as np
import scipy

from genaiz_clinical_data_standard import gcds, gcds_utils
from genaiz_clinical_data_standard.gcds_utils import num_rows, num_cols, sv_, gv_, has_metadata, table_like


def numericize(metadata: list[gcds.GCDSMetadata], table: list[list] | np.ndarray | scipy.sparse.spmatrix,
               standard: str = None, make_one_row_using: str | int = None, suppress_unknown_columns: bool = False) \
        -> tuple[list[gcds.GCDSMetadata], np.ndarray | scipy.sparse.spmatrix]:
    """
    Convert a table into one that contains only numerical values and where there is one row
    per identifier.  Missing values are filed with the column's defined missing value. The original
    metadata and table may be modified with the goal of conserving memory (rather than creating
    new metadata and table).

    Columns of type ignore, date and target as well as all non-0 indexed identifiers are dropped.
    Data spread across multiples rows is placed in one row.  Categorical, numerical unknowns and
    the remaining identifier values are encoded.

    Post-encoding, the metadata and table follow the GCDS common standard
    but not specific standards such as the demographic standard.

    :param metadata: column metadata
    :param table: the table to numericize
    :param standard: the standard to which this table adheres (e.g., flow cytometry)
    :param make_one_row_using: if the standard is not specified then convert to one row on this column (name or index)
    :param suppress_unknown_columns: if true, do not produce unknown columns (convert values only)
    :return: a table of numeric values
    """
    # note: original inputs are modified
    metadata, table = drop_columns(metadata, table)
    # convert to 1 row first since it may lead to less sparsity and a smaller memory footprint
    if standard == gcds.SEROLOGY:
        md = has_metadata(metadata, gcds.COL_COLLECTION_NAME)
        if md is None:
            raise ValueError("Could not find column {}".format(gcds.COL_COLLECTION_NAME))
        metadata, table = make_one_row(md, metadata, table)
    elif standard == gcds.FLOW_CYTOMETRY:
        md = has_metadata(metadata, gcds.COL_EXPERIMENTAL_CONDITION)
        if md is None:
            raise ValueError("Could not find column {}".format(gcds.COL_EXPERIMENTAL_CONDITION))
        metadata, table = make_one_row(md, metadata, table)
    elif standard is None and make_one_row_using is not None:
        if isinstance(make_one_row_using, str):
            md = has_metadata(metadata, make_one_row_using)
        elif isinstance(make_one_row_using, int):
            metadata.sort(key=lambda md_: md_.index)
            md = metadata[make_one_row_using]
            if md.index != make_one_row_using:
                raise ValueError("Metadata indexes are no longer synchronized index:{}".format(make_one_row_using))
        else:
            raise ValueError("make_one_row_using must be an int or str but is {}".format(type(make_one_row_using)))
        if md is None:
            raise ValueError("Could not find column {}".format(gcds.COL_EXPERIMENTAL_CONDITION))
        metadata, table = make_one_row(md, metadata, table)
    metadata, table = encode_numerical_unknowns(metadata, table, suppress_unknown_columns=suppress_unknown_columns)
    if isinstance(table, list):
        metadata, table = encode_categorical_values(metadata, table, suppress_unknown_columns=suppress_unknown_columns)
        encode_identifiers(metadata, table)
    return metadata, table


def drop_columns(metadata: list[gcds.GCDSMetadata], table: list[list] | np.ndarray | scipy.sparse.spmatrix) \
        -> tuple[list[gcds.GCDSMetadata], list[list] | np.ndarray | scipy.sparse.spmatrix]:
    """
    Drops all columns of type ignore, date and target.  All supplemental identifiers are dropped as well.
    Only the 0-indexed identifier is kept.  Metadata is modified.
    :param metadata: metadata
    :param table: the table
    :return: a table without ignore, date and non-0 indexed identifiers
    """
    # due to mem concerns input is modified
    keep: list[int] = []
    for md in metadata:
        if (md.index != 0 and md.dtype == gcds.T_IDENTIFIER) or md.dtype == gcds.T_IGNORE \
                or md.dtype == gcds.T_DATE or md.dtype == gcds.T_TARGET:
            # note: the identifier used to join data across tables is always the first column (standard)
            continue
        keep.append(md.index)
    gcds_utils.filter_metadata(metadata, keep)
    table = gcds_utils.filter_table_columns(table, keep)
    return metadata, table


def make_one_row(groups_md: gcds.GCDSMetadata, metadata: list[gcds.GCDSMetadata],
                 table: list[list] | np.ndarray | scipy.sparse.spmatrix) \
        -> tuple[list[gcds.GCDSMetadata], list[list] | np.ndarray | scipy.sparse.spmatrix]:
    """
    Places all data associated with the 0-indexed identifier in a single row.  There is an assumption
    that data split across rows are grouped by some categorical column.  This naming or grouping
    is used to flatten the data and rename the columns.
    :param groups_md: metadata of the column that contains the groups/conditions/names
    :param metadata: metadata
    :param table: table
    :return: a flattened table where all data related to a 0-indexed identifier is in a single row
    """
    if isinstance(table, list) and not table:
        raise ValueError("Empty table")
    if scipy.sparse.issparse(table) and not scipy.sparse.isspmatrix_csc(table):
        raise ValueError("Sparse table should be in CSC format")
    if num_cols(table) < 3:
        raise ValueError("Table must have at least 3 columns")
    unique_groups = gcds_utils.unique_column_values(table, groups_md.index)
    if len(unique_groups) < 2:
        gcds_utils.drop_metadata_columns(metadata, [groups_md.index])
        table = gcds_utils.drop_table_columns(table, [groups_md.index])
        return metadata, table
    # standard is that the joining/id column is left most
    metadata.sort(key=lambda md_: md_.index)
    identifier_md: gcds.GCDSMetadata = metadata[0]
    if _has_duplicate_pairs(table, identifier_md.index, groups_md.index):
        raise ValueError("Cannot have two or more identical pairs in '{}' and '{}'".format(
            identifier_md.name, groups_md.name
        ))

    # create new metadata and mapping of new_column -> group and original column
    column_mapping: dict[int, tuple[Any, gcds.GCDSMetadata]] = dict()
    new_metadata: list[gcds.GCDSMetadata] = [copy.deepcopy(identifier_md)]
    for group in unique_groups:
        for md in metadata:
            if md.index == identifier_md.index or md.index == groups_md.index:
                continue
            new_md = copy.deepcopy(md)
            new_md.name = str(group) + " " + new_md.name
            new_md.index = len(new_metadata)
            new_metadata.append(new_md)
            column_mapping[new_md.index] = (group, md)
    # create mapping of id and group -> original row
    row_mapping: dict[tuple, int] = dict()
    for i in range(num_rows(table)):
        key = (gv_(table, i, identifier_md.index), gv_(table, i, groups_md.index))
        row_mapping[key] = i
    # fill id column
    unique_ids = gcds_utils.unique_column_values(table, identifier_md.index)
    # fill new table by pulling values in
    new_table = gcds_utils.table_like(table, rows=len(unique_ids), cols=len(new_metadata))
    for i in range(len(unique_ids)):
        sv_(new_table, i, 0, unique_ids[i])
    # fill remainder
    for j in range(1, num_cols(new_table)):
        group, original_col_md = column_mapping[j]
        for i in range(num_rows(new_table)):
            row_unique_id = gv_(new_table, i, 0)
            key = (row_unique_id, group)
            if key in row_mapping:
                original_row_index = row_mapping[key]
                original_value = gv_(table, original_row_index, original_col_md.index)
            else:
                original_value = original_col_md.initial_value()
            sv_(new_table, i, j, original_value)
    return new_metadata, new_table


def _has_duplicate_pairs(table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                         col1: int, col2: int) -> bool:
    """
    Checks for duplicate pairs returning true if one is found.
    :param table: table
    :param col1: first column
    :param col2: second column
    :return: true if a matching pair is found
    """
    pairs = set()
    for i in range(num_rows(table)):
        pair = (gv_(table, i, col1), gv_(table, i, col2))
        if pair in pairs:
            return True
        pairs.add(pair)
    return False


NUMERICAL_UNKNOWN_TYPES = frozenset([
    gcds.T_NUMERIC, gcds.T_DISCRETE, gcds.T_CONTINUOUS, gcds.T_CURRENCY, gcds.T_DURATION
])


def encode_numerical_unknowns(metadata: list[gcds.GCDSMetadata],
                              table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                              suppress_unknown_columns: bool = False) \
        -> tuple[list[gcds.GCDSMetadata], list[list] | np.ndarray | scipy.sparse.spmatrix]:
    """
    Encode numerical unknowns (types: numeric, discrete, continuous, currency, duration) in
    a separate column.  The unknown is replaced with zero.  The new column contains a 1 at the
    same row.  Unknown values are therefore removed without loss of information.
    :param metadata: metadata
    :param table: table
    :param suppress_unknown_columns: if true, do not produce unknown columns (convert values only)
    :return: new metadata and table with additional columns and encoded numerical unknown values
    """
    row_indexes_of_unknown = set()
    new_column_metadata = []
    for md in metadata:
        if md.dtype not in NUMERICAL_UNKNOWN_TYPES:
            continue
        row_indexes_of_unknown.clear()
        for i in range(num_rows(table)):
            if gv_(table, i, md.index) == md.unknown_value:
                row_indexes_of_unknown.add(i)
        if not row_indexes_of_unknown:
            continue
        for i in row_indexes_of_unknown:
            sv_(table, i, md.index, 0)
        if suppress_unknown_columns:
            md.unknown_value = None
            md.fill_value = 0
        else:
            column = table_like(table, cols=1, default_list_value=0)
            for i in row_indexes_of_unknown:
                sv_(column, i, 0, 1)
            table = gcds_utils.concat_table_columns(table, column)
            new_md = copy.deepcopy(md)
            new_md.name = new_md.name + " unk:" + str(md.unknown_value)
            new_md.index = len(metadata) + len(new_column_metadata)
            md.unknown_value = None
            md.fill_value = 0
            new_md.unknown_value = -1
            new_md.fill_value = 1
            new_column_metadata.append(new_md)
    metadata.extend(new_column_metadata)
    return metadata, table


def _encode_boolean_values(metadata: list[gcds.GCDSMetadata],
                           table: list[list] | np.ndarray | scipy.sparse.spmatrix) -> None:
    """
    Encodes boolean values to 1 and -1, where 0 represents the unknown.  The given table is modified
    to reduce the memory footprint of this function.
    :param metadata: metadata
    :param table:  table
    """
    for md in metadata:
        if md.dtype != gcds.T_BOOLEAN:
            continue
        categories = gcds_utils.unique_column_values(table, md.index)
        if len(categories) > 3:
            raise ValueError("Unexpected number of boolean values, including unknown (more than 3)")
        if md.unknown_value in categories:
            categories.remove(md.unknown_value)
        if len(categories) > 2:
            raise ValueError("Unexpected number of boolean values (more than 2)")
        mapping = {md.unknown_value: 0}
        categories.sort()
        for i, cat in enumerate(categories):
            if i == 0:
                mapping[cat] = 1
            elif i == 1:
                mapping[cat] = -1
        md.dtype = gcds.T_DISCRETE
        md.unknown_value = 0
        md.mapping = mapping
        for i in range(num_rows(table)):
            old_value = gv_(table, i, md.index)
            new_value = mapping[old_value]
            sv_(table, i, md.index, new_value)


def _encode_target_values(metadata: list[gcds.GCDSMetadata],
                          table: list[list] | np.ndarray | scipy.sparse.spmatrix) -> None:
    """
    Encodes the target values as integers, including unknown values.
    The metadata and table are modified.
    :param metadata: metadata
    :param table: table
    """
    mapping: dict[Any, int] = dict()
    for md in metadata:
        if md.dtype != gcds.T_TARGET:
            continue
        categories = gcds_utils.unique_column_values(table, md.index)
        categories.sort()
        old_unknown_value = md.unknown_value
        for i, cat in enumerate(categories):
            if cat == md.unknown_value:
                md.unknown_value = i
            mapping[cat] = i
        if old_unknown_value not in mapping:
            mapping[md.unknown_value] = len(mapping)
            md.unknown_value = mapping[md.unknown_value]
        for i in range(num_rows(table)):
            old_value = gv_(table, i, md.index)
            new_value = mapping[old_value]
            sv_(table, i, md.index, new_value)
        md.mapping = mapping


def encode_target_as_array(target_metadata: gcds.GCDSMetadata, table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                           mapping: dict[str, int] = None) -> tuple[dict[str, int], np.ndarray]:
    """
    Encodes a target column by mapping the columns contents to integers.  The metadata is not changed.
    :param target_metadata: the column's metadata
    :param table: the table
    :param mapping: the mapping of column value to integer, if known (otherwise it is created)
    :return: the mapping of column value to integer and the encoded array
    """
    if target_metadata.dtype != gcds.T_TARGET:
        raise ValueError("Cannot encode a non-target column")
    if mapping is None:
        categories = gcds_utils.unique_column_values(table, target_metadata.index)
        categories.sort()
        mapping = dict()
        for i, cat in enumerate(categories):
            mapping[cat] = i
    Y = np.zeros(gcds_utils.num_rows(table), dtype=np.uint32)
    for i in range(gcds_utils.num_rows(table)):
        Y[i] = mapping[gcds_utils.gv_(table, i, target_metadata.index)]
    return mapping, Y


CATEGORICAL_TYPES = frozenset([gcds.T_CATEGORICAL, gcds.T_SET, gcds.T_RANGE])


def _encode_categorical_values(metadata: list[gcds.GCDSMetadata],
                               table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                               suppress_unknown_columns: bool = False) \
        -> tuple[list[gcds.GCDSMetadata], list[list] | np.ndarray | scipy.sparse.spmatrix]:
    """
    Encodes categorical values into multiple 1-hot columns.  Metadata and table are modified
    during encoding.
    :param metadata: metadata
    :param table: table
    :param suppress_unknown_columns: if true, do not produce unknown columns (convert values only)
    :return: a table with categorical values encoded as 1-hot and corresponding metadata
    """
    column_mapping: collections.OrderedDict[tuple[int, Any], gcds.GCDSMetadata] = collections.OrderedDict()
    drop: set[int] = set()
    for md in metadata:
        if md.dtype not in CATEGORICAL_TYPES:
            continue
        drop.add(md.index)
        categories = gcds_utils.unique_column_values(table, md.index, separate_set_items=(md.dtype == gcds.T_SET))
        categories.sort()
        for cat in categories:
            if suppress_unknown_columns and cat == md.unknown_value:
                continue
            new_md = copy.deepcopy(md)
            new_md.index = len(column_mapping)
            new_md.name = new_md.name + " " + str(cat)
            new_md.dtype = gcds.T_DISCRETE
            new_md.unknown_value = -1
            new_md.fill_value = 1 if cat == md.unknown_value else 0
            column_mapping[(md.index, cat)] = new_md
    if column_mapping:
        # create additional table and fill it
        additions = gcds_utils.table_like(table, cols=len(column_mapping), default_list_value=0)
        for md in metadata:
            if md.dtype not in CATEGORICAL_TYPES:
                continue
            for i in range(num_rows(table)):
                categories = gv_(table, i, md.index, separate_set_items=(md.dtype == gcds.T_SET))
                if not isinstance(categories, list):
                    categories = [categories]
                for cat in categories:
                    if suppress_unknown_columns and cat == md.unknown_value:
                        continue
                    new_col_index = column_mapping[(md.index, cat)].index
                    sv_(additions, i, new_col_index, 1)
        gcds_utils.drop_metadata_columns(metadata, drop)
        table = gcds_utils.drop_table_columns(table, drop)
        temp = list(column_mapping.values())
        temp.sort(key=lambda md_: md_.index)
        for md in temp:
            md.index = len(metadata)
            metadata.append(md)
        table = gcds_utils.concat_table_columns(table, additions)
    return metadata, table


def encode_categorical_values(metadata: list[gcds.GCDSMetadata],
                              table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                              suppress_unknown_columns: bool = False) \
        -> tuple[list[gcds.GCDSMetadata], list[list] | np.ndarray | scipy.sparse.spmatrix]:
    """
    Encodes all categorical values.  Booleans are encoded as 1, 0 and -1.  Targets are encoded to integers,
    0 and above.  All other categorical values are 1-hot encoded.
    :param metadata: metadata
    :param table: table
    :param suppress_unknown_columns: if true, do not produce unknown columns (convert values only)
    :return: a table with encoded values, matching metadata
    """
    _encode_boolean_values(metadata, table)
    _encode_target_values(metadata, table)
    metadata, table = _encode_categorical_values(metadata, table, suppress_unknown_columns=suppress_unknown_columns)
    return metadata, table


def encode_identifiers(metadata: list[gcds.GCDSMetadata],
                       table: list[list] | np.ndarray | scipy.sparse.spmatrix) -> None:
    """
    Encode identifiers as integers, 0 and above.  The metadata and table are modified.
    :param metadata: metadata
    :param table: table
    """
    if not isinstance(table, list):
        return
    id_md = gcds_utils.has_metadata_type(metadata, gcds.T_IDENTIFIER)
    if id_md is None:
        return
    if id_md.index != 0:
        raise ValueError("Missing identifier column at index 0 in metadata")
    mapping = dict()
    for i in range(num_rows(table)):
        old_id = gv_(table, i, 0)
        mapping[old_id] = len(mapping)
        sv_(table, i, 0, mapping[old_id])
    id_md.mapping = mapping
