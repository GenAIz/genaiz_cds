"""
Code for merging two tables.  A merge is similar to an SQL join.  In other
words, two tables are merged using the identifier column (left most) of each table.
"""

from typing import Any
from collections import OrderedDict
import copy

import numpy as np
import scipy

from genaiz_clinical_data_standard import gcds
from genaiz_clinical_data_standard.gcds_utils import num_rows, num_cols, sv_, gv_, table_like


def merge(metadata1: list[gcds.GCDSMetadata], table1: list[list] | np.ndarray | scipy.sparse.spmatrix,
          metadata2: list[gcds.GCDSMetadata], table2: list[list] | np.ndarray | scipy.sparse.spmatrix,
          intersection: bool = False) \
        -> tuple[list[gcds.GCDSMetadata], list[list] | np.ndarray | scipy.sparse.spmatrix]:
    """
    Merges two tables and their metadata.  If intersection is true then only the identifiers (and data) shared
    between the two tables appear in the merge, otherwise all identifiers appear in the merge.
    If the merge is a union (not intersection) of the tables then the missing data is filled
    using the initial_value method of the metadata. The original metadata and tables remain unchanged.

    :param metadata1: metadata
    :param table1: table
    :param metadata2: metadata
    :param table2: table
    :param intersection: if true then only the identifiers shared between the two tables appear in the merge
    :return: a merged table
    """
    metadata1.sort(key=lambda md_: md_.index)
    metadata2.sort(key=lambda md_: md_.index)
    if metadata1[0].dtype != gcds.T_IDENTIFIER:
        raise ValueError("Column 0 of metadata 1 is not of type identifier.")
    if metadata2[0].dtype != gcds.T_IDENTIFIER:
        raise ValueError("Column 0 of metadata 2 is not of type identifier.")

    old_row_to_id1 = _identifier_mapping(metadata1[0], table1)
    old_row_to_id2 = _identifier_mapping(metadata2[0], table2)
    id_to_row = _create_identifier_to_row(old_row_to_id1, old_row_to_id2, intersection)

    new_metadata, old_col_to_new_col1, old_col_to_new_col2 = _merge_metadata(metadata1, metadata2)

    new_table = _new_table(len(id_to_row), table1, table2)
    _copy_table_data(id_to_row, old_row_to_id1, old_col_to_new_col1, new_metadata, table1, new_table)
    _copy_table_data(id_to_row, old_row_to_id2, old_col_to_new_col2, new_metadata, table2, new_table)

    if isinstance(new_table, list):
        new_metadata[0].mapping = None
        for identifier, i in id_to_row.items():
            sv_(new_table, i, 0, identifier)
    else:
        new_metadata[0].mapping = id_to_row
        for i in range(num_rows(new_table)):
            sv_(new_table, i, 0, i)

    return new_metadata, new_table


def _identifier_mapping(metadata: gcds.GCDSMetadata, table: list[list] | np.ndarray | scipy.sparse.spmatrix) \
        -> OrderedDict[int, Any]:
    """
    Builds a row to original identifier mapping.  An original identifier is the key of the
    metadata mapping dictionary.  If there is no metadata mapping then the original identifier
    is the value in the identifier column.
    :param metadata: metadata
    :param table: table
    :return: a row ordered dictionary mapping the row index to an original identifier
    """
    identifiers: OrderedDict[int, Any] = OrderedDict()
    for i in range(num_rows(table)):
        identifiers[i] = gv_(table, i, 0)
    if metadata.mapping is not None:
        reverse = dict()
        for orig_identifier in metadata.mapping:
            identifier = metadata.mapping[orig_identifier]
            if identifier in reverse:
                raise ValueError("Cannot have duplicate identifiers in reverse mapping")
            reverse[identifier] = orig_identifier
        for i in identifiers:
            identifiers[i] = reverse[identifiers[i]]
    return identifiers


def _create_identifier_to_row(old_row_to_id1: OrderedDict[int, Any], old_row_to_id2: OrderedDict[int, Any],
                              intersection: bool) -> OrderedDict[Any, int]:
    """
    Constructs a mapping of identifier to row indexes for a merged (new) table.
    Identifier values are collected from old_row_to_id1 and old_row_to_id2.

    :param old_row_to_id1: mapping of old row indexes to identifiers
    :param old_row_to_id2: mapping of old row indexes to identifiers
    :param intersection: if true then only ids existing in both old_row_to_idX appear in the returned mapping
    """
    ids1 = set(old_row_to_id1.values())
    ids2 = set(old_row_to_id2.values())
    identifiers = ids1.intersection(ids2) if intersection else ids1.union(ids2)
    identifiers = list(identifiers)
    identifiers.sort()
    id_to_row: OrderedDict[int, Any] = OrderedDict()
    for i in range(len(identifiers)):
        identifier = identifiers[i]
        id_to_row[identifier] = i
    return id_to_row


def _merge_metadata(metadata1: list[gcds.GCDSMetadata], metadata2: list[gcds.GCDSMetadata]) \
        -> tuple[list[gcds.GCDSMetadata], dict[int, int], dict[int, int]]:
    """
    Merges metadata by making a copy of the metadata.  The metadata must be sorted by index.
    Column names are renamed so that column names remain unique.

    :param metadata1: metadata
    :param metadata2: metadata
    :return: index ordered merged metadata and old column index to new column index mappings
    """
    id_md = copy.copy(metadata1[0])
    if metadata1[0].name != metadata2[0].name:
        id_md.name = metadata1[0].name + " " + metadata2[0].name
    id_md.unknown_value = None
    id_md.fill_value = None
    id_md.state = "merge"
    id_md.mapping = None
    new_metadata = [id_md]
    names = set()
    old_col_to_new_col1: dict[int, int] = dict()
    for i in range(1, len(metadata1)):
        md = copy.deepcopy(metadata1[i])
        md.index = len(new_metadata)
        old_col_to_new_col1[metadata1[i].index] = md.index
        new_metadata.append(md)
        names.add(md.name)
    old_col_to_new_col2: dict[int, int] = dict()
    for i in range(1, len(metadata2)):
        md = copy.deepcopy(metadata2[i])
        if md.name in names:
            md.name += " 2"
        md.index = len(new_metadata)
        old_col_to_new_col2[metadata2[i].index] = md.index
        new_metadata.append(md)
    return new_metadata, old_col_to_new_col1, old_col_to_new_col2


def _new_table(number_rows: int, table1: list[list] | np.ndarray | scipy.sparse.spmatrix,
               table2: list[list] | np.ndarray | scipy.sparse.spmatrix) \
        -> list[list] | np.ndarray | scipy.sparse.spmatrix:
    """
    Create a new table that can correctly accept merged data.  For example, a table made of lists
    cannot be merged to a sparse matrix.  This prevents strings being merged which would cause errors.


    Currently, list and list, dense and dense and dense and sparse may be merged. A future version
    may relax constraints and check metadata types.

    :param number_rows: the number of rows expected in the new table
    :param table1: first table
    :param table2: second table
    :return: a new table of type where merge may occur
    """
    if (isinstance(table1, list) and isinstance(table2, list)) \
            or (isinstance(table1, np.ndarray) and isinstance(table2, np.ndarray)):
        table_like_this = table1
    elif isinstance(table1, np.ndarray) and scipy.sparse.issparse(table2):
        table_like_this = table2
    elif scipy.sparse.issparse(table1) and isinstance(table2, np.ndarray):
        table_like_this = table1
    else:
        raise ValueError("Cannot not merge these two types of tables: {} {}".format(type(table1), type(table2)))
    num_columns = num_cols(table1) + num_cols(table2) - 1   # subtract duplicate identifier
    return table_like(table_like_this, rows=number_rows, cols=num_columns)


def _copy_table_data(id_to_row: OrderedDict[Any, int], old_row_to_id: OrderedDict[int, Any],
                     old_col_to_new_col: dict[int, int], metadata: list[gcds.GCDSMetadata],
                     old_table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                     new_table: list[list] | np.ndarray | scipy.sparse.spmatrix) -> None:
    """
    Copies data from the old table to the new table.  Missing data, in the new table, is
    filled with values pulled from the column's metadata initial_value method.

    :param id_to_row: mapping from identifier to row
    :param old_row_to_id: mapping from old row to identifier
    :param old_col_to_new_col: mapping from old column index to new column index
    :param metadata: metadata of the new table
    :param old_table: old data to copy
    :param new_table: new table where the copy will be placed
    """
    def _copy_value(identifier_, i_: int, j_: int):
        row_index_ = id_to_row[identifier_]
        col_index_ = old_col_to_new_col[j_]
        value_ = gv_(old_table, i_, j_)
        sv_(new_table, row_index_, col_index_, value_)

    # copy values over then fill missing
    if scipy.sparse.issparse(old_table):
        row_indexes, col_indexes = old_table.nonzero()
        for x in range(row_indexes.shape[0]):
            j = int(col_indexes[x])
            if j < 1:
                continue
            i = int(row_indexes[x])
            identifier = old_row_to_id[i]
            if identifier not in id_to_row:
                # drop this row
                continue
            _copy_value(identifier, i, j)
    else:
        for i in range(num_rows(old_table)):
            identifier = old_row_to_id[i]
            if identifier not in id_to_row:
                # drop this row
                continue
            for j in range(1, num_cols(old_table)):
                _copy_value(identifier, i, j)
    missing_identifiers = set(id_to_row.keys()) - set(old_row_to_id.values())
    for missing_id in missing_identifiers:
        for j in range(1, num_cols(old_table)):
            row_index = id_to_row[missing_id]
            col_index = old_col_to_new_col[j]
            if metadata[col_index].index != col_index:
                raise ValueError("Expected sorted metadata")
            new_value = metadata[col_index].initial_value()
            sv_(new_table, row_index, col_index, new_value)
