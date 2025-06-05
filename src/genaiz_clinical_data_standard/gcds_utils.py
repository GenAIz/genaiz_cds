"""
This file contains generic table operations that abstract away the table type.
For example, num_rows is a function that can fetch the number of rows
of a table regardless if the table's type is list, dense/numpy or sparse/scipy.
"""

import copy

import numpy as np
import scipy

from genaiz_clinical_data_standard import gcds


def table_like(table: list[list] | np.ndarray | scipy.sparse.spmatrix, rows: int = -1, cols: int = -1,
               dtype=None, default_list_value=None) -> list[list] | np.ndarray | scipy.sparse.spmatrix:
    """
    Create a new table like the given one.  The table shape and dtype may be overridden.
    List tables are filled with None.
    :param table: the table mimic
    :param rows: number of rows if greater than 0
    :param cols: number of cols if greater than 0
    :param dtype: type to use if not None (only applies if the table is already a dense or sparse matrix)
    :param default_list_value: the default value to use when creating a new table from lists
    :return: a new table filled with None (list) or 0s (dense or sparse matrix)
    """
    if rows < 1:
        rows = len(table) if isinstance(table, list) else table.shape[0]
    if cols < 1:
        if isinstance(table, list):
            cols = len(table[0]) if table else 0
        else:
            cols = table.shape[1]
    if isinstance(table, list):
        new_table = []
        for i in range(rows):
            new_table.append([default_list_value] * cols)
    elif scipy.sparse.issparse(table):
        dtype = table.dtype if dtype is None else dtype
        new_table = scipy.sparse.csc_matrix((rows, cols), dtype=dtype)
    else:
        dtype = table.dtype if dtype is None else dtype
        new_table = np.zeros((rows, cols), dtype=dtype)
    return new_table


def num_rows(table: list[list] | np.ndarray | scipy.sparse.spmatrix) -> int:
    """
    Returns the number of rows of the table
    :param table: table
    :return: number of rows
    """
    return len(table) if isinstance(table, list) else table.shape[0]


def num_cols(table: list[list] | np.ndarray | scipy.sparse.spmatrix) -> int:
    """
    Returns the number of columns of the table
    :param table: table
    :return: number of columns
    """
    if not isinstance(table, list):
        return table.shape[1]
    return len(table[0]) if table else 0


def shape(table: list[list] | np.ndarray | scipy.sparse.spmatrix) -> tuple[int, int]:
    """
    Returns the table's shape
    :param table: table
    :return: the table's shape
    """
    return num_rows(table), num_cols(table)


def gv_(table: list[list] | np.ndarray | scipy.sparse.spmatrix, i: int, j: int, separate_set_items: bool = False):
    """
    Gets a value from the table
    :param table: table
    :param i: row index
    :param j: col index
    :param separate_set_items: if true, assumes the column is of type SET and separates items
    :return: value or a list of values if separate_set_items is true
    """
    if isinstance(table, list):
        value = table[i][j]
        if separate_set_items:
            if not isinstance(value, str):
                raise ValueError("Expected a column of type SET containing str but got type {}".format(type(value)))
            value = value.split(gcds.SET_SEPARATOR)
            for i in range(len(value)):
                value[i] = value[i].strip()
        return value
    else:
        return table[i, j]


def sv_(table: list[list] | np.ndarray | scipy.sparse.spmatrix, i: int, j: int, value) -> None:
    """
    Sets a table's value.
    :param table: table
    :param i: row index
    :param j: col index
    :param value: new value
    """
    if isinstance(table, list):
        table[i][j] = value
    elif isinstance(table, np.ndarray):
        table[i, j] = value
    elif scipy.sparse.issparse(table):
        if value != 0:
            table[i, j] = value
    else:
        raise ValueError("Bad table type {}".format(type(table)))


def has_metadata(metadata: list[gcds.GCDSMetadata], name: str) -> gcds.GCDSMetadata | None:
    """
    Linear search for metadata column matching the given name.  First found is returned.
    :param metadata: a list of metadata
    :param name: name of the column of interest
    :return: the column's metadata or None if it's not found
    """
    for md in metadata:
        if md.name == name or md.normalized_name() == name:
            return md
    return None


def has_metadata_type(metadata: list[gcds.GCDSMetadata], dtype: str) -> gcds.GCDSMetadata | None:
    """
    Linear search for metadata column matching the given name.  First found is returned.
    :param metadata: a list of metadata
    :param dtype: type of the column of interest
    :return: the column's metadata or None if it's not found
    """
    for md in metadata:
        if md.dtype == dtype:
            return md
    return None


def drop_metadata_columns(metadata: list[gcds.GCDSMetadata], drop: list[int] | set[int],
                          make_copy: bool = False) -> list[gcds.GCDSMetadata]:
    """
    Drop columns from metadata.  metadata is modified.
    :param metadata: metadata
    :param drop: list of column indexes to drop, possibly empty
    :param make_copy: if true a deep copy is made and returned
    :return: the metadata with the specified columns dropped
    """
    if isinstance(drop, list):
        drop = set(drop)
    keep: list[int] = []
    for md in metadata:
        if md.index in drop:
            continue
        keep.append(md.index)
    return filter_metadata(metadata, keep, make_copy=make_copy)


def filter_metadata(metadata: list[gcds.GCDSMetadata], keep: list[int] | set[int],
                    make_copy: bool = False) -> list[gcds.GCDSMetadata]:
    """
    Filter metadata columns, keeping only those specified.  metadata is modified.
    :param metadata: metadata
    :param keep: list of column indexes to keep, possibly empty
    :param make_copy: if true a deep copy is made and returned
    :return: the filtered metadata
    """
    # due to mem concerns input is modified
    if isinstance(keep, list):
        keep = set(keep)
    temp_metadata: list[gcds.GCDSMetadata] = []
    for md in metadata:
        if md.index in keep:
            temp_metadata.append(copy.deepcopy(md) if make_copy else md)
        if len(temp_metadata) == 1:
            temp_metadata[-1].index = 0
        elif len(temp_metadata) > 1:
            temp_metadata[-1].index = temp_metadata[-2].index + 1
    if make_copy:
        return temp_metadata
    else:
        metadata.clear()
        metadata.extend(temp_metadata)
        return metadata


def concat_metadata(metadata1: list[gcds.GCDSMetadata], metadata2: list[gcds.GCDSMetadata],
                    make_copy: bool = False) -> list[gcds.GCDSMetadata]:
    """
    Concats metadata changing the index appropriately
    :param metadata1: a list of metadata
    :param metadata2: a list of metadata
    :param make_copy: true if a deep copy should be made of the metadata
    :return: a new list of metadata
    """
    new_metadata: list[gcds.GCDSMetadata] = []
    new_metadata.extend(metadata1)
    new_metadata.extend(metadata2)
    for i in range(len(new_metadata)):
        if make_copy:
            new_metadata[i] = copy.deepcopy(new_metadata[i])
        new_metadata[i].index = i
    return new_metadata


def unique_column_values(table: list[list] | np.ndarray | scipy.sparse.spmatrix, index: int,
                         separate_set_items: bool = False) -> list:
    """
    Construct a list of unique column values for a given column
    :param table: table of data
    :param index: column index to unique
    :param separate_set_items: if true, assumes the column is of type SET and separates items
    :return: a list of unique column values in the column's natural order
    """
    if index >= num_cols(table):
        raise ValueError("Index {} out of bounds".format(index))
    unique_values = set()
    unique_order = []
    for i in range(num_rows(table)):
        value = gv_(table, i, index)
        if separate_set_items:
            if not isinstance(value, str):
                raise ValueError("Expected a column of type SET containing str but got type {}".format(type(value)))
            for item in value.split(gcds.SET_SEPARATOR):
                item = item.strip()
                if item not in unique_values:
                    unique_values.add(item)
                    unique_order.append(item)
        elif value not in unique_values:
            unique_values.add(value)
            unique_order.append(value)
    return unique_order


def copy_table_column(table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                      index: int) -> list | np.ndarray | scipy.sparse.spmatrix:
    """
    Copy a column of data from a table.  The copy will be of the same type os the table.
    Note that sparse matrices return a matrix with shape (n, 1).
    :param table: the table to copy from
    :param index: column index to copy
    :return: a copy of the column
    """
    if index >= num_cols(table):
        raise ValueError("Index {} out of bounds".format(index))
    if isinstance(table, list):
        col = []
        for i in range(len(table)):
            col.append(table[i][index])
    else:
        col = table[:, index]
    return col


def filter_table_columns(table: list[list] | np.ndarray | scipy.sparse.spmatrix, keep: list[int] | set[int]) \
        -> list[list] | np.ndarray | scipy.sparse.spmatrix:
    """
    Creates a filtered copy of the table.
    :param table: the table to copy
    :param keep: the rows to keep
    :return: a filtered copy of the table
    """
    # due to mem concerns input is modified
    if len(keep) > num_cols(table):
        raise ValueError("Cannot have 'keep' contain more values than columns.")
    if isinstance(keep, set):
        keep = list(keep)
        keep.sort()
    if isinstance(table, np.ndarray) or scipy.sparse.issparse(table):
        new_table = table[:, keep]
    else:
        new_table = []
        for i in range(len(table)):
            new_row = []
            for index in keep:
                new_row.append(table[i][index])
            new_table.append(new_row)
    return new_table


def drop_table_columns(table: list[list] | np.ndarray | scipy.sparse.spmatrix, drop: list[int] | set[int]) \
        -> list[list] | np.ndarray | scipy.sparse.spmatrix:
    """
    Creates a copy of the table without the specified columns.
    :param table: the table
    :param drop: columns to drop
    :return: a copy of the table without the specified columns
    """
    if isinstance(drop, list):
        drop = set(drop)
    keep: list[int] = []
    length = len(table[0]) if isinstance(table, list) else table.shape[1]
    for i in range(length):
        if i in drop:
            continue
        keep.append(i)
    return filter_table_columns(table, keep)


def concat_table_columns(table1: list[list] | np.ndarray | scipy.sparse.spmatrix,
                         table2: list[list] | np.ndarray | scipy.sparse.spmatrix) \
        -> list[list] | np.ndarray | scipy.sparse.spmatrix:
    """
    Concatenates or stacks columns of two tables.  The tables are expected to have the same number
    of rows.  A new table is returned.
    :param table1: a table
    :param table2: another table
    :return: a new table containing the columns of the two
    """
    if isinstance(table1, list) and isinstance(table2, list):
        if num_rows(table1) != num_rows(table2):
            raise ValueError("Tables cannot have a differing number of rows")
        new_table = []
        for i in range(num_rows(table1)):
            new_table.append([])
            new_table[-1].extend(table1[i])
            new_table[-1].extend(table2[i])
        return new_table
    elif isinstance(table1, np.ndarray) and isinstance(table2, np.ndarray):
        return np.hstack((table1, table2))
    elif scipy.sparse.issparse(table1) and scipy.sparse.issparse(table2):
        return scipy.sparse.hstack((table1, table2))
    else:
        raise ValueError("Table types mismatch '{}' and '{}'".format(type(table1), type(table2)))


_TO_LIST_EXAMPLE_TABLE = [["example"], ["example"]]


def to_list_table(table: np.ndarray | scipy.sparse.spmatrix) -> list[list]:
    """
    Convert a non-list table to a table constructed from lists.
    :param table: non-list table
    :return: a list table with values from the original
    """
    if isinstance(table, list):
        raise ValueError("Can't convert list table to itself")
    rows = num_rows(table)
    cols = num_cols(table)
    list_table = table_like(_TO_LIST_EXAMPLE_TABLE, rows=rows, cols=cols)
    for i in range(rows):
        for j in range(cols):
            value = gv_(table, i, j)
            if isinstance(value, np.integer):
                list_table[i][j] = int(value)
            elif isinstance(value, np.floating):
                list_table[i][j] = float(value)
            else:
                list_table[i][j] = str(value)
    return list_table
