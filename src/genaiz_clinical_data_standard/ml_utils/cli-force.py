
import os
import argparse
import csv

import numpy as np
import scipy

from genaiz_clinical_data_standard import gcds, gcds_utils


def read_tabular(tabular_path: str):
    if tabular_path.endswith(".csv"):
        with open(tabular_path) as table_stream:
            reader = csv.reader(table_stream)
            table: list[list] = []
            for row in reader:
                table.append(row)
    else:
        with open(tabular_path, 'rb') as table_stream:
            table: np.ndarray | scipy.sparse.coo_matrix = scipy.io.mmread(table_stream)
            if isinstance(table, scipy.sparse.coo_matrix):
                table = table.tocsc()
    return table


# must match with naming conventions; see GCDS column naming conventions
OTHER_CHARS = frozenset(['-', '|', '/', '*', '\\', '(', ')', '+', '>', '<', '=', '_', ' '])


def clean_name(name: str, col_index: int) -> str:
    filtered_name = ''.join(filter(lambda s: s.isalnum() or s in OTHER_CHARS, name))
    if not filtered_name:
        filtered_name = "col " + str(col_index)
    return filtered_name


def guess_dtype(values: list) -> str:
    floats = 0
    ints = 0
    strings = 0
    for v in values:
        try:
            _ = int(v)
            ints += 1
        except ValueError:
            try:
                _ = float(v)
                floats += 1
            except ValueError:
                strings += 1
    if ints == len(values):
        return gcds.T_DISCRETE
    elif floats + ints == len(values):
        return gcds.T_CONTINUOUS
    return gcds.T_CATEGORICAL


def construct_metadata(table: list[list] | np.ndarray | scipy.sparse.coo_matrix,
                       header: list[str] = None) -> list[gcds.GCDSMetadata]:
    names: dict[str, int] = dict()
    metadata: list[gcds.GCDSMetadata] = []
    for j in range(gcds_utils.num_cols(table)):
        if header is None:
            name = "col " + str(j)
        else:
            name = clean_name(header[j], j)
        if name in names:
            names[name] += 1
            name = name + "_" + str(names[name])
        else:
            names[name] = 0
        semantics = "s" + str(j)
        if isinstance(table, list):
            dtype = guess_dtype(gcds_utils.unique_column_values(table, j))
        else:
            dtype = gcds.T_DISCRETE if np.issubdtype(table.dtype, np.integer) else gcds.T_CONTINUOUS
        md = gcds.GCDSMetadata(
            index=j, name=name, semantics=semantics, dtype=dtype,
            state="forced", unknown_value=gcds.DEFAULT_UNKNOWN
        )
        metadata.append(md)
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Force non-conformant tabular data to GCDS format')
    parser.add_argument('-n', help='CSV files have NO first row header', action='store_true', default=False)
    parser.add_argument('tabular', help='path to tabular data (extension is used to determine loading)')
    parser.add_argument('out', help='output prefix used to save metadata and table')
    args = parser.parse_args()

    assert os.path.exists(args.tabular), "tabular data does not exist '{}'".format(args.table)
    metadata_path = args.out + ".json"
    assert not os.path.exists(metadata_path), "metadata path already exists '{}'".format(metadata_path)

    table = read_tabular(args.tabular)
    header = None
    if not args.n and isinstance(table, list):
        header = table[0]
        table = table[1:]
    metadata = construct_metadata(table, header=header)
    if isinstance(table, list):
        gcds.coerce_csv_data(metadata, table)
    gcds.write_to_path(metadata, table, args.out)


if __name__ == "__main__":
    main()
