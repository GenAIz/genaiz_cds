import os
import argparse

import numpy as np
import scipy

from genaiz_clinical_data_standard import gcds
from genaiz_clinical_data_standard.ml_utils import merge


def read(path_metadata: str, path_table: str, force_to_dense: bool) \
        -> tuple[list[gcds.GCDSMetadata], list[list] | np.ndarray | scipy.sparse.coo_matrix]:
    """
    Convenience method for reading metadata and table.
    :param path_metadata: path to metadata
    :param path_table: path to table
    :param force_to_dense: to force the matrix to np.ndarray
    :return: metadata and table
    """
    metadata, table = gcds.read_from_path(path_metadata, path_table)
    if isinstance(table, list):
        table = np.array(table)
    return metadata, table


def main():
    parser = argparse.ArgumentParser(description='Merges two tables in numeric form')
    parser.add_argument('-i', help='merge mode is intersection', action='store_true', default=False)
    parser.add_argument('-f', help='force csv tables to dense (assumes csv is numeric)',
                        action='store_true', default=False)
    parser.add_argument('m1', help='path to metadata 1')
    parser.add_argument('t1', help='path to table 1 (extension is used to determine loading)')
    parser.add_argument('m2', help='path to metadata 2')
    parser.add_argument('t2', help='path to table 2 (extension is used to determine loading)')
    parser.add_argument('out', help='output prefix used to save merged metadata and table')
    args = parser.parse_args()

    assert os.path.exists(args.m1), "metadata does not exist '{}'".format(args.m1)
    assert os.path.exists(args.t1), "table does not exist '{}'".format(args.t1)
    assert os.path.exists(args.m2), "metadata does not exist '{}'".format(args.m2)
    assert os.path.exists(args.t2), "table does not exist '{}'".format(args.t2)
    metadata_path = args.out + ".json"
    assert not os.path.exists(metadata_path), "metadata path already exists '{}'".format(metadata_path)

    metadata1, table1 = read(args.m1, args.t1, args.f)
    metadata2, table2 = read(args.m2, args.t2, args.f)
    metadata, table = merge.merge(metadata1, table1, metadata2, table2, intersection=args.i)
    gcds.write_to_path(metadata, table, args.out)


if __name__ == "__main__":
    main()
