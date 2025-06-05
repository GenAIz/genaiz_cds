
import os
import argparse

import numpy as np

from genaiz_clinical_data_standard import gcds


def main():
    parser = argparse.ArgumentParser(description='Convert a numeric-only table to Matrix Market format')
    parser.add_argument('metadata', help='path to metadata')
    parser.add_argument('table', help='path to table (extension is used to determine loading)')
    parser.add_argument('out', help='output prefix used to save metadata and table')
    args = parser.parse_args()

    assert os.path.exists(args.metadata), "metadata does not exist '{}'".format(args.metadata)
    assert os.path.exists(args.table), "table does not exist '{}'".format(args.table)
    assert args.table.endswith('csv'), "table not in CSV format, or wrong extension"
    metadata_path = args.out + ".json"
    table_path = args.out + ".mm"
    assert not os.path.exists(metadata_path), "metadata path already exists '{}'".format(metadata_path)
    assert not os.path.exists(table_path), "table path already exists '{}'".format(table_path)

    metadata, table = gcds.read_from_path(args.metadata, args.table)
    table = np.array(table)
    gcds.write_to_path(metadata, table, args.out)


if __name__ == "__main__":
    main()
