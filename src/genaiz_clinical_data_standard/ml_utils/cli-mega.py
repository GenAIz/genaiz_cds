
import os
import argparse
import csv

from genaiz_clinical_data_standard import gcds, gcds_utils


def main():
    parser = argparse.ArgumentParser(description='Merges metadata and table into a megadata table (some info is lost)')
    parser.add_argument('metadata', help='path to metadata')
    parser.add_argument('table', help='path to table (extension is used to determine loading)')
    parser.add_argument('out', help='output table path')
    args = parser.parse_args()

    assert os.path.exists(args.metadata), "metadata does not exist '{}'".format(args.metadata)
    assert os.path.exists(args.table), "table does not exist '{}'".format(args.table)
    assert not os.path.exists(args.out), "table path already exists '{}'".format(args.out)

    metadata, table = gcds.read_from_path(args.metadata, args.table)
    if not isinstance(table, list):
        table = gcds_utils.to_list_table(table)

    col_names = []
    semantics = []
    dtypes = []
    states = []
    unknown_values = []
    for md in metadata:
        col_names.append(md.name)
        semantics.append(md.semantics)
        dtypes.append(md.dtype)
        states.append(md.state)
        unknown_values.append(md.unknown_value)

    with open(args.out, 'w') as out_stream:
        writer = csv.writer(out_stream)
        writer.writerow(col_names)
        writer.writerow(semantics)
        writer.writerow(dtypes)
        writer.writerow(states)
        writer.writerow(unknown_values)
        writer.writerows(table)


if __name__ == "__main__":
    main()
