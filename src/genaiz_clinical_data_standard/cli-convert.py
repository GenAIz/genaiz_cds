
import os
import argparse

from genaiz_clinical_data_standard import gcds


def main():
    parser = argparse.ArgumentParser(description='Convert megadata table format into (preferred) GCDS format')
    parser.add_argument('standard', help='standard to apply such as {}'.format(gcds.DEMOGRAPHIC))
    parser.add_argument('megadata', help='megadata table path')
    parser.add_argument('out', help='output path prefix')
    args = parser.parse_args()

    assert args.standard in gcds.STANDARDS, "unknown standard '{}'".format(args.standard)
    assert os.path.exists(args.megadata), "megadata table does not exist '{}'".format(args.megadata)

    metadata_path = args.out + ".json"
    table_path = args.out + ".csv"
    assert not os.path.exists(metadata_path), "metadata path already exists '{}'".format(metadata_path)
    assert not os.path.exists(table_path), "table path already exists '{}'".format(table_path)

    errors = set()
    with open(args.megadata) as in_stream:
        metadata, table = gcds.read_mega_data(args.standard, in_stream, errors)
    if errors:
        print("There were errors while loading the data:")
        for e in errors:
            print(" -", e)
    else:
        gcds.write_to_path(metadata, table, args.out)


if __name__ == "__main__":
    main()
