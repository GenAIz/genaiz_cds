
import os
import argparse

from genaiz_clinical_data_standard import gcds


def main():
    parser = argparse.ArgumentParser(description='Inspect and validate GCDS')
    parser.add_argument('standard', help='standard to apply such as {}'.format(gcds.DEMOGRAPHIC))
    parser.add_argument('metadata', help='path to metadata')
    parser.add_argument('table', help='path to table (extension is used to determine loading)')
    args = parser.parse_args()

    assert args.standard in gcds.STANDARDS, "unknown standard '{}'".format(args.standard)
    assert os.path.exists(args.metadata), "metadata does not exist '{}'".format(args.metadata)
    assert os.path.exists(args.table), "table does not exist '{}'".format(args.table)

    metadata, table = gcds.read_from_path(args.metadata, args.table)
    errors = set()
    gcds.inspect_gcds(args.standard, metadata, table, errors)
    if errors:
        print("There were errors while inspecting the data:")
        for e in errors:
            print(" -", e)
    else:
        print("No errors.")


if __name__ == "__main__":
    main()
