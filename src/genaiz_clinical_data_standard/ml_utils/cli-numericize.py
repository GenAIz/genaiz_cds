
import os
import argparse

from genaiz_clinical_data_standard import gcds, gcds_utils
from genaiz_clinical_data_standard.ml_utils import numericize


def main():
    parser = argparse.ArgumentParser(description='Convert GCDS data to numeric form (without inspection)')
    parser.add_argument('metadata', help='path to metadata')
    parser.add_argument('table', help='path to table (extension is used to determine loading)')
    parser.add_argument('out', help='output prefix used to save metadata and table')
    parser.add_argument('--standard', help='standard to apply such as {}'.format(gcds.DEMOGRAPHIC), default=None)
    args = parser.parse_args()

    assert os.path.exists(args.metadata), "metadata does not exist '{}'".format(args.metadata)
    assert os.path.exists(args.table), "table does not exist '{}'".format(args.table)
    metadata_path = args.out + ".json"
    assert not os.path.exists(metadata_path), "metadata path already exists '{}'".format(metadata_path)

    metadata, table = gcds.read_from_path(args.metadata, args.table)
    num_metadata, num_table = numericize.numericize(metadata, table, standard=args.standard)
    gcds.write_to_path(num_metadata, num_table, args.out)


if __name__ == "__main__":
    main()
