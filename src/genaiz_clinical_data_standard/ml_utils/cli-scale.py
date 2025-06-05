
import os
import argparse

import numpy as np
import joblib

from genaiz_clinical_data_standard import gcds
from genaiz_clinical_data_standard.ml_utils import scale


def main():
    parser = argparse.ArgumentParser(description='Scale numeric table data')
    parser.add_argument('metadata', help='path to metadata')
    parser.add_argument('table', help='path to table (extension is used to determine loading)')
    parser.add_argument('out', help='output prefix used to save metadata and table')
    parser.add_argument('--scaler', help='path to use to load scaler', default=None)
    parser.add_argument('--savescaler', help='path to use to save scaler', default=None)
    args = parser.parse_args()

    assert os.path.exists(args.metadata), "metadata does not exist '{}'".format(args.metadata)
    assert os.path.exists(args.table), "table does not exist '{}'".format(args.table)
    metadata_path = args.out + ".json"
    table_path = args.out + ".mm"
    assert not os.path.exists(metadata_path), "metadata path already exists '{}'".format(metadata_path)
    assert not os.path.exists(table_path), "table path already exists '{}'".format(table_path)
    if args.savescaler:
        assert not os.path.exists(args.savescaler), "scaler path already exists '{}'".format(args.savescaler)
    scaler = None
    if args.scaler:
        assert os.path.exists(args.scaler), "scaler does not exist '{}'".format(args.scaler)
        scaler = joblib.load(args.scaler)
        print("Using existing scaler")

    metadata, table = gcds.read_from_path(args.metadata, args.table)
    if isinstance(table, list):
        table = np.array(table)
    standardScaler, table = scale.scale(table, scaler=scaler)
    gcds.write_to_path(metadata, table, args.out)
    if args.savescaler and not args.scaler:
        joblib.dump(standardScaler, args.savescaler)


if __name__ == "__main__":
    main()
