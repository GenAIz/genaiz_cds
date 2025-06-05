"""
GENAIZ clinical data standard (GCDS)

This is a python implementation of the GCDS.
This module facilitates loading, inspecting and working with GCDS data.
"""

import re
import math
import datetime
import csv
import json
import os

import numpy as np
import scipy


# Supported genaiz clinical data standards
DEMOGRAPHIC = "gcds_demographic_data"
SEROLOGY = "gcds_serology_data"
FLOW_CYTOMETRY = "gcds_flow_cytometry_data"
SINGLE_CELL_SEQ = "gcds_single_cell_seq"

STANDARDS = frozenset([DEMOGRAPHIC, SEROLOGY, FLOW_CYTOMETRY, SINGLE_CELL_SEQ])

# Data types
T_CATEGORICAL = "categorical"
T_IDENTIFIER = "identifier"
T_TARGET = "target"
T_BOOLEAN = "boolean"
T_SET = "set"
T_NUMERIC = "numeric"
T_DISCRETE = "discrete"
T_CONTINUOUS = "continuous"
T_RANGE = "range"
T_CURRENCY = "currency"
T_DATE = "date"
T_DURATION = "duration"
T_IGNORE = "ignore"

DATA_TYPES = frozenset({
    T_CATEGORICAL, T_IDENTIFIER, T_TARGET, T_BOOLEAN, T_SET, T_NUMERIC, T_DISCRETE, T_CONTINUOUS,
    T_RANGE, T_CURRENCY, T_DATE, T_DURATION, T_IGNORE
})

COL_PARTICIPANT_ID = "participant_identifier"
COL_CLINICAL_GROUP = "clinical_group"
COL_COLLECTION_NAME = "collection_name"
COL_EXPERIMENTAL_CONDITION = "experimental_condition"

# Mappings from column name to type.  Allows users
# to omit the type when completing data.

DEMOGRAPHIC_INFERENCE_MAP = {
    COL_PARTICIPANT_ID: T_IDENTIFIER,
    COL_CLINICAL_GROUP: T_TARGET,
    "age": T_NUMERIC,
    "sex": T_BOOLEAN,
    "race": T_CATEGORICAL,
    "ethnicity": T_CATEGORICAL,
    "education_level": T_CATEGORICAL,
    "salary_bracket": T_RANGE,
    "nationality": T_SET
}

SEROLOGY_INFERENCE_MAP = {
    COL_PARTICIPANT_ID: T_IDENTIFIER,
    COL_COLLECTION_NAME: T_CATEGORICAL,
    "sample_identifier": T_IDENTIFIER,
    "sample_collection_date": T_DATE,
    "assay_type": T_CATEGORICAL,
    "initial_symptoms": T_SET,
    "symptom_severity": T_CATEGORICAL,
    "comorbidities": T_SET,
    "duration_of_symptoms": T_DISCRETE,
    "period_from_onset": T_DISCRETE
}

FLOW_CYTOMETRY_INFERENCE_MAP = {
    COL_PARTICIPANT_ID: T_IDENTIFIER,
    COL_EXPERIMENTAL_CONDITION: T_CATEGORICAL,
    "sample_identifier": T_IDENTIFIER
}

INFERENCE_MAP = {
    DEMOGRAPHIC: DEMOGRAPHIC_INFERENCE_MAP,
    SEROLOGY: SEROLOGY_INFERENCE_MAP,
    FLOW_CYTOMETRY: FLOW_CYTOMETRY_INFERENCE_MAP
}

# Recommended unknown value
DEFAULT_UNKNOWN = "!unknown"

# Reserved state which represents data from source, such as an instrument
STATE_SOURCE = "source"

# Reserved char that separates items in a set
SET_SEPARATOR = ","


class GCDSMetadata:
    """
    Represent GCDS meta data - usually a column of data within a table.
    """

    # column index
    INDEX = "index"
    # human readable name for the data
    NAME = "name"
    # computer readable semantics of the data
    SEMANTICS = "semantics"
    # data type
    DTYPE = "dtype"
    # state of the data, typically processing state
    STATE = "state"
    # when creating new data precedence is given to the fill value
    # if the fill value is None then the unknown value is used instead
    # allows fill values to be known rather than defaulting to unknown
    FILL_VALUE = "fill_value"
    # unknown value used by this column
    # this is important for sparse matrices which are saved in matrix market format which cannot
    # represent NaN or anything other than a number
    UNKNOWN_VALUE = "unknown_value"
    # mapping from original column values to integers such as mapping identifiers to numeric values
    MAPPING = "mapping"

    RE_SPACE = re.compile("\\s+")

    def __init__(self, index: int = -1, name: str | None = None, semantics: str | None = None,
                 dtype: str | None = None, state: str | None = None, unknown_value=None, fill_value=None):
        self.index: int = index
        self.name: str | None = name
        self.semantics: str | None = semantics
        self.dtype: str | None = dtype
        self.state: str | None = state
        self.unknown_value = unknown_value
        self.fill_value = fill_value
        self.mapping: dict[str, int] | None = None

    def normalized_name(self):
        """
        :return: the lowercase column name with spaces converted to underscores (extra spaces are removed)
        """
        name = self.RE_SPACE.subn(" ", self.name)[0]
        return name.strip().lower().replace(" ", "_")

    def is_coherent(self, errors: set[str] = None) -> bool:
        """
        Check if this metadata is minimally coherent
        :param errors: bucket for errors
        :return: true if this metadata is minimally coherent
        """
        if errors is None:
            errors = set()
        return not inspect_metadata(self, errors)

    def inspect_data_column(self, data: list | np.ndarray | scipy.sparse.spmatrix, errors: set[str]) -> bool:
        """
        Determines if data conforms to this metadata definition.
        :param data: data to check
        :param errors: bucket for errors
        :return: true if the data conforms to this metadata definition
        """
        return inspect_data_column(self, data, errors)

    def initial_value(self):
        """
        Return an initial value for an empty cell, which is the fill value unless it is None
        in which case it is the unknown value.
        :return: an initial value where the fill value takes precedence
        """
        return self.unknown_value if self.fill_value is None else self.fill_value

    def from_dictionary(self, dictionary: dict):
        """
        Copies the dictionary metadata values into this metadata.
        :param dictionary: dictionary to copy values from
        :return: this metadata object (convenience of GCDSMetadata().from_dictionary(d))
        """
        self.index = dictionary[self.INDEX]
        self.name = dictionary[self.NAME]
        self.semantics = dictionary[self.SEMANTICS]
        self.dtype = dictionary[self.DTYPE]
        self.state = dictionary[self.STATE]
        self.unknown_value = dictionary[self.UNKNOWN_VALUE]
        self.fill_value = dictionary[self.FILL_VALUE]
        self.mapping = dictionary[self.MAPPING]
        return self

    def to_dictionary(self, dictionary: dict = None) -> dict:
        """
        Copies this metadata into a dictionary.
        :param dictionary: dictionary to fill, if None then creates one and returns it
        :return: the dictionary with a copy of this metadata
        """
        if dictionary is None:
            dictionary = dict()
        dictionary[self.INDEX] = self.index
        dictionary[self.NAME] = self.name
        dictionary[self.SEMANTICS] = self.semantics
        dictionary[self.DTYPE] = self.dtype
        dictionary[self.STATE] = self.state
        dictionary[self.FILL_VALUE] = self.fill_value
        dictionary[self.UNKNOWN_VALUE] = self.unknown_value
        dictionary[self.MAPPING] = None if self.mapping is None else self.mapping.copy()
        return dictionary

    def __eq__(self, other):
        if not isinstance(other, GCDSMetadata):
            return False
        return self.index == other.index and self.name == other.name and self.semantics == other.semantics \
            and self.dtype == other.dtype and self.state == other.state \
            and self.unknown_value == other.unknown_value and self.fill_value == other.fill_value \
            and self.mapping == other.mapping

    def __str__(self):
        a = "index={}, name='{}', semantics='{}', dtype='{}', state='{}'".format(
            self.index, self.name, self.semantics, self.dtype, self.state
        )
        b = "unknown_value='{}', fill_value='{}', mapping='{}'".format(
            self.unknown_value, self.fill_value, self.mapping
        )
        return "GCDSMetadata({}, {})".format(a, b)

    def __repr__(self):
        return self.__str__()


def infer_metadata(table: list[list] | np.ndarray | scipy.sparse.spmatrix) -> list[GCDSMetadata]:
    """
    Infers metadata from the given data table.  The table may be a megadata table or a matrix
    of numeric values.
    :param table: a table from which metadata is inferred
    :return: the metadata
    """
    if isinstance(table, list):
        metadata = infer_metadata_from_csv(table)
    elif isinstance(table, np.ndarray) or scipy.sparse.issparse(table):
        metadata = infer_metadata_from_matrix(table)
    else:
        raise ValueError("Cannot infer metadata from type '{}'".format(type(table)))
    return metadata


def infer_metadata_from_csv(table: list[list]) -> list[GCDSMetadata]:
    """
    Infers metadata from a megadata table (CSV format) described in the GCDS.
    :param table: the megadata table
    :return: the metadata pulled from the megadata table
    """
    if len(table) < 5:
        raise ValueError("Missing (mandatory) metadata rows")
    metadata: list[GCDSMetadata] = []
    for j in range(len(table[0])):
        md = GCDSMetadata()
        md.index = j
        md.name = table[0][j]
        md.semantics = table[1][j]
        md.dtype = table[2][j]
        md.state = table[3][j]
        md.unknown_value = table[4][j]
        metadata.append(md)
    return metadata


def infer_metadata_from_matrix(table: np.ndarray | scipy.sparse.spmatrix) -> list[GCDSMetadata]:
    """
    Infer metadata from a sparse or dense matrix.  A narrow set of types are supported/allowed,
    being float, integer and boolean.
    :param table: the matrix
    :return: the metadata pulled from the matrix
    """
    if len(table.shape) != 2:
        raise ValueError("Cannot handle numpy array of shape {} (only 2d is allowed)")
    metadata: list[GCDSMetadata] = []
    for i in range(0, table.shape[1]):
        metadata.append(GCDSMetadata(index=i, name="col_{}".format(i), state=STATE_SOURCE))
    if np.issubdtype(table.dtype, np.floating):
        for md in metadata:
            md.dtype = T_CONTINUOUS
    elif np.issubdtype(table.dtype, np.integer):
        for md in metadata:
            md.dtype = T_DISCRETE
    elif np.issubdtype(table.dtype, np.bool):
        for md in metadata:
            md.dtype = T_BOOLEAN
    else:
        raise ValueError("Cannot handle numpy dtype {} (only float, int and bool is allowed)".format(table.dtype))
    return metadata


def infer_missing_data_types(standard: str, metadata: list[GCDSMetadata]) -> None:
    """
    Infers missing data types given some standard (e.g., demographic).  The standard's
    reserved column names are used to infer data types.
    :param standard: the standard to apply
    :param metadata: the metadata to complete
    """
    if standard not in INFERENCE_MAP:
        raise ValueError("Unknown standard {}".format(standard))
    inference_map = INFERENCE_MAP[standard]
    for md in metadata:
        if md.dtype:
            continue
        name = md.normalized_name()
        if name in inference_map:
            md.dtype = inference_map[name]


# CSV data needs to coerced into more appropriate types
# The mapping below indicates how this should be and is done.
DATA_TYPE_COERCION_MAP = {
    T_CATEGORICAL: str, T_IDENTIFIER: str, T_TARGET: str, T_BOOLEAN: str,
    T_SET: str, T_NUMERIC: float, T_DISCRETE: int, T_CONTINUOUS: float,
    T_RANGE: str, T_CURRENCY: float, T_DATE: str, T_DURATION: float
}


def coerce_csv_data(metadata: list[GCDSMetadata], table: list[list]) -> None:
    """
    Coerce (CSV) table data into more appropriate values.
    :param metadata: metadata used to guide coercion
    :param table: table to coerce (cannot be in megadata format - cannot start with metadata)
    """
    metadata.sort(key=lambda md_: md_.index)
    for j in range(len(metadata)):
        md = metadata[j]
        if md.dtype == T_IGNORE:
            continue
        for i in range(len(table)):
            if table[i][j] == md.unknown_value:
                continue
            func = DATA_TYPE_COERCION_MAP[md.dtype]
            if isinstance(md.mapping, dict) and func == str:
                # have a mapping from str to int so we don't force the values to string, but rather to int
                func = int
            table[i][j] = func(table[i][j])


# regex describing valid data names (column names)
COL_NAME_REGEX = re.compile(r"(\w|-|_| |\||/|\*|\(|\)|\+|>|<|=)+")


def inspect_metadata(metadata: GCDSMetadata, errors: set[str]) -> bool:
    """
    Inspects metadata to determine if it minimally conforms to the metadata standard.
    :param metadata: metadata to check
    :param errors: bucket for errors
    :return: true if there were errors
    """
    if not isinstance(metadata.index, int) or metadata.index < 0:
        errors.add("Metadata with index {} has a bad index".format(metadata.index))
    if COL_NAME_REGEX.fullmatch(metadata.name) is None:
        errors.add(
            "Metadata {} with name '{}' contains invalid characters".format(metadata.index, metadata.name)
        )
    # semantics are to be defined.
    if not isinstance(metadata.dtype, str) or metadata.dtype not in DATA_TYPES:
        errors.add("Metadata with index {} has a bad type '{}'.".format(metadata.index, metadata.dtype))
    if not isinstance(metadata.state, str):
        errors.add("Metadata with index {} has a bad state '{}'.".format(metadata.index, metadata.dtype))
    if metadata.unknown_value is None \
            or (isinstance(metadata.unknown_value, str) and metadata.unknown_value.strip() == ""):
        errors.add("Metadata with index {} has a bad unknown value of 'None'.".format(metadata.index))
    return len(errors) != 0


# regex describing valid identifiers
IDENTIFIER_REGEX = re.compile(r"(\w|-|_|:|\.)+")

# regex describing valid numeric values
NUMERIC_REGEX = re.compile(
    r"(-?[1-9]\.[0-9]+[eE]-?[0-9]+|-?[0-9]+\.[0-9]+|-?[0-9]+\.?|-?inf)"
)

# regex describing valid ranges
RANGE_REGEX = re.compile(NUMERIC_REGEX.pattern + " *- *" + NUMERIC_REGEX.pattern)


def inspect_value(metadata: GCDSMetadata, value: None | str | int | float, errors: set[str]) -> bool:
    """
    Inspects a single value to determine if it conforms to the given metadata standard.
    :param metadata: metadata
    :param value: value to check
    :param errors: bucket for errors
    :return: true if there were errors
    """
    if metadata.dtype == T_IGNORE:
        return False
    if value is None:
        errors.add("Column {} '{}' has a missing value".format(metadata.index, metadata.name))
        return True
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            errors.add("Column {} '{}' has a missing value".format(metadata.index, metadata.name))
            return True
    if value == metadata.unknown_value \
            or (metadata.unknown_value == DEFAULT_UNKNOWN and isinstance(value, str)
                and value.startswith(DEFAULT_UNKNOWN)):
        return False
    if metadata.dtype == T_CATEGORICAL or metadata.dtype == T_TARGET:
        if not isinstance(value, str):
            errors.add("Column {} '{}' has bad value '{}'".format(metadata.index, metadata.name, value))
    elif metadata.dtype == T_IDENTIFIER:
        if not isinstance(value, str) or IDENTIFIER_REGEX.fullmatch(value) is None:
            errors.add("Bad identifier '{}' in column {} '{}'".format(value, metadata.index, metadata.name))
    elif metadata.dtype == T_BOOLEAN:
        if not isinstance(value, str):
            errors.add("Column {} '{}' has bad value {}".format(metadata.index, metadata.name, value))
    elif metadata.dtype == T_SET:
        if not isinstance(value, str):
            errors.add("Column {} '{}' has bad value '{}'".format(metadata.index, metadata.name, value))
    elif metadata.dtype == T_NUMERIC or metadata.dtype == T_CONTINUOUS or metadata.dtype == T_CURRENCY:
        if not isinstance(value, float) and not isinstance(value, int):
            errors.add("Bad value '{}' in column {} '{}'".format(value, metadata.index, metadata.name))
    elif metadata.dtype == T_DISCRETE:
        if not (isinstance(value, int) or (isinstance(value, float) and (value.is_integer() or math.isnan(value)))):
            errors.add("Bad discrete value '{}' in column {} '{}'".format(value, metadata.index, metadata.name))
    elif metadata.dtype == T_RANGE:
        if isinstance(value, str):
            mo = RANGE_REGEX.match(value)
            if mo is None or float(mo.group(1)) >= float(mo.group(2)):
                errors.add("Bad range value '{}' in column {} '{}'".format(value, metadata.index, metadata.name))
        else:
            errors.add("Bad range value '{}' in column {} '{}'".format(value, metadata.index, metadata.name))
    elif metadata.dtype == T_DATE:
        try:
            datetime.datetime.fromisoformat(value)
        except (ValueError, TypeError):
            errors.add("Bad date value '{}' in column {} '{}'".format(value, metadata.index, metadata.name))
    elif metadata.dtype == T_DURATION:
        if not isinstance(value, float) and not isinstance(value, int) or value <= 0:
            errors.add("Bad value '{}' in column {} '{}'".format(value, metadata.index, metadata.name))
    else:
        raise ValueError("Unknown metadata type.")
    return len(errors) != 0


# maps the known standards to a set of required columns
REQUIRED_COLUMNS = {
    DEMOGRAPHIC: frozenset({COL_PARTICIPANT_ID, COL_CLINICAL_GROUP}),
    SEROLOGY: frozenset({COL_PARTICIPANT_ID, COL_COLLECTION_NAME}),
    FLOW_CYTOMETRY: frozenset({COL_PARTICIPANT_ID, COL_EXPERIMENTAL_CONDITION})
}


def inspect_gcds_metadata(standard: str, metadata: list[GCDSMetadata], errors: set[str]) -> bool:
    """
    Inspect metadata given a GCDS to determine if the metadata conforms to the standard.
    :param standard: standard to apply (e.g., demographic)
    :param metadata: metadata to check
    :param errors: bucket for errors
    :return: true if there were errors
    """
    if standard not in REQUIRED_COLUMNS:
        raise ValueError("Unknown standard {}".format(standard))
    inference_map = INFERENCE_MAP[standard]
    required: set[str] = set(REQUIRED_COLUMNS[standard])
    duplicates: set[str] = set()
    indexes: set = set()
    check_targets: bool = False
    for md in metadata:
        indexes.add(md.index)
        if md.dtype == T_TARGET:
            if check_targets:
                errors.add("Cannot have multiple target columns, '{}'.".format(md.name))
            check_targets = True
        normed_name = md.normalized_name()
        if normed_name in duplicates:
            errors.add("Duplicate column name '{}'".format(normed_name))
        else:
            duplicates.add(normed_name)
        inspect_metadata(md, errors)
        if normed_name in required:
            required.remove(normed_name)
        if md.index == 0 and md.dtype != T_IDENTIFIER:
            errors.add("Left most column is not an identifier, instead it is '{}'".format(md.dtype))
        if normed_name in inference_map and inference_map[normed_name] != md.dtype:
            errors.add("Column's type does not match standard: '{}' is '{}' but should be '{}'".format(
                md.name, md.dtype, inference_map[normed_name])
            )
    if required:
        errors.add("Missing required column(s): {}".format(required))
    if len(indexes) != len(metadata):
        errors.add("Cannot have duplicate indexes.")
    if 0 not in indexes:
        errors.add("Column indexes must start at 0 and don't.")
    if len(metadata) - 1 not in indexes:
        errors.add("Column indexes must stop at {} and don't.".format(len(metadata) - 1))
    return len(errors) != 0


def inspect_data_column(metadata: GCDSMetadata, column: list | np.ndarray | scipy.sparse.spmatrix,
                        errors: set[str]) -> bool:
    """
    Inspect a column of data to determine if it conforms to the given metadata.
    :param metadata: metadata
    :param column: column to check relative to the metadata
    :param errors: bucket for errors
    :return: true if there were errors
    """
    generate_error: bool = False
    if isinstance(column, list):
        if metadata.dtype == T_BOOLEAN:
            unique_values = set()
            for value in column:
                inspect_value(metadata, value, errors)
                unique_values.add(value)
            if metadata.unknown_value in unique_values:
                unique_values.remove(metadata.unknown_value)
            if len(unique_values) > 2:
                errors.add("Boolean column '{}' cannot have more than 2 values".format(metadata.name))
                generate_error = True
        else:
            for value in column:
                inspect_value(metadata, value, errors)
    elif metadata.dtype in {T_NUMERIC, T_CONTINUOUS, T_CURRENCY, T_DURATION}:
        if not np.issubdtype(column.dtype, np.integer) and not np.issubdtype(column.dtype, np.floating):
            generate_error = True
    elif metadata.dtype == T_DISCRETE:
        if not np.issubdtype(column.dtype, np.integer):
            generate_error = True
    else:
        generate_error = True
    if generate_error:
        errors.add("Type mismatch between metadata and data for column '{}'".format(metadata.name))
    return generate_error


def inspect_data(metadata: list[GCDSMetadata], table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                 errors: set[str]) -> bool:
    """
    Inspect data to determine if it conforms to the given metadata.
    :param metadata: metadata
    :param table: table to check relative to the metadata
    :param errors: bucket for errors
    :return: true if there were errors
    """
    if isinstance(table, list):
        column = []
        for j in range(len(metadata)):
            column.clear()
            for i in range(len(table)):
                column.append(table[i][j])
            inspect_data_column(metadata[j], column, errors)
    else:
        for j in range(len(metadata)):
            inspect_data_column(metadata[j], table[:, j], errors)
    return len(errors) != 0


def inspect_gcds(standard: str, metadata: list[GCDSMetadata],
                 table: list[list] | np.ndarray | scipy.sparse.spmatrix, errors: set[str]) -> bool:
    """
    Inspect metadata and data to determine if, as a whole, they conform to the given standard (e.g., demographic).
    :param standard: standard to apply (e.g., demographic)
    :param metadata: metadata to check
    :param table: table to check
    :param errors: bucket for errors
    :return: true if there were errors
    """
    if inspect_gcds_metadata(standard, metadata, errors):
        return True
    return inspect_data(metadata, table, errors)


def read_mega_data(standard: str, in_stream, errors: set[str]) -> tuple[list[GCDSMetadata] | None, list[list] | None]:
    """
    Reads a megadata table into metadata and a table.  Metadata is inferred and missing metadata
    is added.  Data is coerced and checked.
    :param standard: standard to apply (e.g., demographic)
    :param in_stream: stream containing the megadata table
    :param errors: bucket for errors
    :return: metadata and table
    """
    reader = csv.reader(in_stream)
    table: list[list] = []
    for row in reader:
        table.append(row)
    if len(table) > 0 and len(table[0]) > 0:
        # excel can leave a BOM character in CSVs
        table[0][0] = table[0][0].replace("\ufeff", "")
    metadata = infer_metadata(table)
    infer_missing_data_types(standard, metadata)
    if inspect_gcds_metadata(standard, metadata, errors):
        return None, None
    table = table[5:]
    coerce_csv_data(metadata, table)
    inspect_data(metadata, table, errors)
    return metadata, table


def read(metadata_stream, format_mm: bool,
         table_stream) -> tuple[list[GCDSMetadata], list[list] | np.ndarray | scipy.sparse.coo_matrix]:
    """
    Reads metadata from a stream.
    :param metadata_stream: the metadata stream
    :param format_mm: true if the stream contains matrix market format and false if it contains CSV format
    :param table_stream: the data table stream
    :param coerce: if true runs coerce_csv_data, only if the table is loaded from a csv
    :return: metadata and table
    """
    metadata = json.load(metadata_stream)
    for i in range(len(metadata)):
        metadata[i] = GCDSMetadata().from_dictionary(metadata[i])
    if format_mm:
        table: np.ndarray | scipy.sparse.coo_matrix = scipy.io.mmread(table_stream)
        if isinstance(table, scipy.sparse.coo_matrix):
            table = table.tocsc()
    else:
        reader = csv.reader(table_stream)
        table: list[list] = []
        for row in reader:
            table.append(row)
        coerce_csv_data(metadata, table)
    return metadata, table


def read_from_path(metadata_path: str, table_path: str) \
        -> tuple[list[GCDSMetadata], list[list] | np.ndarray | scipy.sparse.spmatrix]:
    """
    Reads metadata and a table from paths.
    :param metadata_path: the metadata path
    :param table_path: the table path
    :return: the metadata and table
    """
    with open(metadata_path) as metadata_in:
        is_mm_format = False if table_path.endswith(".csv") else True
        table_read_mode = 'rb' if is_mm_format else 'r'
        with open(table_path, table_read_mode) as table_in:
            return read(metadata_in, is_mm_format, table_in)


def write(metadata: list[GCDSMetadata], table: list[list] | np.ndarray | scipy.sparse.spmatrix,
          metadata_stream, table_stream) -> None:
    """
    Writes metadata and table to streams.  The table data format depends on the table's type.
    A list results in CSV output otherwise matrix market is tried.
    :param metadata: the metadata
    :param table: the table
    :param metadata_stream: the metadata output stream
    :param table_stream: the table output stream
    """
    json_metadata: list[dict] = []
    for i in range(len(metadata)):
        json_metadata.append(metadata[i].to_dictionary())
    json.dump(json_metadata, metadata_stream)
    if isinstance(table, list):
        writer = csv.writer(table_stream)
        writer.writerows(table)
    else:
        scipy.io.mmwrite(table_stream, table)


def write_to_path(metadata: list[GCDSMetadata], table: list[list] | np.ndarray | scipy.sparse.spmatrix,
                  path_prefix: str, cannot_exist: bool = True) -> None:
    """
    Writes metadata and table using the path prefix.  The prefix is combined with
    appropriate extensions (".mm", ".csv", ".json") given the type of data being written.

    :param metadata: the metadata
    :param table: the table
    :param path_prefix: the path prefix to which an extension is appended
    :param cannot_exist: if true then an error is thrown if files already exist
    """
    metadata_path = path_prefix + ".json"
    if cannot_exist and os.path.exists(metadata_path):
        raise IOError("metadata path/file already exists '{}'".format(metadata_path))
    if isinstance(table, list):
        table_path = path_prefix + ".csv"
        table_write_mode = 'w'
    else:
        table_path = path_prefix + ".mm"
        table_write_mode = 'bw'
    if cannot_exist and os.path.exists(metadata_path):
        raise IOError("table path/file already exists '{}'".format(table_path))
    with open(metadata_path, 'w') as metadata_out:
        with open(table_path, table_write_mode) as table_out:
            write(metadata, table, metadata_out, table_out)
