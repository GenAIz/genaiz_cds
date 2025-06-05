# GENAIZ Clinical data standard (GCDS)
This repository contains code implementing the GCDS in python.

The GCDS is described in `doc/GenAIz_clinical_data_standard.docx`.

GCDS allows a person to convert between a megadata table and the GCDS, and check any existing GCDS data for correctness.  There is a CLI and API that facilitates this functionality.

This repository also contains utility functions for machine learning.  In particular, the functionality to convert GCDS to numerical only data and to merge tables, or matrices, of this data is provided.

__NOTE__: Compressed sparse column (CSC) format is preferred for sparse matrices because CSC format allows for efficient column manipulations.


# Setup
To setup this repository do the following from within the repository's root:
```
python3.10 -m venv pyvenv
source pyvenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Set the python path: `export PYTHONPATH=$PWD/src`.

Note: a linux-like development environment is assumed.


# Unit tests
Unit tests can be run with `python -m unittest discover -s tests`.


# Example use
The commands below demonstrate conversion and inspection of demographic data.   Pseudo-demographic data is used in this example (from the unit test data directory).  Note that a separate output directory called `data` is created to store output.
```commandline
mkdir data
python src/genaiz_clinical_data_standard/cli-convert.py gcds_demographic_data tests/data/megadata.csv data/testdata
python src/genaiz_clinical_data_standard/cli-check.py gcds_demographic_data data/testdata.json data/testdata.csv
```

Building on the commands and data manipulation above, the commands below demonstrate data manipulation for machine learning.  The following command converts data to numerical form.  
```commandline
python src/genaiz_clinical_data_standard/ml_utils/cli-numericize.py --standard gcds_demographic_data data/testdata.json data/testdata.csv data/numeric
```

The data is converted to matrix market format, which is easy to load using `scipy`.
```commandline
python src/genaiz_clinical_data_standard/ml_utils/cli-mm.py data/numeric.json data/numeric.csv data/numeric_mm
```

If one follows all the above commands then there is two copies of the same table in two different formats.  Although silly, the following command will merge these.  Note that the command includes a switch which allows for intersection vs union merges (see merge function documentation).
```commandline
python src/genaiz_clinical_data_standard/ml_utils/cli-merge.py data/numeric.json data/numeric.csv data/numeric_mm.json data/numeric_mm.mm data/merged
```

Finally, the data can be scaled.  This is usually the last step prior to training a model.  The first example below saves the scaler.  The second example is similar to the first but loads the saved scaler.
```commandline
python src/genaiz_clinical_data_standard/ml_utils/cli-scale.py data/merged.json data/merged.mm data/scaled --savescaler data/scaled.scaler
python src/genaiz_clinical_data_standard/ml_utils/cli-scale.py data/merged.json data/merged.mm data/rescaled --scaler data/scaled.scaler
```

## Forcing tabular data to GCDS
Tabular data can be forced into GCDS format. Doing this will result in data that does not necessarily conform to standard, but is usable (e.g., ML).  Note that the "forced" data may need to be further adjusted or modified.  For example, an identifier column may need to be added.
```commandline
python src/genaiz_clinical_data_standard/ml_utils/cli-force.py data/testdata.csv data/forced_example
```

## Generating a mega data file
One can generate mega data files by combining the metadata and table.  This is usually done on small data sets so that the data can be viewed more easily for debugging purposes.   
```commandline
python src/genaiz_clinical_data_standard/ml_utils/cli-mega.py metadata.json table.csv table.mega.csv
```

# Distribution
Running the command `python3 -m build --wheel` will build a wheel which may be installed by pip after.

Please update the 'version' in pyproject.toml file everytime a change occurs in the code, and run this command to create a new updated wheel. You may need to manually install the build module with `pip install build`

The change log must be updated as well.
