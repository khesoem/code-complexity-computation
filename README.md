# Replication Package

This repository provides the programs and data necessary to replicate the results of our research paper "*Cognitive Code Complexity: An EEG Validation Study*".

## Contents
The repository includes:
- **MCC Computation Tool**: The tool we have developed to compute MCC-PD and MCC-MPI for Python programs.
- **Data Analysis Artifact**: The data and analysis related to the complexity of code snippets and the participants' cognitive load.

### MCC Computation Tool
You can run the MCC computation tool using the following command, after installing the required dependencies via poetry:
```
python main.py -r <path_to_code_snippets_directory>
```
Replace `<path_to_code_snippets_directory>` with the path to the directory containing your Python code snippets.
To run the tool on samples considered in the paper, replace `<path_to_code_snippets_directory>` with `./samples`.

Note that the tool works for a subset of Python programs that follow the grammar indicated in Listing 2 of the paper.

### Data Analysis Artifact
The samples used in the paper are available in the `samples` directory.

The `data-analysis` directory contains the scripts and data used to analyze the complexity of the code snippets and the participants' cognitive load as follows:
- **`compute_traditional_metrics.py`**: Computes traditional complexity metrics such as Lines of Code (LOC) and Cyclomatic Complexity.
- **`llm.py`**: Runs a linear mixed model analysis on the EEG data.
- **`snippet_metrics.csv`**: Contains the complexity metrics computed for the code snippets.
This includes LOC, Cyclomatic Complexity, DepDegree, Halstead Vocabulary, and MCC metrics.
DepDegree and Halstead Vocabulary are computed manually with the assistance of OpenAI's o3 model.
- **`eeg.csv`**: Contains cognitive load based the EEG data collected from participants while they were tracing code snippets.
- **`survey_cl.csv`**: Contains cognitive load based on the survey data collected from participants after tracing code snippets.
- **`cl_me_correlation.csv`**: Calculates the correlation between cognitive load based on EEG and survey data.
- **`eeg-analysis/eeg_pipeline.m`**: Contains the entire pipeline for analyzing EEG data and computing final correlations.
This file is a MATLAB script that requires the [EEGLAB toolbox](https://eeglab.org/tutorials/01_Install/Install.html) to run.
The script provides a user-friendly interface to run the experiment with various data cleaning and filtering techniques.

## Remark
The repository contains all materials necessary for replication except the raw EEG data from the participants, owing to privacy concerns because of the biometric identification potential.
