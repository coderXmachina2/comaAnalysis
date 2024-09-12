import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from scipy import stats
from statsmodels import robust

from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

# Initialize counters
def count_stuff(array_spects, txtPrnt=""):
    """
    Probe archive stuff.
    """
    _1dspect = 0
    _2dimg = 0
    _1darrays = []
    _2darrays = []
    # Iterate through the items in the data array
    print(txtPrnt, "N artefacts:", len(array_spects))
    for item in array_spects:
        # Check if the item is a numpy array
        if isinstance(item, np.ndarray):
            # print("Yes!")
            # Check if it's a 2D array by looking at the number of dimensions
            if item.ndim == 2:
                _2dimg += 1  # Increment counter for 2D images (nested arrays)
                _2darrays.append(item)
            elif item.ndim == 1:
                _1darrays.append(item)
                _1dspect += 1  # Increment counter for 1D spectrum (non-nested arrays)

    print("1D spectrum count:", _1dspect)
    print("2D image count:", _2dimg)

    return (_1darrays, _2darrays)


def extract_functions_and_args(lines):
    """
    # Function to extract function headers, their docstrings, and input arguments

    """
    function_definitions = []
    function_pattern = re.compile(r"^\s*def\s+(\w+)\((.*?)\):")
    in_docstring = False
    docstring_buffer = []
    current_function = ""
    current_function_name = ""
    current_function_args = ""

    for line in lines:
        # Detect function definitions and capture the function name and arguments
        match = function_pattern.match(line)
        if match:
            if current_function:
                # Append the current function, its argument count, and associated docstring
                argument_count = (
                    len(current_function_args.split(","))
                    if current_function_args
                    else 0
                )
                function_definitions.append(
                    f"{current_function}\n{argument_count} input arguments\n{''.join(docstring_buffer)}"
                )

            # Start a new function definition
            current_function = line.strip()
            current_function_name = match.group(1)
            current_function_args = match.group(2)
            docstring_buffer = []
            in_docstring = False

        # Detect starting and ending docstrings
        if '"""' in line or "'''" in line:
            docstring_buffer.append(line)
            in_docstring = not in_docstring
        elif in_docstring:
            docstring_buffer.append(line)

    # Add the last function in case there's one still in progress
    if current_function:
        argument_count = (
            len(current_function_args.split(",")) if current_function_args else 0
        )
        function_definitions.append(
            f"{current_function}\n{argument_count} input arguments\n{''.join(docstring_buffer)}"
        )

    return function_definitions


# Function to process Python files and extract comments, imports, and functions
def process_python_files(file_path):
    with open(file_path, "r") as py_file:
        lines = py_file.readlines()

    # Create a .txt file with no comment lines
    txt_filename = os.path.basename(file_path).replace(".py", ".txt")
    txt_filepath = os.path.join(txt_directory, txt_filename)
    filtered_lines = [line for line in lines if not line.lstrip().startswith("#")]
    with open(txt_filepath, "w") as txt_file:
        txt_file.writelines(filtered_lines)

    # Extract import statements and function headers
    import_lines = [
        line for line in lines if line.strip().startswith(("import", "from"))
    ]
    function_lines = extract_functions_and_args(lines)

    # Count imports and functions
    import_count = len(import_lines)
    function_count = len(function_lines)

    # Check for __main__ execution
    has_main = any(line.strip() == "if __name__ == '__main__':" for line in lines)
    main_execution = "has a main() execution" if has_main else "has no main() execution"

    # Create a meta.txt file with imports, function headers, and main execution status
    meta_filename = os.path.basename(file_path).replace(".py", "_meta.txt")
    meta_filepath = os.path.join(meta_directory, meta_filename)
    with open(meta_filepath, "w") as meta_file:
        if import_lines:
            meta_file.write(f"{import_count} Imports found:\n")
            meta_file.writelines(import_lines)
            meta_file.write("\n")
        else:
            meta_file.write("No imports found\n\n")

        if function_lines:
            meta_file.write(f"{function_count} Functions found:\n")
            meta_file.writelines(f"{func}\n" for func in function_lines)
        else:
            meta_file.write("No functions in this file\n\n")

        meta_file.write(f"{main_execution}\n")

    # Extract all comment lines
    comment_lines = [line for line in lines if line.lstrip().startswith("#")]
    comment_count = len(comment_lines)

    # Create a comments.txt file to store all extracted comments
    comments_filename = os.path.basename(file_path).replace(".py", "_comments.txt")
    comments_filepath = os.path.join(comments_directory, comments_filename)
    with open(comments_filepath, "w") as comments_file:
        if comment_count > 0:
            comments_file.write(f"{comment_count} comment lines found:\n")
            comments_file.writelines(comment_lines)
        else:
            comments_file.write("No comments in this file\n")
