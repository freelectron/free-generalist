import re
import subprocess
import sys

from ..tools.data_model import ContentResource
from ..models.core import llm
from clog import get_logger


logger = get_logger(__name__)


def write_python_eda(resources:list[ContentResource]) -> str:
    """ 
    Exploratory Data Analysis. This function loads the files (also called, resources) and inspects them by:
        1. Analysing what the files are and how to open them just using commond/standard python packages
        2. Briefly inspecting the contents of the resources (if needed) and see what type of data and format they hold;
        3. Checking the names of columns, variables, packages or anything that might be needed.

    If you are given some code (python) and the task is to execute, see what python packages are used.

    Args:
        resources (list[ContentResource]): List of resources (files or links) to analyse.

    Returns:
        str: Generated EDA description or code that inspects the provided resources.
    """
    prompt = f"""
    You are programmer that writes well commented (you print comments to STDOUT) and simple python code. 
    Given the following RESOURCES: 
    {resources}
    
    For each resources that is a of ** .CSV , .PARQUET , .XLSX ** , write python code that: 
        1. Briefly inspect the contents of the resources (if needed) and see what type of data and format they hold;
        2. if the provided file are csv, parquet or excel: check the names of columns and describe the table.
        
    **IMPORTANT**: never try to install any packages, assume that they are available. 
    **IMPORTANT**: do not make any assumptions about the data, explore columns and datatypes available. 
    **IMPORTANT**: do not change or autocorrect the links in the resources      
    **IMPORTANT**: instead of in-line code comments,  use `print` function to print those comments before the executing the command 
        e.g., print("Loaded file from <folder_name>"), print("Found these columns in the csv file.")   

    -------
    Example
    Resources: 
        ContentResource(
            provided_by='user', 
            content='file provided with the main task',
            link='/Users/anton.kerel/projects/freelectron/free-generalist/evaluation/gaia/2023/validation/3r938r93-f3f-222.xlsx', 
            metadata={{'note': 'the file is already in the list of available resources'}}
        )
    You should start by opening the file : 
        print("Loading '/Users/anton.kerel/projects/freelectron/free-generalist/evaluation/gaia/2023/validation/3r938r93-f3f-222.xlsx' ")
         ..... the rest of the code ..... 
                       
    DO NOT give any additional text apart from python code. If resources ARE NOT .CSV , .PARQUET , .XLSX output: ```python print()```
    Start your python code now (always include python formating directive ```python ```).
    """
    result = llm.complete(prompt)
    
    return result.text

def write_python_task(task: str, eda_results: str, resources: list[ContentResource]):
    """
    This function writes code that accomplishes a task given some (file or web) resources.
    It receives the description of the task, some meta-information about the resources it needs to use and their contents,
    and links to those resources. It then generates that code that:
        1. uses python standard libraries and some common packages like pandas, numpy, matplotlib, nltk, beautifulsoup4.
        2. write code that reads the necessary info from only the necessary resources (using their url/uri links)
        3. accomplishes the ask of the task

    Args:
        task (str): Description of the task to complete.
        eda_results (str): Meta-information describing the contents and schema
            of the resources.
        resources (list[ContentResource]): List of resources (files or links)
            that can be used to accomplish the task.

    Returns:
        str: Generated Python code as plain text.
    """
    prompt = f"""    
    Task:
    {task}

    Resources:
    {resources}

    Exploratory Data Analysis results from resources:
    {eda_results}

    Write python code that:
    1. uses python standard libraries and COMMONLY USED ONLY packages 
    2. executes the task 
     
    **IMPORTANT**: never try to install any packages. Assume the most common are available.
    **IMPORTANT**: do not change or autocorrect the resource links, e.g., <>/freelectron/<> should stay as 'freelectron'     
    
    -------
    Example
    Task: get the sum of the columns representing directions  
    Resources: 
        ContentResource(
            provided_by='user', 
            content='file provided with the main task',
            link='/Users/anton.kerel/projects/freelectron/free-generalist/evaluation/gaia/2023/validation/3r938r93-f3f-222.xlsx', 
            metadata={{'note': 'the file is already in the list of available resources'}}
        )
    Exploratory Data Analysis results from resources: 
        "Loaded '/Users/anton.kerel/projects/freelectron/free-generalist/evaluation/gaia/2023/validation/3r938r93-f3f-222.xlsx' "
         'Available columns: "up", "down", "sales" '
     You should start by:
        import pandas as pd 
        df = pd.read_excel('/Users/anton.kerel/projects/freelectron/free-generalist/evaluation/gaia/2023/validation/3r938r93-f3f-222.xlsx')
        up_sum = df["up"].sum()
        down_sum = df["down"].sum()
        print("total:", up_sum + down_sum)
        
    Provide in code comments on what you are doing, DO NOT give any additional text apart from python code and comments.
    Start your python code now (always include python formating directive ```python ```).
    """
    result = llm.complete(prompt)

    return result.text

def run_code(text_with_code: str) -> str:
    """
    Execute the first python code math in `text_with_code` in a separate process.
    The child process will have the same packages available as the parent one.
    """
    code_match = re.search(r"```python\n(.*?)```", text_with_code, re.DOTALL)
    if not code_match:
        return "--- No python code block found, try to rerunning the step ---"
    extracted_code = code_match.group(1)

    # TODO: to double check,
    process = subprocess.run(
        [sys.executable, "-c", extracted_code],
        capture_output=True,
        text=True,
        check=False # Don't raise an exception for non-zero exit codes
    )

    output_string = ""
    if process.stdout:
        output_string += f"--- STDOUT ---\n{process.stdout}\n"
    if process.stderr:
        output_string += f"--- STDERR ---\n{process.stderr}\n"
    if not output_string.strip():
        output_string = "--- No output produced ---"

    return output_string
