
import re
import subprocess
import sys

from ..tools.data_model import ContentResource
from ..models.core import llm
from clog import get_logger


logger = get_logger(__name__)


def write_python_eda(resources:list[ContentResource]) -> str: 
    """ 
    Exploratory Data Analysis
    """
    prompt = f"""
    You MAY BE given a list of resources that contains local binary or text files.
    Your task is to describe what each local resource contains.

    Write python code that would first try to load each resource and then inspect its contents.
    You may use popular or standard python libraries. 

    Your code should output meta information for each file.
    For example, if you are given and .csv or .xls file, output the columns that it contains and provide basical schema and datatypes. 

    Resources:
    {resources}    
    
    **IMPORTANT**: if no local files are given in the Resources section, write a python command that prints "No resources to analyse"  
    Start your python code now. 
    """
    result = llm.complete(prompt)
    
    return result.text

def write_python_task(task: str, eda_results: str, resources: list[ContentResource]):

    prompt = f"""
    Task:
    {task}

    Resources to use: 
    {resources}

    Information about resources:
    {eda_results}

    Given the resources and their specifications, write python code that completes the task.
    """
    result = llm.complete(prompt)

    return result.text

def run_code(text_with_code: str) -> str:
    """
    """
    code_match = re.search(r"```python\n(.*?)```", text_with_code, re.DOTALL)
    extracted_code = code_match.group(1)

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
