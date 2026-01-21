import subprocess
import sys

from ..models.core import llm
from clog import get_logger


logger = get_logger(__name__)


def do_table_eda(file_path: str) -> str:
    """
    This tool performs Exploratory Data Analysis on a file. Namely, it:
      - checks what columns are present in the table
      - the data types and columns present in the table
      - it provides summary statistics of the columns

    Args:
        file_path (str): absolute path to the file to inspect

    Returns:
        str: Generated EDA analysis.
    """
    import pandas as pd
    from pathlib import Path

    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_ext}. Supported formats: csv, xlsx, xls, parquet"

        output = []
        output.append(f"=== Exploratory Data Analysis for {Path(file_path).name} ===")

        output.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        output.append("--- Column Information ---")
        output.append(f"Columns: {list(df.columns)}")

        output.append("--- Data Types ---")
        output.append(str(df.dtypes))

        output.append("--- Summary Statistics ---")
        nans = df.isna().sum()
        counts = df.count()
        for col in df.columns:
            ser = df[col]
            nans_v = int(nans[col])
            count_v = int(counts[col])
            mean_v = ser.mean() if pd.api.types.is_numeric_dtype(ser) else None
            try:
                min_v = ser.min()
                max_v = ser.max()
            except Exception:
                min_v = None
                max_v = None
            output.append(f"Column: {col}")
            output.append(f"  nans: {nans_v} | count: {count_v}")
            output.append(f"  mean: {mean_v} | min: {min_v} | max: {max_v}")

        output.append("--- Missing Values ---")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            output.append(str(missing[missing > 0]))
        else:
            output.append("No missing values found")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error performing EDA on {file_path}: {str(e)}")
        return f"Error performing EDA: {str(e)}"

def write_code(task: str, context: str | None = None, file_path: str | None = None) -> str:
    """
    This tool writes code that accomplishes a task given some file resource.

    It receives the description of the task, context as the previous relevant tool calls outputs and a filepath to work with.
    Then, it generates code that:
        - performs the task, using the necessary resources (the given file path) and the occurred contex (e.g., previous errors while executing, Exploratory Analysis, other tool calls)
        - uses python standard libraries and/or may use common packages (e.g., pandas, numpy, matplotlib, nltk, beautifulsoup4)

    Args:
        task (str): Description of the task to complete.
        context (str): all the previous errors while executing, exploratory analysis on relevant file, other tool calls outputs
        file_path (str): File path to the file relevant for the task.

    Returns:
        str: Generated Python code.
    """
    prompt = f"""Generate clean, executable Python code to accomplish the following task.

Task: {task}

Context: {context}

File: {file_path if file_path else "None"}

Requirements:
- Generate complete, executable Python code
- Use standard Python libraries and/or common packages (pandas, numpy, matplotlib, nltk, beautifulsoup4, etc.)
- Read the file at the given path and perform the requested task
- Handle potential errors gracefully

Return ONLY the Python code, without any additional explanation or markdown formatting."""

    try:
        response = llm.complete(prompt)

        code = response.text.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        logger.info(f"Generated code for task: {task[:50]}...")
        return code

    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        return f"# Error generating code: {str(e)}"

def execute_code(file_path: str) -> str:
    """
    This tool executes python code in the file.

    Args:
        file_path (str): filepath to the python code to be executed

    Returns:
        str: output of the execution.
    """
    from pathlib import Path

    if not Path(file_path).exists():
        logger.error(f"File is not found: {file_path}")
        return f"Error: File not found: {file_path}"

    if not file_path.endswith('.py'):
        logger.error(f"File must be a Python file (.py): {file_path}")
        return f"Error: File must be a Python file (.py): {file_path}"

    try:
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = []
        if result.stdout:
            output.append("=== STDOUT ===")
            output.append(result.stdout)

        if result.stderr:
            output.append("\n=== STDERR ===")
            output.append(result.stderr)

        if result.returncode != 0:
            output.append(f"\n=== Exit Code: {result.returncode} ===")

        logger.info(f"Executed code from {file_path} with exit code {result.returncode}")
        return "\n".join(output) if output else "Code executed successfully with no output."

    except subprocess.TimeoutExpired:
        logger.error(f"Code execution timeout for {file_path}")
        return "Error: Code execution timed out after 60 seconds"
    except Exception as e:
        logger.error(f"Error executing code from {file_path}: {str(e)}")
        return f"Error executing code: {str(e)}"

