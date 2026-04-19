import subprocess
import sys
import re

from ..models.core import MLFlowLLMWrapper
from .base import BaseTool
from clog import get_logger


logger = get_logger(__name__)


class TableEdaTool(BaseTool):
    name = "do_table_eda"
    description = "Performs Exploratory Data Analysis on a CSV/Excel/Parquet file."

    def run(self, file_path: str) -> str:
        """
        Performs Exploratory Data Analysis on a file.

        Args:
            file_path: Absolute path to the file to inspect (csv, xlsx, xls, parquet).

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


class WriteCodeTool(BaseTool):
    name = "write_code"
    description = "Writes Python code that accomplishes a task, optionally using provided context."

    def __init__(self, llm: MLFlowLLMWrapper):
        self.llm = llm

    def run(self, task: str, context: str | None = None) -> str:
        """
        Writes Python code to accomplish a task.

        Args:
            task: Description of the task to complete.
            context: Previous errors, EDA output, or other relevant tool call outputs.

        Returns:
            str: Generated Python code.
        """
        prompt = f"""Generate clean, executable Python code to accomplish the following task.

Task: {task}

Context: {context}

Requirements:
- Generate complete and executable Python code, do not assume or make up path files that were not given in this prompt
- If needed, use standard Python libraries and/or common packages (pandas, numpy, matplotlib, nltk, beautifulsoup4, etc.)
- If needed, read the file at the given path and perform the requested task
- Handle potential errors gracefully

Return the Python code, within python formatting, i.e., ``python <your code> ```"""

        try:
            response = self.llm.complete(prompt)
            logger.info(f"Generated code for task: {task}\nRaw Output:\n{response.text}")

            python_match = re.search(r"python\s*\n(.+)", response.text, re.DOTALL | re.IGNORECASE)
            if not python_match:
                raise ValueError("Python code was not parsed correctly: just output python code (```python <your code> ```) and nothing else.")

            return python_match.group(1)

        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return f"# Error generating code: {str(e)}"


class ExecuteCodeTool(BaseTool):
    name = "execute_code"
    description = "Executes a Python file and returns its stdout/stderr output."

    def run(self, file_path: str) -> str:
        """
        Executes Python code in the given file.

        Args:
            file_path: Absolute path to the Python file to execute.

        Returns:
            str: Output of the execution.
        """
        from pathlib import Path

        if not Path(file_path).exists():
            logger.error(f"File is not found: {file_path}")
            return f"Error: File not found: {file_path}"

        try:
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            output = []
            if result.returncode != 0:
                output.append(f"\n=== Could not produce results, exit code: {result.returncode} ===")
            else:
                output.append(f"\n=== SUCCESSFULLY RAN THE CODE FOR THE TASK. SEE OUTPUT BELOW ===")
            if result.stdout:
                output.append("STDOUT:")
                output.append(result.stdout)
            if result.stderr:
                output.append("STDERR:")
                output.append(result.stderr)

            logger.info(f"Executed code from {file_path} with error {result.stderr}")
            return "\n".join(output) if output else "Code executed successfully with no output."

        except subprocess.TimeoutExpired:
            logger.error(f"Code execution timeout for {file_path}")
            return "Error: Code execution timed out after 60 seconds"
        except Exception as e:
            logger.error(f"Error executing code from {file_path}: {str(e)}")
            return f"Error executing code: {str(e)}"
