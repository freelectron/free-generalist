import fnmatch
from pathlib import Path

from clog import get_logger
from . import BaseTool

GREP_EXCLUDE_PATTERNS = ["*.pyc", "*.log", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.pdf"]
GREP_EXCLUDE_DIRS = ["__pycache__", ".git", "node_modules", ".venv", ".idea"]

logger = get_logger(__name__)


def _is_excluded(p: Path) -> bool:
    if any(part in GREP_EXCLUDE_DIRS for part in p.parts):
        return True
    return any(fnmatch.fnmatch(p.name, pat) for pat in GREP_EXCLUDE_PATTERNS)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Reads and returns the contents of a file."

    def run(self, file_path: str) -> str:
        """
        Reads and returns the contents of a file.

        Args:
            file_path: Absolute path to the file to read.

        Returns:
            str: Contents of the file.
        """
        path = Path(file_path).expanduser()
        if not path.exists():
            return f"Error: File not found: {file_path}"
        if not path.is_file():
            return f"Error: Path is not a file: {file_path}"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return f"Error reading file: {e}"


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "Lists files in a directory, optionally recursively."

    def run(self, directory: str, recursive: bool = False) -> str:
        """
        Lists files in a directory, optionally recursively.

        Args:
            directory: Absolute path to the directory to list.
            recursive: If True, list files in all subdirectories as well.

        Returns:
            str: Newline-separated list of file paths.
        """
        path = Path(directory).expanduser()
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        if not path.is_dir():
            return f"Error: Path is not a directory: {directory}"
        try:
            if recursive:
                files = [str(p) for p in path.rglob("*") if p.is_file() and not _is_excluded(p)]
            else:
                files = [str(p) for p in path.iterdir() if p.is_file() and not _is_excluded(p)]
            if not files:
                return "No files found."
            return "\n".join(files)
        except Exception as e:
            logger.error(f"Error listing {directory}: {e}")
            return f"Error listing files: {e}"


class FindFileTool(BaseTool):
    name = "find_file"
    description = "Finds files matching a glob pattern within a directory."

    def run(self, directory: str, pattern: str) -> str:
        """
        Finds files matching a glob pattern within a directory.

        Args:
            directory: Absolute path to the directory to search.
            pattern: Glob pattern to match filenames against (e.g., '*.py', 'config*.json').

        Returns:
            str: Newline-separated list of matching file paths.
        """
        path = Path(directory).expanduser()
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        if not path.is_dir():
            return f"Error: Path is not a directory: {directory}"
        try:
            matches = [str(p) for p in path.rglob(pattern) if p.is_file() and not _is_excluded(p)]
            if not matches:
                return f"No files matching '{pattern}' found in {directory}."
            return "\n".join(matches)
        except Exception as e:
            logger.error(f"Error searching {directory} for {pattern}: {e}")
            return f"Error finding files: {e}"


class GrepFilesTool(BaseTool):
    name = "grep_files"
    description = "Finds files whose contents contain all of the given substrings."

    def run(self, directory: str, substrings: list[str], recursive: bool = True) -> str:
        """
        Finds files whose contents contain all of the given substrings.

        Args:
            directory: Absolute path to the directory to search.
            substrings: One or more strings that must all be present in a file.
            recursive: If True, search files in all subdirectories as well.

        Returns:
            str: Newline-separated list of matching file paths with matching lines.
        """
        path = Path(directory).expanduser()
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        if not path.is_dir():
            return f"Error: Path is not a directory: {directory}"
        try:
            files = [p for p in (path.rglob("*") if recursive else path.iterdir()) if p.is_file() and not _is_excluded(p)]
            results = []
            for file in files:
                try:
                    text = file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if all(s in text for s in substrings):
                    matching_lines = [
                        f"  {i+1}: {line}"
                        for i, line in enumerate(text.splitlines())
                        if any(s in line for s in substrings)
                    ]
                    results.append(str(file))
                    results.extend(matching_lines)
            if not results:
                return f"No files containing {substrings} found in {directory}."
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error grepping {directory} for {substrings}: {e}")
            return f"Error searching files: {e}"


class ReplaceFileContentsTool(BaseTool):
    name = "replace_file_contents"
    description = "Overwrites the contents of destination_file with the contents of source_file."

    def run(self, source_file: str, destination_file: str) -> str:
        """
        Overwrites the contents of destination_file with the contents of source_file.

        Args:
            source_file: Absolute path to the file to copy from.
            destination_file: Absolute path to the file to overwrite.

        Returns:
            str: Confirmation message or error.
        """
        src = Path(source_file).expanduser()
        dst = Path(destination_file).expanduser()
        if not src.exists():
            return f"Error: Source file not found: {source_file}"
        if not src.is_file():
            return f"Error: Source path is not a file: {source_file}"
        if not dst.exists():
            return f"Error: Destination file not found: {destination_file}"
        if not dst.is_file():
            return f"Error: Destination path is not a file: {destination_file}"
        try:
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return f"Successfully copied contents from {source_file} to {destination_file}."
        except Exception as e:
            logger.error(f"Error replacing {destination_file} with {source_file}: {e}")
            return f"Error replacing file contents: {e}"
