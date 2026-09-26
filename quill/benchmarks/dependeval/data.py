"""DependEval dataset item parser."""

DATA_PATH = "/tmp/DependEval/data/python/task2_python_final.json"

def parse_dependeval_content(item):
    """Parse DependEval item into file_name → code mapping.

    DependEval format has file headers like:
      'path/to/file.py'
      :code here...
    or sometimes:
      'path/to/file.py':
      code here...
    """
    files = [f.strip("'\" ") for f in item["files"]]
    content = item["content"]
    gt = [f.strip("'\" ") for f in item["gt"]]

    # Build regex to match any of the file paths as headers
    # Headers appear as: 'path/to/file.py'\n: or 'path/to/file.py':
    file_contents = {}
    current_file = None
    current_lines = []

    for line in content.split("\n"):
        # Strip quotes and colons from the line to check for file headers
        stripped = line.strip()
        cleaned = stripped.strip("'\"").rstrip(":")

        matched_file = None
        for f in files:
            f_clean = f.strip("'\"")
            if cleaned == f_clean:
                matched_file = f_clean
                break

        # Also check if line starts with : (continuation of header)
        if not matched_file and stripped == ":" and current_file:
            continue  # skip the colon line after header

        if matched_file:
            if current_file:
                file_contents[current_file] = "\n".join(current_lines)
            current_file = matched_file
            current_lines = []
        elif current_file:
            # Skip leading colon lines
            if stripped.startswith(":") and not current_lines:
                current_lines.append(stripped[1:])
            else:
                current_lines.append(line)

    if current_file:
        file_contents[current_file] = "\n".join(current_lines)

    return files, file_contents, gt
