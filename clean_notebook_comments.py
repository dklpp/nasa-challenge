#!/usr/bin/env python3
"""
Script to remove comments from Jupyter notebook code cells.
"""

import json
import re
from pathlib import Path

def remove_python_comments_from_lines(lines):
    """Remove Python comments from a list of code lines."""
    cleaned_lines = []
    
    for line in lines:
        # Check if line contains a string that might have # in it
        in_string = False
        in_single_quote = False
        in_double_quote = False
        in_triple_single = False
        in_triple_double = False
        escaped = False
        comment_start = -1
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if escaped:
                escaped = False
                i += 1
                continue
                
            if char == '\\':
                escaped = True
                i += 1
                continue
            
            # Check for string boundaries
            if not in_string:
                if i <= len(line) - 3 and line[i:i+3] == '"""':
                    in_triple_double = True
                    in_string = True
                    i += 2
                elif i <= len(line) - 3 and line[i:i+3] == "'''":
                    in_triple_single = True
                    in_string = True
                    i += 2
                elif char == '"':
                    in_double_quote = True
                    in_string = True
                elif char == "'":
                    in_single_quote = True
                    in_string = True
            else:
                if in_triple_double and i <= len(line) - 3 and line[i:i+3] == '"""':
                    in_triple_double = False
                    in_string = False
                    i += 2
                elif in_triple_single and i <= len(line) - 3 and line[i:i+3] == "'''":
                    in_triple_single = False
                    in_string = False
                    i += 2
                elif in_double_quote and char == '"':
                    in_double_quote = False
                    in_string = False
                elif in_single_quote and char == "'":
                    in_single_quote = False
                    in_string = False
            
            # Look for comment start
            if not in_string and char == '#':
                comment_start = i
                break
                
            i += 1
        
        if comment_start >= 0:
            cleaned_lines.append(line[:comment_start].rstrip())
        else:
            cleaned_lines.append(line)
    
    return cleaned_lines

def clean_notebook(notebook_path):
    """Clean comments from a Jupyter notebook."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        modified = False
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, str):
                    source = source.split('\n')
                
                original_source = source[:]
                cleaned_source = remove_python_comments_from_lines(source)
                
                if cleaned_source != original_source:
                    # Join back as a list of strings with newlines
                    if len(cleaned_source) > 0:
                        cell['source'] = [line + '\n' if i < len(cleaned_source) - 1 else line 
                                        for i, line in enumerate(cleaned_source)]
                    else:
                        cell['source'] = []
                    modified = True
        
        if modified:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")
        return False

def main():
    """Main function to clean all notebook files."""
    project_root = Path(__file__).parent
    notebook_dir = project_root / 'notebooks'
    
    if not notebook_dir.exists():
        print("No notebooks directory found")
        return
    
    cleaned_notebooks = []
    
    for notebook_path in notebook_dir.glob('*.ipynb'):
        print(f"Processing {notebook_path}...")
        if clean_notebook(notebook_path):
            cleaned_notebooks.append(str(notebook_path))
    
    print(f"\nCleaned {len(cleaned_notebooks)} notebooks:")
    for notebook in cleaned_notebooks:
        print(f"  - {notebook}")

if __name__ == "__main__":
    main()