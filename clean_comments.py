
"""
Script to remove all comments from code files in the project.
Handles JavaScript, TypeScript, Python, CSS, and JSON files.
"""

import os
import re
import json
from pathlib import Path

def remove_js_ts_comments(content):
    """Remove // and /* */ comments from JavaScript/TypeScript files."""

    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:

        in_string = False
        in_single_quote = False
        in_double_quote = False
        in_template = False
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
            

            if not in_string:
                if char == '"' and not in_single_quote:
                    in_double_quote = True
                    in_string = True
                elif char == "'" and not in_double_quote:
                    in_single_quote = True
                    in_string = True
                elif char == '`':
                    in_template = True
                    in_string = True
            else:
                if in_double_quote and char == '"':
                    in_double_quote = False
                    in_string = False
                elif in_single_quote and char == "'":
                    in_single_quote = False
                    in_string = False
                elif in_template and char == '`':
                    in_template = False
                    in_string = False
            

            if not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                comment_start = i
                break
                
            i += 1
        
        if comment_start >= 0:
            cleaned_lines.append(line[:comment_start].rstrip())
        else:
            cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    


    result = []
    i = 0
    while i < len(content):
        if i < len(content) - 1 and content[i:i+2] == '/*':

            j = i + 2
            while j < len(content) - 1:
                if content[j:j+2] == '*/':
                    i = j + 2
                    break
                j += 1
            else:

                break
        else:
            result.append(content[i])
            i += 1
    
    return ''.join(result)

def remove_python_comments(content):
    """Remove # comments from Python files."""
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:

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
            

            if not in_string and char == '#':
                comment_start = i
                break
                
            i += 1
        
        if comment_start >= 0:
            cleaned_lines.append(line[:comment_start].rstrip())
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def remove_css_comments(content):
    """Remove /* */ comments from CSS files."""
    result = []
    i = 0
    while i < len(content):
        if i < len(content) - 1 and content[i:i+2] == '/*':

            j = i + 2
            while j < len(content) - 1:
                if content[j:j+2] == '*/':
                    i = j + 2
                    break
                j += 1
            else:

                break
        else:
            result.append(content[i])
            i += 1
    
    return ''.join(result)

def clean_file(file_path):
    """Clean comments from a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        if file_path.suffix in ['.js', '.jsx', '.ts', '.tsx']:
            content = remove_js_ts_comments(content)
        elif file_path.suffix == '.py':
            content = remove_python_comments(content)
        elif file_path.suffix == '.css':
            content = remove_css_comments(content)
        elif file_path.suffix == '.json':


            return False
        else:
            return False
        

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to clean all files in the project."""
    project_root = Path(__file__).parent
    

    extensions = {'.js', '.jsx', '.ts', '.tsx', '.py', '.css'}
    

    skip_dirs = {'node_modules', '.git', '__pycache__', '.next', 'dist', 'build'}
    
    cleaned_files = []
    
    for file_path in project_root.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:

            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            print(f"Processing {file_path}...")
            if clean_file(file_path):
                cleaned_files.append(str(file_path))
    
    print(f"\nCleaned {len(cleaned_files)} files:")
    for file in cleaned_files:
        print(f"  - {file}")

if __name__ == "__main__":
    main()