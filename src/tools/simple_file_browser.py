#!/usr/bin/env python3
"""
Simple File Browser and Reader
A lightweight script to search folders, select files, and handle basic file formats.
Core functionality without external dependencies.
"""

import os
import sys
import json
import csv
from pathlib import Path
import subprocess
import platform
from typing import List, Dict, Any, Optional


class SimpleFileBrowser:
    """A simple file browser and reader utility with minimal dependencies."""
    
    def __init__(self):
        self.supported_formats = {
            '.txt': self.read_text_file,
            '.csv': self.read_csv_file,
            '.json': self.read_json_file,
            '.md': self.read_text_file,
            '.log': self.read_text_file,
            '.py': self.read_text_file,
            '.js': self.read_text_file,
            '.html': self.read_text_file,
            '.css': self.read_text_file,
            '.sql': self.read_text_file,
            '.yaml': self.read_text_file,
            '.yml': self.read_text_file,
            '.ini': self.read_text_file,
            '.cfg': self.read_text_file,
            '.conf': self.read_text_file,
        }

    def get_subdirectories(self, folder_path: str) -> List[str]:
        """Get all subdirectories in a folder and one level below each subdirectory.

        Returns a flattened, sorted list (for backward compatibility with other methods)
        and stores a nested dictionary of the discovered structure on self.subdir_tree.

        Nested dict format:
        {
            "<top_level_dir_path>": {
                "<child_dir_path>": {},
                ...
            },
            ...
        }
        """
        try:
            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            if not folder.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {folder_path}")

            subdirs = []
            tree: Dict[str, Dict] = {}

            for item in folder.iterdir():
                try:
                    if item.is_dir():
                        top_path = str(item)
                        subdirs.append(top_path)
                        children_map: Dict[str, Dict] = {}
                        # add immediate child directories (one level below)
                        try:
                            for child in item.iterdir():
                                if child.is_dir():
                                    child_path = str(child)
                                    subdirs.append(child_path)
                                    children_map[child_path] = {}
                        except PermissionError:
                            # Skip children we can't access
                            pass
                        tree[top_path] = children_map
                except PermissionError:
                    # Skip top-level entries we can't access
                    continue

            # Preserve order but remove duplicates (in case of symlinks or duplicates)
            unique_subdirs = list(dict.fromkeys(subdirs))
            # Sort consistently (case-insensitive)
            unique_subdirs = sorted(unique_subdirs, key=lambda p: p.lower())

            # Store nested dict on the instance for callers that want the tree structure
            self.subdir_tree = tree

            return unique_subdirs

        except Exception as e:
            print(f"Error getting subdirectories: {e}")
            # Ensure attribute exists even on error
            self.subdir_tree = {}
            return []

    def search_folder(self, folder_path: str, file_pattern: str = "*") -> List[str]:
        """Search for files in a folder with optional pattern matching."""
        try:
            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            if not folder.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {folder_path}")
            
            # Get all files in current directory only (not recursive)
            files = []
            for file_path in folder.glob(file_pattern):
                if file_path.is_file():
                    files.append(str(file_path))
            
            return sorted(files)
            
        except Exception as e:
            print(f"Error searching folder: {e}")
            return []

    def filter_supported_files(self, files: List[str]) -> List[str]:
        """Filter files to only include supported formats."""
        supported_files = []
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            if ext in self.supported_formats:
                supported_files.append(file_path)
        return supported_files

    def display_directory_list(self, directories: List[str], current_path: str) -> None:
        """Display a numbered list of directories for user selection."""
        if not directories:
            print("No subdirectories found in this directory.")
            return
        
        print(f"\nCurrent directory: {current_path}")
        print(f"Found {len(directories)} subdirectories:")
        print("-" * 50)
        
        # Add option to go to parent directory if not at root
        parent_path = Path(current_path).parent
        if str(parent_path) != current_path:  # Not at root
            print(f"  0. .. (Parent directory: {parent_path})")
        
        for i, dir_path in enumerate(directories, 1):
            dir_obj = Path(dir_path)
            try:
                # Count files and subdirs
                file_count = len([f for f in dir_obj.iterdir() if f.is_file()])
                subdir_count = len([d for d in dir_obj.iterdir() if d.is_dir()])
                print(f"{i:3d}. {dir_obj.name}")
                print(f"     Files: {file_count}, Subdirs: {subdir_count}")
            except PermissionError:
                print(f"{i:3d}. {dir_obj.name} (Access denied)")
            print()

    def display_file_list(self, files: List[str]) -> None:
        """Display a numbered list of files for user selection."""
        if not files:
            print("No files found or no supported files in the directory.")
            return
        
        print("\nFound the following files:")
        print("-" * 50)
        for i, file_path in enumerate(files, 1):
            file_obj = Path(file_path)
            size = self.get_file_size(file_path)
            print(f"{i:3d}. {file_obj.name}")
            print(f"     Size: {size}")
            print(f"     Type: {file_obj.suffix.upper()}")
            print()

    def get_file_size(self, file_path: str) -> str:
        """Get human-readable file size."""
        try:
            size = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        except:
            return "Unknown"

    def select_directory(self, directories: List[str], current_path: str) -> Optional[str]:
        """Allow user to select a directory from the list."""
        if not directories:
            return None
        
        parent_path = Path(current_path).parent
        has_parent = str(parent_path) != current_path
        
        while True:
            try:
                if has_parent:
                    choice = input(f"\nSelect a directory (0 for parent, 1-{len(directories)}) or 'q' to quit: ").strip()
                else:
                    choice = input(f"\nSelect a directory (1-{len(directories)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                index = int(choice)
                
                if index == 0 and has_parent:
                    return str(parent_path)
                elif has_parent and 1 <= index <= len(directories):
                    return directories[index - 1]
                elif not has_parent and 1 <= index <= len(directories):
                    return directories[index - 1]
                else:
                    if has_parent:
                        print(f"Please enter a number between 0 and {len(directories)}")
                    else:
                        print(f"Please enter a number between 1 and {len(directories)}")
                    
            except ValueError:
                print("Please enter a valid number or 'q' to quit")

    def select_file(self, files: List[str]) -> Optional[str]:
        """Allow user to select a file from the list."""
        if not files:
            return None
        
        while True:
            try:
                choice = input(f"\nSelect a file (1-{len(files)}), 'b' to go back, or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                elif choice.lower() == 'b':
                    return 'back'
                
                index = int(choice) - 1
                if 0 <= index < len(files):
                    return files[index]
                else:
                    print(f"Please enter a number between 1 and {len(files)}")
                    
            except ValueError:
                print("Please enter a valid number, 'b' to go back, or 'q' to quit")

    def read_text_file(self, file_path: str) -> Dict[str, Any]:
        """Read a text-based file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return {
                'type': 'text',
                'content': content,
                'lines': len(content.splitlines()),
                'characters': len(content)
            }
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                return {
                    'type': 'text',
                    'content': content,
                    'lines': len(content.splitlines()),
                    'characters': len(content),
                    'encoding': 'latin-1'
                }
            except Exception as e:
                return {'type': 'error', 'message': f"Error reading file: {e}"}

    def read_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Read a CSV file using built-in csv module."""
        try:
            rows = []
            with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
                # Try to detect delimiter
                sample = csvfile.read(1024)
                csvfile.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.reader(csvfile, delimiter=delimiter)
                rows = list(reader)
            
            if rows:
                headers = rows[0] if rows else []
                data_rows = rows[1:] if len(rows) > 1 else []
                
                return {
                    'type': 'csv',
                    'headers': headers,
                    'rows': data_rows,
                    'total_rows': len(rows),
                    'total_columns': len(headers) if headers else 0,
                    'preview': rows[:5]  # First 5 rows
                }
            else:
                return {'type': 'csv', 'message': 'Empty CSV file'}
                
        except Exception as e:
            return {'type': 'error', 'message': f"Error reading CSV: {e}"}

    def read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Read a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            return {
                'type': 'json',
                'data': data,
                'pretty_json': json.dumps(data, indent=2),
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'size': len(str(data))
            }
        except Exception as e:
            return {'type': 'error', 'message': f"Error reading JSON: {e}"}

    def display_file_content(self, result: Dict[str, Any], file_path: str) -> None:
        """Display file content based on file type."""
        print(f"\n{'='*60}")
        print(f"FILE: {Path(file_path).name}")
        print(f"PATH: {file_path}")
        print(f"{'='*60}")
        
        if result['type'] == 'error':
            print(f"ERROR: {result['message']}")
            return
        
        if result['type'] == 'text':
            print(f"Type: Text File")
            print(f"Lines: {result['lines']}")
            print(f"Characters: {result['characters']}")
            if 'encoding' in result:
                print(f"Encoding: {result['encoding']}")
            print(f"\nContent Preview (first 1000 characters):")
            print("-" * 40)
            print(result['content'][:1000])
            if len(result['content']) > 1000:
                print("\n... (content truncated)")
        
        elif result['type'] == 'csv':
            print(f"Type: CSV File")
            if 'total_rows' in result:
                print(f"Rows: {result['total_rows']}")
                print(f"Columns: {result['total_columns']}")
                print(f"Headers: {', '.join(result['headers']) if result['headers'] else 'No headers detected'}")
                print(f"\nData Preview (first 5 rows):")
                print("-" * 40)
                for i, row in enumerate(result['preview']):
                    print(f"Row {i+1}: {', '.join(str(cell) for cell in row)}")
            else:
                print(result.get('message', 'No data'))
        
        elif result['type'] == 'json':
            print(f"Type: JSON File")
            if result['keys']:
                print(f"Top-level keys: {', '.join(result['keys'])}")
            print(f"Size: {result['size']} characters")
            print(f"\nContent Preview:")
            print("-" * 40)
            preview = result['pretty_json'][:1000]
            print(preview)
            if len(result['pretty_json']) > 1000:
                print("\n... (content truncated)")

    def open_file_externally(self, file_path: str) -> None:
        """Open file with default system application."""
        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', file_path])
            else:  # Linux
                subprocess.run(['xdg-open', file_path])
            print(f"Opened file externally: {file_path}")
        except Exception as e:
            print(f"Error opening file externally: {e}")

    def run(self):
        """Main application loop."""
        print("Simple File Browser and Reader with Directory Navigation")
        print("=" * 60)
        print("Supported formats: TXT, CSV, JSON, MD, LOG, PY, JS, HTML, CSS, SQL, YAML, INI")
        print("For PDF and Excel support, use file_browser_tool.py with additional dependencies")
        
        # Start with current directory or user input
        current_path = None
        
        while True:
            # Get initial folder path if not set
            if current_path is None:
                folder_path = input("\nEnter initial folder path to browse (or 'q' to quit): ").strip()
                
                if folder_path.lower() == 'q':
                    print("Goodbye!")
                    return
                
                if not folder_path:
                    folder_path = "."  # Current directory
                
                current_path = os.path.abspath(folder_path)
            
            print(f"\nBrowsing: {current_path}")
            
            # Get subdirectories
            subdirectories = self.get_subdirectories(current_path)
            
            # Get files in current directory
            all_files = self.search_folder(current_path)
            supported_files = self.filter_supported_files(all_files)
            
            # Show directory navigation first
            if subdirectories:
                self.display_directory_list(subdirectories, current_path)
                print(f"\nOptions:")
                print("- Select a directory number to navigate")
                print("- Type 'f' to view files in current directory")
                print("- Type 'b' to go back to parent directory")
                print("- Type 'n' to enter a new path")
                print("- Type 'q' to quit")
                
                while True:
                    choice = input("\nYour choice: ").strip().lower()
                    
                    if choice == 'q':
                        print("Goodbye!")
                        return
                    elif choice == 'n':
                        current_path = None
                        break
                    elif choice == 'b':
                        # Go back to parent directory
                        parent_path = Path(current_path).parent
                        if str(parent_path) != current_path:  # Not at root
                            current_path = str(parent_path)
                            break
                        else:
                            print("Already at root directory. Cannot go back further.")
                    elif choice == 'f':
                        # Show files in current directory
                        if supported_files:
                            self.display_file_list(supported_files)
                            selected_file = self.select_file(supported_files)
                            
                            if selected_file == 'back':
                                break  # Go back to directory view
                            elif selected_file is None:
                                print("Goodbye!")
                                return
                            else:
                                # Read and display file content
                                file_ext = Path(selected_file).suffix.lower()
                                if file_ext in self.supported_formats:
                                    result = self.supported_formats[file_ext](selected_file)
                                    self.display_file_content(result, selected_file)
                                    
                                    # Ask if user wants to open externally
                                    while True:
                                        action = input("\nOptions: (o)pen externally, (b)ack to files, (d)irectories, (q)uit: ").strip().lower()
                                        if action == 'o':
                                            self.open_file_externally(selected_file)
                                            break
                                        elif action == 'b':
                                            break  # Back to file list
                                        elif action == 'd':
                                            break  # Back to directory list
                                        elif action == 'q':
                                            print("Goodbye!")
                                            return
                                        else:
                                            print("Please enter 'o', 'b', 'd', or 'q'")
                                else:
                                    print(f"Unsupported file format: {file_ext}")
                        else:
                            print("No supported files found in current directory.")
                            input("Press Enter to continue...")
                        break
                    else:
                        # Try to parse as directory selection
                        try:
                            dir_index = int(choice)
                            parent_path = Path(current_path).parent
                            has_parent = str(parent_path) != current_path
                            
                            if dir_index == 0 and has_parent:
                                current_path = str(parent_path)
                                break
                            elif has_parent and 1 <= dir_index <= len(subdirectories):
                                current_path = subdirectories[dir_index - 1]
                                break
                            elif not has_parent and 1 <= dir_index <= len(subdirectories):
                                current_path = subdirectories[dir_index - 1]
                                break
                            else:
                                if has_parent:
                                    print(f"Please enter a number between 0 and {len(subdirectories)}, or 'f', 'b', 'n', 'q'")
                                else:
                                    print(f"Please enter a number between 1 and {len(subdirectories)}, or 'f', 'b', 'n', 'q'")
                        except ValueError:
                            print("Please enter a valid option: directory number, 'f', 'b', 'n', or 'q'")
            
            else:
                # No subdirectories, show files directly
                if supported_files:
                    print("No subdirectories found. Showing files:")
                    self.display_file_list(supported_files)
                    selected_file = self.select_file(supported_files)
                    
                    if selected_file == 'back':
                        parent_path = Path(current_path).parent
                        if str(parent_path) != current_path:  # Not at root
                            current_path = str(parent_path)
                        else:
                            print("Already at root directory.")
                            input("Press Enter to continue...")
                        continue
                    elif selected_file is None:
                        print("Goodbye!")
                        return
                    else:
                        # Read and display file content
                        file_ext = Path(selected_file).suffix.lower()
                        if file_ext in self.supported_formats:
                            result = self.supported_formats[file_ext](selected_file)
                            self.display_file_content(result, selected_file)
                            
                            # Ask if user wants to open externally
                            while True:
                                action = input("\nOptions: (o)pen externally, (b)ack, (n)ew path, (q)uit: ").strip().lower()
                                if action == 'o':
                                    self.open_file_externally(selected_file)
                                    break
                                elif action == 'b':
                                    break
                                elif action == 'n':
                                    current_path = None
                                    break
                                elif action == 'q':
                                    print("Goodbye!")
                                    return
                                else:
                                    print("Please enter 'o', 'b', 'n', or 'q'")
                        else:
                            print(f"Unsupported file format: {file_ext}")
                else:
                    print("No supported files found in this directory.")
                    while True:
                        choice = input("Options: (b)ack, (n)ew path, (q)uit: ").strip().lower()
                        if choice == 'b':
                            parent_path = Path(current_path).parent
                            if str(parent_path) != current_path:  # Not at root
                                current_path = str(parent_path)
                            else:
                                print("Already at root directory.")
                                continue
                            break
                        elif choice == 'n':
                            current_path = None
                            break
                        elif choice == 'q':
                            print("Goodbye!")
                            return
                        else:
                            print("Please enter 'b', 'n', or 'q'")


def main():
    """Main entry point."""
    browser = SimpleFileBrowser()
    
    try:
        browser.run()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
