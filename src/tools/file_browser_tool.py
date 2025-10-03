#!/usr/bin/env python3
"""
File Browser and Reader Tool
A comprehensive script to search folders, select files, and handle multiple file formats.
Supports: TXT, CSV, Excel (XLS/XLSX), PDF, JSON, XML, and more.
"""

import os
import sys
import pandas as pd
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import platform
from typing import List, Dict, Any, Optional

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


class FileBrowser:
    """A comprehensive file browser and reader utility."""
    
    def __init__(self):
        self.supported_formats = {
            '.txt': self.read_text_file,
            '.csv': self.read_csv_file,
            '.json': self.read_json_file,
            '.xml': self.read_xml_file,
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
        
        if EXCEL_AVAILABLE:
            self.supported_formats.update({
                '.xlsx': self.read_excel_file,
                '.xls': self.read_excel_file,
            })
        
        if PDF_AVAILABLE:
            self.supported_formats['.pdf'] = self.read_pdf_file

    def search_folder(self, folder_path: str, file_pattern: str = "*") -> List[str]:
        """
        Search for files in a folder with optional pattern matching.
        
        Args:
            folder_path: Path to search in
            file_pattern: File pattern to match (default: "*" for all files)
            
        Returns:
            List of file paths found
        """
        try:
            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            if not folder.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {folder_path}")
            
            # Get all files recursively
            files = []
            for file_path in folder.rglob(file_pattern):
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
            print(f"     Path: {file_obj.parent}")
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

    def select_file(self, files: List[str]) -> Optional[str]:
        """Allow user to select a file from the list."""
        if not files:
            return None
        
        while True:
            try:
                choice = input(f"\nSelect a file (1-{len(files)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                index = int(choice) - 1
                if 0 <= index < len(files):
                    return files[index]
                else:
                    print(f"Please enter a number between 1 and {len(files)}")
                    
            except ValueError:
                print("Please enter a valid number or 'q' to quit")

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
        """Read a CSV file."""
        try:
            df = pd.read_csv(file_path)
            return {
                'type': 'csv',
                'data': df,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'preview': df.head().to_string()
            }
        except Exception as e:
            return {'type': 'error', 'message': f"Error reading CSV: {e}"}

    def read_excel_file(self, file_path: str) -> Dict[str, Any]:
        """Read an Excel file."""
        try:
            # Get all sheet names
            xl_file = pd.ExcelFile(file_path)
            sheets_data = {}
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = {
                    'data': df,
                    'shape': df.shape,
                    'columns': list(df.columns)
                }
            
            return {
                'type': 'excel',
                'sheets': list(xl_file.sheet_names),
                'data': sheets_data,
                'total_sheets': len(xl_file.sheet_names)
            }
        except Exception as e:
            return {'type': 'error', 'message': f"Error reading Excel: {e}"}

    def read_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Read a PDF file."""
        if not PDF_AVAILABLE:
            return {'type': 'error', 'message': 'PyPDF2 not installed. Install with: pip install PyPDF2'}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page.extract_text()
                
                return {
                    'type': 'pdf',
                    'content': text_content,
                    'pages': len(pdf_reader.pages),
                    'metadata': pdf_reader.metadata
                }
        except Exception as e:
            return {'type': 'error', 'message': f"Error reading PDF: {e}"}

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

    def read_xml_file(self, file_path: str) -> Dict[str, Any]:
        """Read an XML file."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Convert to string for display
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return {
                'type': 'xml',
                'content': content,
                'root_tag': root.tag,
                'root_attrib': root.attrib,
                'element_count': len(list(root.iter()))
            }
        except Exception as e:
            return {'type': 'error', 'message': f"Error reading XML: {e}"}

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
            print(f"Shape: {result['shape']} (rows, columns)")
            print(f"Columns: {', '.join(result['columns'])}")
            print(f"\nData Preview:")
            print("-" * 40)
            print(result['preview'])
        
        elif result['type'] == 'excel':
            print(f"Type: Excel File")
            print(f"Sheets: {result['total_sheets']}")
            print(f"Sheet Names: {', '.join(result['sheets'])}")
            
            for sheet_name, sheet_data in result['data'].items():
                print(f"\n--- Sheet: {sheet_name} ---")
                print(f"Shape: {sheet_data['shape']}")
                print(f"Columns: {', '.join(sheet_data['columns'])}")
                print(sheet_data['data'].head().to_string())
        
        elif result['type'] == 'pdf':
            print(f"Type: PDF File")
            print(f"Pages: {result['pages']}")
            if result['metadata']:
                print(f"Metadata: {result['metadata']}")
            print(f"\nContent Preview (first 1000 characters):")
            print("-" * 40)
            print(result['content'][:1000])
            if len(result['content']) > 1000:
                print("\n... (content truncated)")
        
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
        
        elif result['type'] == 'xml':
            print(f"Type: XML File")
            print(f"Root Tag: {result['root_tag']}")
            print(f"Root Attributes: {result['root_attrib']}")
            print(f"Total Elements: {result['element_count']}")
            print(f"\nContent Preview (first 1000 characters):")
            print("-" * 40)
            print(result['content'][:1000])
            if len(result['content']) > 1000:
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
        print("File Browser and Reader Tool")
        print("=" * 40)
        
        # Get folder path
        while True:
            folder_path = input("\nEnter folder path to search (or 'q' to quit): ").strip()
            
            if folder_path.lower() == 'q':
                print("Goodbye!")
                return
            
            if not folder_path:
                folder_path = "."  # Current directory
            
            # Search for files
            print(f"\nSearching in: {os.path.abspath(folder_path)}")
            all_files = self.search_folder(folder_path)
            
            if not all_files:
                print("No files found in the specified directory.")
                continue
            
            # Filter to supported files
            supported_files = self.filter_supported_files(all_files)
            
            if not supported_files:
                print(f"No supported files found. Found {len(all_files)} total files.")
                print(f"Supported formats: {', '.join(self.supported_formats.keys())}")
                continue
            
            # Display files and let user select
            self.display_file_list(supported_files)
            selected_file = self.select_file(supported_files)
            
            if selected_file is None:
                continue
            
            # Read and display file content
            file_ext = Path(selected_file).suffix.lower()
            if file_ext in self.supported_formats:
                result = self.supported_formats[file_ext](selected_file)
                self.display_file_content(result, selected_file)
                
                # Ask if user wants to open externally
                while True:
                    action = input("\nOptions: (o)pen externally, (n)ew search, (q)uit: ").strip().lower()
                    if action == 'o':
                        self.open_file_externally(selected_file)
                        break
                    elif action == 'n':
                        break
                    elif action == 'q':
                        print("Goodbye!")
                        return
                    else:
                        print("Please enter 'o', 'n', or 'q'")
            else:
                print(f"Unsupported file format: {file_ext}")


def main():
    """Main entry point."""
    # Check for required dependencies
    missing_deps = []
    
    if not PDF_AVAILABLE:
        missing_deps.append("PyPDF2 (for PDF support)")
    
    if not EXCEL_AVAILABLE:
        missing_deps.append("openpyxl (for Excel support)")
    
    if missing_deps:
        print("Optional dependencies missing:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install PyPDF2 openpyxl")
        print("The tool will work without these, but with limited format support.\n")
    
    browser = FileBrowser()
    
    try:
        browser.run()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
