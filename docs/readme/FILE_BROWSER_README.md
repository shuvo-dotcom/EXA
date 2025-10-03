# File Browser and Reader Tools

This directory contains two file browser scripts that allow you to search folders, select files, and view their contents in multiple formats.

## Scripts Overview

### 1. `simple_file_browser.py` - Lightweight Version
A basic file browser with no external dependencies that handles common text-based formats.

**Supported Formats:**
- Text files (.txt, .md, .log, .py, .js, .html, .css, .sql, .yaml, .yml, .ini, .cfg, .conf)
- CSV files (.csv) - using built-in Python csv module
- JSON files (.json)

**Usage:**
```bash
python simple_file_browser.py
```

### 2. `file_browser_tool.py` - Full-Featured Version
A comprehensive file browser with support for additional formats including Excel and PDF.

**Supported Formats:**
- All formats from simple version
- Excel files (.xlsx, .xls) - requires openpyxl
- PDF files (.pdf) - requires PyPDF2
- Enhanced CSV handling with pandas
- XML files (.xml)

**Installation:**
```bash
# Install optional dependencies
pip install -r file_browser_requirements.txt

# Or install individually
pip install PyPDF2 openpyxl pandas
```

**Usage:**
```bash
python file_browser_tool.py
```

## Features

### Common Features (Both Scripts)
- **Recursive Folder Search**: Searches through all subdirectories
- **File Filtering**: Only shows supported file formats
- **File Information**: Displays file size, path, and type
- **Content Preview**: Shows formatted preview of file contents
- **External Opening**: Opens files with default system applications
- **Interactive Interface**: Easy-to-use numbered selection

### Advanced Features (Full Version Only)
- **Excel Support**: Reads multiple sheets, shows data structure
- **PDF Support**: Extracts text content from PDF files
- **Enhanced CSV**: Better data type detection and preview with pandas
- **XML Support**: Parses and displays XML structure
- **Better Error Handling**: More robust file reading with encoding detection

## How to Use

1. **Run the script**:
   ```bash
   python simple_file_browser.py
   # or
   python file_browser_tool.py
   ```

2. **Enter folder path**: 
   - Type the full path to the folder you want to search
   - Press Enter to use current directory (.)
   - Type 'q' to quit

3. **Select a file**:
   - Browse the numbered list of supported files
   - Enter the number of the file you want to view
   - Type 'q' to quit or search a new folder

4. **View content**:
   - The script will display a formatted preview of the file content
   - Different file types are handled appropriately (tables for CSV, structured for JSON, etc.)

5. **Choose action**:
   - Type 'o' to open the file externally with your default application
   - Type 'n' to start a new search
   - Type 'q' to quit

## Examples

### Searching Current Directory
```
Enter folder path to search (or 'q' to quit): .
```

### Searching Specific Folder
```
Enter folder path to search (or 'q' to quit): C:\Users\Dante\Documents\Projects
```

### Searching with Relative Path
```
Enter folder path to search (or 'q' to quit): ../data
```

## File Type Handling

### CSV Files
- Shows number of rows and columns
- Displays column headers
- Previews first few rows
- Enhanced version shows data types

### JSON Files
- Pretty-printed with proper indentation
- Shows top-level keys for dictionaries
- Displays file size information

### Excel Files (Full version only)
- Lists all sheet names
- Shows data structure for each sheet
- Previews data from each sheet

### PDF Files (Full version only)
- Extracts text content from all pages
- Shows page count and metadata
- Handles basic text extraction

### Text Files
- Shows line count and character count
- Handles different encodings (UTF-8, Latin-1)
- Displays content preview

## Error Handling

Both scripts include robust error handling for:
- Invalid folder paths
- Permission issues
- Corrupted files
- Encoding problems
- Missing dependencies (graceful degradation)

## Tips

1. **Large Files**: Content is automatically truncated for display to prevent overwhelming output
2. **Encoding**: The script tries multiple encodings if UTF-8 fails
3. **Dependencies**: The simple version works without any additional installations
4. **Performance**: For large directories, the search might take a moment - this is normal
5. **Formats**: If you need support for additional formats, modify the `supported_formats` dictionary

## Troubleshooting

### "No files found"
- Check that the folder path exists
- Verify you have read permissions
- Ensure there are files with supported extensions

### "Error reading file"
- File might be corrupted or in use by another application
- Try the simple version if the full version fails
- Check file permissions

### Missing Dependencies (Full Version)
- Install required packages: `pip install -r file_browser_requirements.txt`
- Or use the simple version which has no dependencies

## Customization

You can easily extend either script by:
1. Adding new file extensions to `supported_formats`
2. Creating new reading functions following the existing pattern
3. Modifying the display format in `display_file_content`
4. Adding new search filters or file type detection
