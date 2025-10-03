# -*- coding: utf-8 -*-
"""
File CRUD Operations Demonstration Script

This script demonstrates how to use the FileCRUD class for various file operations.

@author: AI Architecture System
"""

import os
import sys
from pathlib import Path

# Add the tools directory to the path
sys.path.append(os.path.dirname(__file__))
from src.ai.file_operations.file_CRUD import FileCRUD

def demonstrate_crud_operations():
    """
    Demonstrate all CRUD operations with various file types
    """
    
    # Initialize the CRUD handler
    crud = FileCRUD()
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("FILE CRUD OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    # 1. CREATE OPERATIONS
    print("\n1. CREATE OPERATIONS")
    print("-" * 30)
    
    # Create a simple text file
    print("\n1.1 Creating a simple text file...")
    result = crud.execute_crud_operation(
        action="create",
        input_data="This is a sample text file created by the CRUD system.",
        output_location="output/sample_text.txt",
        output_structure="txt",
        output_extension="txt"
    )
    print(f"Result: {result['status']} - {result['message']}")
    
    # Create a JSON file with AI assistance
    print("\n1.2 Creating a JSON configuration file with AI...")
    result = crud.execute_crud_operation(
        action="create",
        output_location="output/app_config.json",
        output_structure="json",
        output_extension="json",
        user_request="Create a configuration file for a Python web application with database settings, logging configuration, and API endpoints",
        ai_context="You are a helpful configuration file generator for Python applications."
    )
    print(f"Result: {result['status']} - {result['message']}")
    
    # Create a CSV file with sample data
    print("\n1.3 Creating a CSV file with sample data...")
    sample_csv_data = """Name,Age,City,Department
John Doe,30,New York,Engineering
Jane Smith,25,Los Angeles,Marketing
Bob Johnson,35,Chicago,Sales
Alice Brown,28,Houston,HR"""
    
    result = crud.execute_crud_operation(
        action="create",
        input_data=sample_csv_data,
        output_location="output/employees.csv",
        output_structure="csv",
        output_extension="csv"
    )
    print(f"Result: {result['status']} - {result['message']}")
    
    # 2. READ OPERATIONS
    print("\n\n2. READ OPERATIONS")
    print("-" * 30)
    
    # Read the text file we just created
    print("\n2.1 Reading the text file...")
    result = crud.execute_crud_operation(
        action="read",
        input_location="output/sample_text.txt"
    )
    print(f"Result: {result['status']} - {result['message']}")
    if result['status'] == 'success':
        print(f"Content: {result['content']}")
    
    # Read and analyze the CSV file
    print("\n2.2 Reading and analyzing the CSV file...")
    result = crud.execute_crud_operation(
        action="read",
        input_location="output/employees.csv",
        user_request="Analyze this employee data and provide insights about the demographics and department distribution",
        ai_context="You are a data analyst specializing in HR analytics."
    )
    print(f"Result: {result['status']} - {result['message']}")
    if result['status'] == 'success' and result.get('ai_analysis'):
        print(f"AI Analysis: {result['ai_analysis'][:200]}...")
    
    # Read the README file if it exists
    print("\n2.3 Reading project README...")
    if os.path.exists("README.md"):
        result = crud.execute_crud_operation(
            action="read",
            input_location="README.md",
            user_request="Summarize the main purpose and key features of this project",
            ai_context="You are a technical documentation analyst."
        )
        print(f"Result: {result['status']} - {result['message']}")
        if result['status'] == 'success' and result.get('ai_analysis'):
            print(f"Project Summary: {result['ai_analysis'][:300]}...")
    
    # 3. UPDATE OPERATIONS
    print("\n\n3. UPDATE OPERATIONS")
    print("-" * 30)
    
    # Update the text file
    print("\n3.1 Updating the text file...")
    result = crud.execute_crud_operation(
        action="update",
        input_data="This line was added during the update operation.",
        input_location="output/sample_text.txt",
        output_location="output/sample_text_updated.txt",
        user_request="Add a timestamp and version information to this file",
        ai_context="You are a file management assistant."
    )
    print(f"Result: {result['status']} - {result['message']}")
    
    # Update CSV with new employee data
    print("\n3.2 Updating employee CSV with new data...")
    result = crud.execute_crud_operation(
        action="update",
        input_location="output/employees.csv",
        output_location="output/employees_updated.csv",
        user_request="Add 3 new employees: Sarah Wilson (32, Boston, Engineering), Mike Davis (29, Seattle, Marketing), Lisa Garcia (31, Denver, Sales)",
        ai_context="You are a data management assistant for HR systems."
    )
    print(f"Result: {result['status']} - {result['message']}")
    
    # 4. SAFE DELETE DEMONSTRATION (without actual deletion)
    print("\n\n4. DELETE OPERATIONS (DEMONSTRATION)")
    print("-" * 40)
    
    # Show what happens when deletion is not explicitly requested
    print("\n4.1 Attempting deletion without explicit request...")
    result = crud.execute_crud_operation(
        action="delete",
        input_location="output/sample_text.txt",
        user_request="I want to remove this file"  # Note: doesn't contain "delete"
    )
    print(f"Result: {result['status']} - {result['message']}")
    
    # Show what happens with explicit deletion request
    print("\n4.2 Deletion with explicit request (commented out for safety)...")
    print("# To actually delete a file, use:")
    print("# result = crud.execute_crud_operation(")
    print("#     action='delete',")
    print("#     input_location='output/sample_text.txt',")
    print("#     user_request='Please delete this file permanently'")
    print("# )")
    
    # 5. ERROR HANDLING DEMONSTRATION
    print("\n\n5. ERROR HANDLING DEMONSTRATION")
    print("-" * 35)
    
    # Try to read a non-existent file
    print("\n5.1 Attempting to read non-existent file...")
    result = crud.execute_crud_operation(
        action="read",
        input_location="nonexistent_file.txt"
    )
    print(f"Result: {result['status']} - {result['message']}")
    
    # Try an invalid action
    print("\n5.2 Attempting invalid action...")
    result = crud.execute_crud_operation(
        action="invalid_action",
        input_location="output/sample_text.txt"
    )
    print(f"Result: {result['status']} - {result['message']}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    # List created files
    print("\nFiles created during demonstration:")
    output_files = list(output_dir.glob("*"))
    for file_path in output_files:
        print(f"  - {file_path}")


def interactive_crud_demo():
    """
    Interactive demonstration allowing user to test CRUD operations
    """
    crud = FileCRUD()
    
    print("\n" + "=" * 60)
    print("INTERACTIVE CRUD DEMONSTRATION")
    print("=" * 60)
    print("Enter 'quit' to exit at any time")
    
    while True:
        print("\n" + "-" * 40)
        print("Available operations:")
        print("1. Create file")
        print("2. Read file")
        print("3. Update file")
        print("4. Delete file")
        print("5. Quit")
        
        choice = input("\nSelect operation (1-5): ").strip()
        
        if choice == '5' or choice.lower() == 'quit':
            break
        
        try:
            if choice == '1':
                # Create operation
                output_location = input("Output file path: ").strip()
                output_structure = input("Output format (txt/json/csv/xlsx): ").strip() or "txt"
                user_request = input("AI request (optional): ").strip()
                input_data = input("Initial data (optional): ").strip()
                
                result = crud.execute_crud_operation(
                    action="create",
                    input_data=input_data if input_data else None,
                    output_location=output_location,
                    output_structure=output_structure,
                    output_extension=output_structure,
                    user_request=user_request if user_request else None
                )
                
            elif choice == '2':
                # Read operation
                input_location = input("Input file path: ").strip()
                user_request = input("AI analysis request (optional): ").strip()
                
                result = crud.execute_crud_operation(
                    action="read",
                    input_location=input_location,
                    user_request=user_request if user_request else None
                )
                
            elif choice == '3':
                # Update operation
                input_location = input("Input file path: ").strip()
                output_location = input("Output file path (optional): ").strip()
                user_request = input("Update request: ").strip()
                additional_data = input("Additional data (optional): ").strip()
                
                result = crud.execute_crud_operation(
                    action="update",
                    input_data=additional_data if additional_data else None,
                    input_location=input_location,
                    output_location=output_location if output_location else None,
                    user_request=user_request
                )
                
            elif choice == '4':
                # Delete operation
                input_location = input("File path to delete: ").strip()
                confirmation = input("Type 'delete' to confirm: ").strip()
                
                result = crud.execute_crud_operation(
                    action="delete",
                    input_location=input_location,
                    user_request=f"Please {confirmation} this file permanently"
                )
                
            else:
                print("Invalid choice. Please select 1-5.")
                continue
            
            print(f"\nResult: {result['status']} - {result['message']}")
            
            if result['status'] == 'success':
                if 'content' in result:
                    print(f"Content preview: {str(result['content'])[:200]}...")
                if 'ai_analysis' in result and result['ai_analysis']:
                    print(f"AI Analysis: {result['ai_analysis'][:300]}...")
                if 'output_path' in result:
                    print(f"Output saved to: {result['output_path']}")
                    
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("File CRUD System Demonstration")
    print("Choose demonstration mode:")
    print("1. Automated demonstration")
    print("2. Interactive demonstration")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        demonstrate_crud_operations()
    elif choice == '2':
        interactive_crud_demo()
    else:
        print("Invalid choice. Running automated demonstration...")
        demonstrate_crud_operations()
