import os

def list_files_in_directory(path):
    # Loop through directories and files recursively
    for root, dirs, files in os.walk(path):
        # Print directories
        for dir in dirs:
            print(f"[DIR] {os.path.join(root, dir)}")
        # Print files
        for file in files:
            print(f"[FILE] {os.path.join(root, file)}")

# Change this to the desired directory path
directory_path = "D:/mangopro"
list_files_in_directory(directory_path)
