import subprocess

# Define the file paths for the three Python files you want to run sequentially
face_path = "face.py"
file1_path = "fresh2.py"
file2_path = "fresh_squat2.py"
file3_path = "fresh_pushup2.py"

# Define a function to run a Python file using subprocess
def run_python_file(file_path):
    try:
        subprocess.run(["python", file_path], check=True)
        print(f"Successfully ran {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {file_path}: {e}")

# Run the three Python files sequentially
run_python_file(face_path)
run_python_file(file1_path)
run_python_file(file2_path)
run_python_file(file3_path)
