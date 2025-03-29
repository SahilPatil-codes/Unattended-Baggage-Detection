import os
import subprocess
import sys
import shutil

def get_python_executable():
    if sys.platform == 'win32':
        return os.path.join('venv', 'Scripts', 'python.exe')
    return os.path.join('venv', 'bin', 'python')

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error: {error.decode('utf-8')}")
        sys.exit(1)
    return output.decode('utf-8')

def build():
    # Ensure we're in a virtual environment
    if not os.path.exists('venv'):
        print("Virtual environment not found. Running setup...")
        subprocess.call([sys.executable, 'setup.py'])

    python = get_python_executable()

    # Install requirements
    print("Installing requirements...")
    run_command(f"{python} -m pip install -r requirements.txt")

    # Create the executable
    print("Creating executable...")
    run_command(f"{python} -m PyInstaller --name=AirportSecurityGUI --windowed --onefile airport_security_gui.py")

    # Copy necessary files
    print("Copying necessary files...")
    shutil.copy("yolov10m.pt", "dist/yolov10m.pt")
    shutil.copy("logo.png", "dist/logo.png")

    print("Build complete. Executable is in the 'dist' folder.")

if __name__ == '__main__':
    build()