import os
import sys
import subprocess

def get_python_executable():
    if sys.platform == 'win32':
        return os.path.join('venv', 'Scripts', 'python.exe')
    return os.path.join('venv', 'bin', 'python')

def run():
    # Check if venv exists, if not, run setup.py
    if not os.path.exists('venv'):
        print("Virtual environment not found. Running setup...")
        subprocess.call([sys.executable, 'setup.py'])

    # Get the python executable from the virtual environment
    python = get_python_executable()

    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        # Run the build script
        subprocess.call([python, 'build.py'])
    else:
        # Run the GUI script
        subprocess.call([python, 'airport_security_gui.py'])

if __name__ == '__main__':
    run()