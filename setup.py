import os
import venv
import subprocess
import sys

def create_venv():
    venv.create('venv', with_pip=True)

def get_python_executable():
    if sys.platform == 'win32':
        return os.path.join('venv', 'Scripts', 'python.exe')
    return os.path.join('venv', 'bin', 'python')

def install_requirements():
    python = get_python_executable()
    subprocess.check_call([python, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([python, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def download_yolo_model():
    python = get_python_executable()
    subprocess.check_call([python, '-c', 'from ultralytics import YOLO; YOLO("yolov10m.pt")'])
    print("YOLOv10 model downloaded successfully.")

if __name__ == '__main__':
    if not os.path.exists('venv'):
        print("Creating virtual environment...")
        create_venv()
    
    print("Installing requirements...")
    install_requirements()
    
    print("Downloading YOLO model...")
    download_yolo_model()
    
    print("Setup completed successfully.")