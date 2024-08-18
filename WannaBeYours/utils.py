import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install(package, import_name=None):
    try:
        if import_name is None:
            __import__(package)
        else:
            __import__(import_name)
    except ImportError:
        install(package)

# Libraries for DonaDev
libraries = {
    'torch': 'torch',
    'torchvision': 'torchvision',
    'Pillow': 'PIL',
    'requests': 'requests',
    'gitpython': 'git',
    'matplotlib': 'matplotlib',
}

for package, import_name in libraries.items():
    check_and_install(package, import_name)

print("All necessary libraries are installed.")
