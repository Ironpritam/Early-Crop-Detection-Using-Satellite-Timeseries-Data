import sys
from cx_Freeze import setup, Executable

base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Use "Win32GUI" for GUI applications

setup(
    name="CropApp",
    version="1.0",
    description="Description of your application",
    executables=[Executable("Crop_app.py", base=base)],
    options={
        "build_exe": {
            # "packages": ["os"],  # Add any packages used by your application
            "include_files": ["Images/background_image.jpg", "model.py", "dataset_build.py"]
        }
    }
)