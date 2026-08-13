# Dependency profiles

Use Python 3.11 and install only the profile needed for the task:

```powershell
python -m pip install -r requirements/ci.txt
python -m pip install -r requirements/core.txt
python -m pip install -r requirements/analysis.txt
python -m pip install -r requirements/documents.txt
```

`ci.txt` is intentionally minimal. It supports the 23-module portable unit-test
allowlist and is the only profile installed by the required GitHub Actions job.

`core.txt` covers the portable BlazePose/RGB-D scripts. It selects one OpenCV
distribution (`opencv-contrib-python`, also required by MediaPipe) because
installing multiple distributions that provide the same `cv2` namespace is
fragile. `analysis.txt` adds offline plotting,
mesh-registration, and video helpers. Open3D 0.18 legacy APIs are not compatible
with every NumPy 2.x operation; the existing registration workflow should keep to
the tensor API or be validated in an isolated environment. `documents.txt` is
independent and only supports document-generation utilities.

## SDK-managed environments

The following dependencies are deliberately absent from all PyPI requirement
files:

- `pyzed` is supplied by a matching Stereolabs ZED SDK installation. The frozen
  experiments used ZED SDK/pyzed 5.4.0.
- `isaacsim`, `omni.*`, `pxr`, and `carb` must run inside an Isaac Sim configured
  Python environment.

Do not replace either SDK with an unrelated package of the same name from PyPI.
GPU drivers, camera firmware, downloaded body-tracking models, Isaac Sim assets,
and real-person captures are also external prerequisites rather than repository
dependencies.

The pins describe the environment used to organise this repository; they are not
a claim that every historical experiment is reproducible on a hardware-free
machine. See the repository environment and reproducibility documentation before
running hardware or simulator workflows.
