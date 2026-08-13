# Third-party notices

This file is a practical dependency inventory, not legal advice and not a
substitute for the licence text shipped by each provider. The root
Apache-2.0 `LICENSE` covers only original software identified in
`LICENSE_SCOPE.md`; it does not relicense dependencies, data, models, simulator
assets, or other third-party material.

## Python packages

The requirement profiles reference third-party packages including NumPy,
MediaPipe, OpenCV, psutil, Matplotlib, Open3D, imageio-ffmpeg, python-docx,
python-pptx, pypdf, and PyMuPDF. Their copyright and licence terms remain with
their respective projects. Verify the current upstream terms for every selected
version before distributing a document-tooling environment.

Packages are installed from their normal distribution channels and are not
relicensed by this repository. Review resolved package metadata and bundled
licence files when producing a release or container, because transitive
dependencies may add obligations.

## Vendor SDKs and simulator assets

Stereolabs ZED SDK/`pyzed`, NVIDIA Isaac Sim/Omniverse modules, GPU drivers,
body-tracking models, and simulator content are external, vendor-managed
components. They are deliberately excluded from PyPI requirements. Installation,
redistribution, cloud execution, and captured-data use remain subject to the
applicable vendor and asset licences.

The MediaPipe pose model files under `models/mediapipe/` are third-party model
artifacts. Confirm their source, version, checksum, notice, and redistribution terms
before publishing a release archive.

## Data and generated artifacts

Real-person SVO/SVO2 files, RGB-D exports, the local `mesh.obj`, and generated
experiment output are excluded from Git by default. Consent to participate in a
study is not automatically permission to publish raw biometric data. A public
release should use synthetic or explicitly cleared examples and document their
provenance separately.
