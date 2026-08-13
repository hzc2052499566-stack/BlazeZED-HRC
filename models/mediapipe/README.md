# MediaPipe model assets

This directory records the MediaPipe task bundles used by the multi-person
engineering pilot. Binary task files are local third-party assets and are
ignored by Git until their source, version, notice, and redistribution terms
have been reviewed for a release.

The registered assets are:

- `pose_landmarker_lite.task`: BlazePose GHUM Lite bundle for the live-oriented
  bridge; local SHA-256
  `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.
- `pose_landmarker_full.task`: BlazePose GHUM Full bundle for the
  accuracy-oriented offline bridge; local SHA-256
  `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad`.

Both bundles are used by `tools/offline_multiperson_blazepose_rgbd_v1.py`.

Do not replace a model file in place after any capture or replay has been
registered. Add a new filename and method version instead.

The checksums identify the local frozen assets; they do not grant permission to
redistribute them and are not a download source.
