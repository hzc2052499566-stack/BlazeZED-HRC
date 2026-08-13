# Security policy

## Supported code

Security fixes target the current default branch. Historical experiment snapshots,
SDK-managed environments, and frozen artifacts are retained for provenance and
may not receive backports.

## Reporting a vulnerability

Please use GitHub's private **Report a vulnerability** / security-advisory feature
for this repository when it is available. If it is unavailable, contact the
repository maintainers through a private channel they publish. Do not open a public
issue containing exploit details, credentials, camera addresses, serial numbers,
personal data, or identifiable RGB-D frames.

Include the affected commit, impact, minimal reproduction, and suggested
mitigation. Use synthetic inputs whenever possible. The maintainers can coordinate
acknowledgement, remediation, and disclosure timing after receiving the report.

## Data and operational safety

SVO/SVO2 recordings and RGB-D exports can contain identifiable biometric and
environmental information. Keep them outside Git, minimise access, and follow the
applicable consent and retention policy. Treat network camera endpoints, local
launchers, SDK logs, manifests containing machine metadata, and cloud credentials
as sensitive.

This research code is not a certified robot-safety system. Before connecting it to
physical motion, use independent safety controls, workspace limits, emergency-stop
procedures, and a risk assessment appropriate to the robot and site.
