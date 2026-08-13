## Summary

Describe the change, its intended scope, and the evidence used to validate it.

## Validation

List exact commands and results. If a test was not applicable or could not run,
state why.

```text
python -m compileall -q tools tests
python tests/portable_suite.py
```

## Research and data checklist

- [ ] I preserved frozen protocols, manifests, hashes, raw captures, and registered negative results.
- [ ] I did not commit SVO/SVO2 files, identifiable RGB-D data, `mesh.obj`, generated `output/`, credentials, or machine-local paths.
- [ ] I did not describe inferred K4 joints as measured observations.
- [ ] I kept BlazePose + ZED RGB-D as the baseline and ZED Body Tracking as a reference/comparison.
- [ ] I used the registered endpoint name and denominator for any reported metric.
- [ ] I scoped single-person, simulation, offline, and engineering evidence accurately.
- [ ] I documented any dependency, hardware, SDK, simulator, or asset prerequisite.

## Code checklist

- [ ] The portable suite passes, or I documented why it is unaffected/not runnable.
- [ ] New tests are deterministic and do not require private data unless clearly separated from portable CI.
- [ ] Paths are repository-relative or configurable; no new workstation-specific defaults were added.
- [ ] Documentation was updated where user-facing behaviour or reproducibility changed.
