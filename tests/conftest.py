"""Import torch before scikit-learn, or Windows refuses to load it.

On Windows, scipy/scikit-learn and torch each ship their own Intel OpenMP
runtime. Whichever loads second loses: importing sklearn first and torch after
raises

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading ...\\torch\\lib\\c10.dll

Collecting the test suite imports `lbi.probes` (and therefore sklearn) before it
reaches `tests/test_hooks.py`, so the whole run died at collection even though
`pytest tests/test_hooks.py` on its own passed. Importing torch here, before any
test module is collected, fixes the order once for the whole session.

This is deliberately not done in `lbi/__init__.py`: the tests and the synthetic
demo are supposed to run with no torch installed at all, and importing it at
package level would make it a hard dependency. Linux and macOS are unaffected.
"""

try:  # pragma: no cover - environment dependent
    import torch  # noqa: F401
except Exception:  # torch is optional; the CPU-only tests skip past it
    pass
