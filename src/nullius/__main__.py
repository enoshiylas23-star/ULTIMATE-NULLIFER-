"""Allow `python -m nullius` to dispatch to the CLI."""

from __future__ import annotations

import sys

from nullius.cli import main

if __name__ == "__main__":
    sys.exit(main())
