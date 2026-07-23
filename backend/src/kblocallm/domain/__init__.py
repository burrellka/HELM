"""Pure derivation logic.

Everything in here is a pure function over data the upstream already gave us.
No I/O, no httpx, no FastAPI. That is deliberate: these are the functions that
turn a raw router payload into a number an operator will act on, and per the
spec's §6 standard a wrong one here is worse than no UI at all. Pure means
table-testable, and table-testable is the only reason to believe them.

If you find yourself wanting to `await` something in this package, the thing
you want belongs in `api/` or `upstream/` instead.
"""
