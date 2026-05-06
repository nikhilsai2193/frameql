"""
FrameQL — SQL queries on Pandas DataFrames.

Execute standard SQL directly against in-memory Pandas DataFrames without
loading data into a database. Powered by sqlglot for SQL parsing.

Basic usage::

    import pandas as pd
    from frameql import FrameQL

    users = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    engine = FrameQL({"users": users})
    result = engine.query("SELECT name FROM users WHERE id > 1")
"""

from frameql._engine import FrameQL

__all__ = ["FrameQL"]
__version__ = "0.1.0"
