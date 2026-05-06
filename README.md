# FrameQL

Execute SQL queries directly on Pandas DataFrames — no database required.

FrameQL lets you write standard SQL against in-memory DataFrames. It uses [sqlglot](https://github.com/tobymao/sqlglot) to parse SQL and executes the resulting query plan entirely in Python/Pandas. This is useful when you have data already loaded in memory and want to express complex transformations as SQL rather than chained DataFrame operations.

---

## Installation

```bash
pip install frameql
```

Requires Python 3.9+.

---

## Quick Start

```python
import pandas as pd
from frameql import FrameQL

users = pd.DataFrame({
    "user_id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
})
orders = pd.DataFrame({
    "order_id": [101, 102, 103],
    "user_id": [1, 1, 2],
    "amount": [50.0, 30.0, 20.0],
})

engine = FrameQL({"users": users, "orders": orders})

result = engine.query("""
    SELECT u.name, SUM(o.amount) AS total_spent
    FROM users u
    INNER JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.name
    ORDER BY total_spent DESC
""")
print(result)
#       name  total_spent
# 0    Alice         80.0
# 1      Bob         20.0
```

---

## Detailed Usage

### Initializing the Engine

Pass a dictionary of table name → DataFrame to `FrameQL`. Table names are case-insensitive.

```python
from frameql import FrameQL

engine = FrameQL({
    "employees": employees_df,
    "departments": departments_df,
})
```

### Running SELECT Queries

Use `engine.query(sql)` for any SELECT statement. It returns a `pd.DataFrame`.

```python
result = engine.query("SELECT name FROM employees WHERE salary > 60000")
```

### Running DML Statements

Use `engine.execute(sql)` for INSERT, UPDATE, and DELETE. It mutates the in-memory DataFrames and returns `None`. It also accepts SELECT and routes to `query()` in that case.

```python
# INSERT
engine.execute("INSERT INTO employees (id, name, salary) VALUES (4, 'Dana', 70000)")

# UPDATE
engine.execute("UPDATE employees SET salary = salary * 1.1 WHERE department = 'Engineering'")

# DELETE
engine.execute("DELETE FROM employees WHERE salary < 40000")

# SELECT (returns a DataFrame)
result = engine.execute("SELECT COUNT(*) AS cnt FROM employees")
```

### Accessing Modified Tables

After DML operations, the modified DataFrame is available at `engine.tables[table_name]`:

```python
engine.execute("DELETE FROM orders WHERE amount < 25")
print(engine.tables["orders"])
```

---

## How It Works

1. **SQL Parsing** — `sqlglot` parses the SQL string into an AST.
2. **Logical Plan** — FrameQL scans the AST upfront to build a `QueryPlan`, resolving column references, collecting aggregate specifications, and identifying window functions before any execution begins.
3. **Execution Pipeline** — The plan is executed in SQL-standard order:
   - `WHERE` → filter rows
   - `GROUP BY` → aggregate
   - Window functions (computed after GROUP BY, before SELECT)
   - `HAVING` → post-aggregate filter
   - `SELECT` → project columns
   - `DISTINCT` → deduplicate
   - `ORDER BY` → sort (aliases from SELECT are visible here)
   - `LIMIT` → truncate

CTEs are materialized as temporary tables scoped to the query and cleaned up automatically.

---

## Features (v0.1.0)

### SELECT
- `SELECT *`, column list, aliases, arithmetic expressions, literals
- `DISTINCT`
- `LIMIT`

### Filtering
- `WHERE` with `=`, `!=`, `<`, `<=`, `>`, `>=`
- `AND`, `OR`, `NOT`
- `IS NULL` / `IS NOT NULL`
- `IN` with literal list or subquery
- Tuple / multi-column `IN`: `(a, b) IN (SELECT x, y FROM …)`
- `EXISTS` / `NOT EXISTS` (correlated and non-correlated)
- `ANY` / `ALL` quantifiers

### Joins
- `INNER JOIN`, `LEFT JOIN`
- Equi-joins and non-equi-join conditions
- Mixed conditions on JOIN `ON` clauses (column = column AND column = literal)

### Aggregation
- `GROUP BY` with `SUM`, `COUNT`, `AVG`, `MIN`, `MAX`
- `COUNT(DISTINCT col)`
- Bare aggregates without `GROUP BY`
- `HAVING` clause (reuses computed aggregates, no recomputation)

### Window Functions
- `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`
- `SUM()`, `AVG()`, `MIN()`, `MAX()`, `COUNT()` as window functions
- `LAG()` and `LEAD()` with optional offset and default value
- `PARTITION BY` and `ORDER BY` within window spec

### Subqueries
- Scalar subqueries in `SELECT` and `WHERE`
- Subquery in `FROM` clause (derived table)
- Correlated subqueries

### Set Operations
- `UNION` (deduplicates), `UNION ALL`
- `INTERSECT`, `EXCEPT`
- `ORDER BY` and `LIMIT` on set operation results
- Set operations inside CTEs and subqueries

### CTEs
- `WITH` clause (Common Table Expressions)
- Chained CTEs (each CTE can reference earlier ones)
- CTEs containing window functions, aggregations, or set operations

### DML
- `INSERT INTO … VALUES (…)` — single and multi-row
- `INSERT INTO … SELECT …`
- `UPDATE … SET … WHERE …` with subqueries and expressions
- `DELETE FROM … WHERE …` with subqueries
- `CASE` expressions in `SET` clauses

---

## Limitations / Not Supported

- `RIGHT JOIN` and `FULL OUTER JOIN`
- `CROSS JOIN` syntax (workaround: `INNER JOIN` with always-true condition, or omit ON)
- Recursive CTEs
- Window frame clauses (`ROWS BETWEEN …`, `RANGE BETWEEN …`)
- `ROLLUP`, `CUBE`, `GROUPING SETS`
- String functions (`UPPER`, `LOWER`, `TRIM`, `SUBSTRING`, etc.)
- Date/time functions
- `COALESCE`, `NULLIF`, `IIF`
- `LIKE` and `ILIKE` pattern matching
- Multi-column `COUNT(DISTINCT col1, col2)` — use single-column form
- `MERGE` / `UPSERT`
- Schema enforcement or type coercion on INSERT

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
