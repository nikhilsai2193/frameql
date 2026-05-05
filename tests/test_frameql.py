"""
FrameQL regression tests.

Run with:  pytest tests/test_frameql.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from FrameQL import FrameQL


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    users = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
    })
    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "user_id": [1, 1, 2, 3, 99],
        "amount": [50.0, 30.0, 20.0, 100.0, 15.0],
    })
    return FrameQL({"users": users, "orders": orders})


@pytest.fixture
def payments_engine():
    payments = pd.DataFrame({
        "payment_id": [1, 2, 3, 4, 5],
        "user_id": [1, 1, 2, 2, 3],
        "amount": [100.0, 200.0, 150.0, 50.0, 300.0],
        "status": ["paid", "paid", "pending", "paid", "paid"],
    })
    return FrameQL({"payments": payments})


@pytest.fixture
def products_engine():
    products = pd.DataFrame({
        "product_id": [1, 2, 3, 4],
        "name": ["Widget", "Gadget", "Doohickey", "Thingamajig"],
        "price": [9.99, 19.99, 4.99, 14.99],
        "category": ["A", "B", "A", "B"],
    })
    return FrameQL({"products": products})


# ─────────────────────────────────────────────────────────────
# 1. Basic SELECT
# ─────────────────────────────────────────────────────────────

class TestBasicSelect:
    def test_select_star(self, engine):
        result = engine.query("SELECT * FROM users")
        assert list(result.columns) == ["users.user_id", "users.name"]
        assert len(result) == 4

    def test_select_columns(self, engine):
        result = engine.query("SELECT user_id, name FROM users")
        assert set(result.columns) == {"user_id", "name"}
        assert len(result) == 4

    def test_select_alias(self, engine):
        result = engine.query("SELECT user_id AS id, name AS full_name FROM users")
        assert "id" in result.columns
        assert "full_name" in result.columns
        assert list(result["id"]) == [1, 2, 3, 4]

    def test_select_arithmetic(self, payments_engine):
        result = payments_engine.query("SELECT payment_id, amount * 2 AS doubled FROM payments")
        assert "doubled" in result.columns
        assert list(result["doubled"]) == [200.0, 400.0, 300.0, 100.0, 600.0]

    def test_select_literal(self, engine):
        result = engine.query("SELECT user_id, 'hello' AS greeting FROM users")
        assert list(result["greeting"]) == ["hello", "hello", "hello", "hello"]


# ─────────────────────────────────────────────────────────────
# 2. WHERE
# ─────────────────────────────────────────────────────────────

class TestWhere:
    def test_where_eq(self, engine):
        result = engine.query("SELECT user_id, name FROM users WHERE user_id = 1")
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_where_gt(self, engine):
        result = engine.query("SELECT order_id FROM orders WHERE amount > 25")
        assert len(result) == 3  # 50, 30, 100

    def test_where_and(self, engine):
        result = engine.query("SELECT order_id FROM orders WHERE amount > 25 AND amount < 60")
        assert len(result) == 2  # 50, 30

    def test_where_or(self, engine):
        result = engine.query("SELECT user_id FROM users WHERE user_id = 1 OR user_id = 3")
        assert set(result["user_id"]) == {1, 3}

    def test_where_in(self, engine):
        result = engine.query("SELECT user_id, name FROM users WHERE user_id IN (1, 2)")
        assert set(result["user_id"]) == {1, 2}

    def test_where_is_null(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, None, 30.0]})
        e = FrameQL({"t": df})
        result = e.query("SELECT id FROM t WHERE val IS NULL")
        assert list(result["id"]) == [2]

    def test_where_is_not_null(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, None, 30.0]})
        e = FrameQL({"t": df})
        result = e.query("SELECT id FROM t WHERE val IS NOT NULL")
        assert list(result["id"]) == [1, 3]

    def test_where_not(self, engine):
        result = engine.query("SELECT user_id FROM users WHERE NOT user_id = 1")
        assert 1 not in list(result["user_id"])


# ─────────────────────────────────────────────────────────────
# 3. JOINs
# ─────────────────────────────────────────────────────────────

class TestJoins:
    def test_inner_join(self, engine):
        result = engine.query("""
            SELECT u.user_id, u.name, o.amount
            FROM users u
            INNER JOIN orders o ON u.user_id = o.user_id
        """)
        # Users 1,2,3 have orders; user 99 in orders has no user match
        assert len(result) == 4  # Alice×2, Bob×1, Charlie×1
        assert set(result["name"]) == {"Alice", "Bob", "Charlie"}

    def test_left_join(self, engine):
        result = engine.query("""
            SELECT u.user_id, u.name, o.amount
            FROM users u
            LEFT JOIN orders o ON u.user_id = o.user_id
        """)
        # All 4 users; David has no orders (NULL amount)
        assert len(result) == 5  # Alice×2, Bob×1, Charlie×1, David×1(NULL)
        david_row = result[result["name"] == "David"]
        assert len(david_row) == 1
        assert pd.isna(david_row.iloc[0]["amount"])

    def test_inner_join_filter(self, engine):
        result = engine.query("""
            SELECT u.name, o.amount
            FROM users u
            INNER JOIN orders o ON u.user_id = o.user_id
            WHERE o.amount > 40
        """)
        assert all(result["amount"] > 40)

    def test_join_no_alias(self, engine):
        result = engine.query("""
            SELECT users.user_id, orders.amount
            FROM users
            INNER JOIN orders ON users.user_id = orders.user_id
        """)
        assert len(result) == 4


# ─────────────────────────────────────────────────────────────
# 4. GROUP BY + Aggregates
# ─────────────────────────────────────────────────────────────

class TestGroupBy:
    def test_sum(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            GROUP BY user_id
        """)
        assert len(result) == 3
        row = result[result["user_id"] == 1].iloc[0]
        assert row["total"] == 300.0

    def test_count(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, COUNT(*) AS cnt
            FROM payments
            GROUP BY user_id
        """)
        row = result[result["user_id"] == 1].iloc[0]
        assert row["cnt"] == 2

    def test_avg(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, AVG(amount) AS avg_amount
            FROM payments
            GROUP BY user_id
        """)
        row = result[result["user_id"] == 1].iloc[0]
        assert row["avg_amount"] == 150.0

    def test_min_max(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, MIN(amount) AS mn, MAX(amount) AS mx
            FROM payments
            GROUP BY user_id
        """)
        row = result[result["user_id"] == 1].iloc[0]
        assert row["mn"] == 100.0
        assert row["mx"] == 200.0

    def test_count_distinct(self, payments_engine):
        result = payments_engine.query("""
            SELECT COUNT(DISTINCT user_id) AS unique_users
            FROM payments
        """)
        assert result.iloc[0]["unique_users"] == 3

    def test_no_group_aggregate(self, payments_engine):
        result = payments_engine.query("""
            SELECT SUM(amount) AS grand_total
            FROM payments
        """)
        assert result.iloc[0]["grand_total"] == 800.0

    def test_group_col_not_leaked(self, payments_engine):
        """Temp columns used during GROUP BY should not appear in output."""
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            GROUP BY user_id
        """)
        for col in result.columns:
            assert not str(col).startswith("__"), f"Temp column leaked: {col}"


# ─────────────────────────────────────────────────────────────
# 5. HAVING
# ─────────────────────────────────────────────────────────────

class TestHaving:
    def test_having_simple(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            GROUP BY user_id
            HAVING SUM(amount) > 200
        """)
        assert len(result) == 2  # user 1 (300) and user 3 (300)
        assert all(result["total"] > 200)

    def test_having_alias(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, COUNT(*) AS cnt
            FROM payments
            GROUP BY user_id
            HAVING COUNT(*) >= 2
        """)
        assert len(result) == 2  # user 1 (2) and user 2 (2)

    def test_having_with_where(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            WHERE status = 'paid'
            GROUP BY user_id
            HAVING SUM(amount) > 100
        """)
        # Paid amounts: user1=300, user2=50, user3=300
        assert all(result["total"] > 100)


# ─────────────────────────────────────────────────────────────
# 6. ORDER BY  (critical bug fixes)
# ─────────────────────────────────────────────────────────────

class TestOrderBy:
    def test_order_by_column_asc(self, engine):
        result = engine.query("SELECT user_id, name FROM users ORDER BY user_id ASC")
        assert list(result["user_id"]) == [1, 2, 3, 4]

    def test_order_by_column_desc(self, engine):
        result = engine.query("SELECT user_id, name FROM users ORDER BY user_id DESC")
        assert list(result["user_id"]) == [4, 3, 2, 1]

    def test_order_by_alias(self, payments_engine):
        """BUG FIX: ORDER BY alias defined in SELECT."""
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            GROUP BY user_id
            ORDER BY total DESC
        """)
        totals = list(result["total"])
        assert totals == sorted(totals, reverse=True)

    def test_order_by_aggregate(self, payments_engine):
        """BUG FIX: ORDER BY aggregate expression must NOT recompute."""
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            GROUP BY user_id
            ORDER BY SUM(amount) DESC
        """)
        totals = list(result["total"])
        assert totals == sorted(totals, reverse=True)

    def test_order_by_does_sort(self, engine):
        """BUG FIX: sort_values result must be assigned."""
        result = engine.query("SELECT order_id, amount FROM orders ORDER BY amount ASC")
        amounts = list(result["amount"])
        assert amounts == sorted(amounts)

    def test_order_by_non_projected_col(self, engine):
        """ORDER BY on a column present in the table."""
        result = engine.query("""
            SELECT name FROM users ORDER BY user_id ASC
        """)
        assert list(result["name"]) == ["Alice", "Bob", "Charlie", "David"]

    def test_order_by_with_limit(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, amount FROM payments ORDER BY amount DESC LIMIT 2
        """)
        assert len(result) == 2
        assert list(result["amount"]) == sorted(result["amount"], reverse=True)


# ─────────────────────────────────────────────────────────────
# 7. CASE expression
# ─────────────────────────────────────────────────────────────

class TestCase:
    def test_case_basic(self, engine):
        result = engine.query("""
            SELECT user_id,
                   CASE
                     WHEN user_id = 1 THEN 'Alice'
                     WHEN user_id = 2 THEN 'Bob'
                     ELSE 'Other'
                   END AS label
            FROM users
        """)
        labels = dict(zip(result["user_id"], result["label"]))
        assert labels[1] == "Alice"
        assert labels[2] == "Bob"
        assert labels[3] == "Other"
        assert labels[4] == "Other"

    def test_case_no_else(self, engine):
        result = engine.query("""
            SELECT user_id,
                   CASE WHEN user_id = 1 THEN 'one' END AS lbl
            FROM users
        """)
        lbl = dict(zip(result["user_id"], result["lbl"]))
        assert lbl[1] == "one"
        assert lbl[2] is None or (isinstance(lbl[2], float) and np.isnan(lbl[2]))

    def test_case_in_where(self, engine):
        result = engine.query("""
            SELECT user_id FROM users
            WHERE CASE WHEN user_id < 3 THEN 1 ELSE 0 END = 1
        """)
        assert set(result["user_id"]) == {1, 2}


# ─────────────────────────────────────────────────────────────
# 8. DISTINCT
# ─────────────────────────────────────────────────────────────

class TestDistinct:
    def test_distinct(self, payments_engine):
        result = payments_engine.query("SELECT DISTINCT user_id FROM payments")
        assert len(result) == 3
        assert len(result["user_id"].unique()) == 3

    def test_distinct_with_where(self, payments_engine):
        result = payments_engine.query(
            "SELECT DISTINCT user_id FROM payments WHERE amount > 100"
        )
        assert all(uid in [1, 2, 3] for uid in result["user_id"])


# ─────────────────────────────────────────────────────────────
# 9. LIMIT
# ─────────────────────────────────────────────────────────────

class TestLimit:
    def test_limit(self, engine):
        result = engine.query("SELECT user_id FROM users LIMIT 2")
        assert len(result) == 2

    def test_limit_with_order(self, engine):
        result = engine.query("SELECT user_id FROM users ORDER BY user_id DESC LIMIT 1")
        assert len(result) == 1
        assert result.iloc[0]["user_id"] == 4


# ─────────────────────────────────────────────────────────────
# 10. Subqueries
# ─────────────────────────────────────────────────────────────

class TestSubqueries:
    def test_subquery_in_where(self, engine):
        result = engine.query("""
            SELECT user_id, name FROM users
            WHERE user_id IN (SELECT user_id FROM orders)
        """)
        # orders has user_ids 1,2,3,99; users has 1,2,3,4
        assert set(result["user_id"]) == {1, 2, 3}

    def test_scalar_subquery_in_select(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, amount,
                   (SELECT MAX(amount) FROM payments) AS max_amount
            FROM payments
            WHERE user_id = 1
        """)
        assert all(result["max_amount"] == 300.0)

    def test_scalar_subquery_in_where(self, payments_engine):
        result = payments_engine.query("""
            SELECT payment_id, amount FROM payments
            WHERE amount > (SELECT AVG(amount) FROM payments)
        """)
        avg = 800.0 / 5
        assert all(result["amount"] > avg)

    def test_correlated_subquery(self, payments_engine):
        result = payments_engine.query("""
            SELECT p.user_id, p.amount,
                   (SELECT SUM(p2.amount)
                    FROM payments p2
                    WHERE p2.user_id = p.user_id) AS user_total
            FROM payments p
            WHERE p.user_id = 1
        """)
        assert len(result) == 2
        assert all(result["user_total"] == 300.0)

    def test_subquery_in_from(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, total
            FROM (
                SELECT user_id, SUM(amount) AS total
                FROM payments
                GROUP BY user_id
            ) sub
            WHERE total > 200
        """)
        assert all(result["total"] > 200)


# ─────────────────────────────────────────────────────────────
# 11. NULL semantics
# ─────────────────────────────────────────────────────────────

class TestNullSemantics:
    def test_null_comparison_false(self):
        df = pd.DataFrame({"id": [1, 2], "val": [None, 5.0]})
        e = FrameQL({"t": df})
        result = e.query("SELECT id FROM t WHERE val = val")
        # NULL = NULL should be False in SQL
        assert 1 not in list(result["id"]) or len(result) <= 1

    def test_is_null(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [None, 5.0, None]})
        e = FrameQL({"t": df})
        result = e.query("SELECT id FROM t WHERE val IS NULL")
        assert set(result["id"]) == {1, 3}

    def test_is_not_null(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [None, 5.0, None]})
        e = FrameQL({"t": df})
        result = e.query("SELECT id FROM t WHERE val IS NOT NULL")
        assert list(result["id"]) == [2]


# ─────────────────────────────────────────────────────────────
# 12. Edge cases from the known-bugs list
# ─────────────────────────────────────────────────────────────

class TestKnownBugFixes:
    def test_order_by_aggregate_no_recompute(self, payments_engine):
        """
        SELECT user_id, SUM(amount) FROM payments GROUP BY user_id ORDER BY SUM(amount)
        The SUM must NOT be recomputed after GROUP BY.
        """
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            GROUP BY user_id
            ORDER BY SUM(amount) ASC
        """)
        assert list(result["total"]) == sorted(result["total"])
        # Verify values are correct, not re-aggregated nonsense
        assert set(result["total"]) == {300.0, 200.0, 300.0}

    def test_order_by_alias_resolves(self, payments_engine):
        """ORDER BY alias defined in SELECT must work."""
        result = payments_engine.query("""
            SELECT SUM(amount) AS grand_total
            FROM payments
            ORDER BY grand_total
        """)
        assert result.iloc[0]["grand_total"] == 800.0

    def test_group_by_temp_cols_not_in_output(self, payments_engine):
        """GROUP BY internal temp cols must not appear in final output."""
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            GROUP BY user_id
            HAVING SUM(amount) > 100
            ORDER BY total DESC
        """)
        for col in result.columns:
            assert not col.startswith("__"), f"Temp col leaked: {col}"
            assert not col.startswith("tmp_"), f"Old temp col leaked: {col}"

    def test_sort_actually_sorts(self, engine):
        """Regression: sort_values result must be assigned (was missing = )."""
        result = engine.query("SELECT order_id, amount FROM orders ORDER BY amount DESC")
        amounts = list(result["amount"])
        assert amounts[0] == max(amounts), "Rows not sorted - sort_values result not assigned"

    def test_having_reuses_aggregate(self, payments_engine):
        """HAVING must reuse the same aggregate computed in GROUP BY, not recompute."""
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total
            FROM payments
            GROUP BY user_id
            HAVING SUM(amount) = 300
        """)
        assert all(result["total"] == 300.0)
        assert len(result) == 2  # user 1 and user 3

    def test_multiple_aggregates_in_select(self, payments_engine):
        result = payments_engine.query("""
            SELECT user_id, SUM(amount) AS total, COUNT(*) AS cnt, AVG(amount) AS avg
            FROM payments
            GROUP BY user_id
        """)
        row1 = result[result["user_id"] == 1].iloc[0]
        assert row1["total"] == 300.0
        assert row1["cnt"] == 2
        assert row1["avg"] == 150.0

    def test_join_then_group_order(self, engine):
        """Full pipeline: JOIN → GROUP BY → ORDER BY alias."""
        result = engine.query("""
            SELECT u.name, SUM(o.amount) AS total_spent
            FROM users u
            INNER JOIN orders o ON u.user_id = o.user_id
            GROUP BY u.name
            ORDER BY total_spent DESC
        """)
        assert list(result.columns) == ["name", "total_spent"]
        totals = list(result["total_spent"])
        assert totals == sorted(totals, reverse=True)
