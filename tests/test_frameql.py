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


# ─────────────────────────────────────────────────────────────
# 13. EXISTS / NOT EXISTS
# ─────────────────────────────────────────────────────────────

class TestExists:
    def test_exists(self, engine):
        """EXISTS returns users who have at least one order."""
        result = engine.query("""
            SELECT name
            FROM users u
            WHERE EXISTS (
                SELECT 1 FROM orders o WHERE o.user_id = u.user_id
            )
        """)
        assert set(result["name"]) == {"Alice", "Bob", "Charlie"}
        assert len(result) == 3

    def test_not_exists(self, engine):
        """NOT EXISTS returns users who have no orders."""
        result = engine.query("""
            SELECT name
            FROM users u
            WHERE NOT EXISTS (
                SELECT 1 FROM orders o WHERE o.user_id = u.user_id
            )
        """)
        assert set(result["name"]) == {"David"}
        assert len(result) == 1

    def test_exists_non_correlated(self, engine):
        """Non-correlated EXISTS: true for all rows when subquery returns rows."""
        result = engine.query("""
            SELECT user_id FROM users
            WHERE EXISTS (SELECT 1 FROM orders WHERE amount > 10)
        """)
        assert len(result) == 4  # all users — subquery always returns rows

    def test_not_exists_non_correlated(self, engine):
        """Non-correlated NOT EXISTS: false for all rows when subquery returns rows."""
        result = engine.query("""
            SELECT user_id FROM users
            WHERE NOT EXISTS (SELECT 1 FROM orders WHERE amount > 10)
        """)
        assert len(result) == 0  # no users — subquery always returns rows


# ─────────────────────────────────────────────────────────────
# 14. ANY / ALL
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def any_all_engine():
    products = pd.DataFrame({
        "product_id": [1, 2, 3, 4],
        "name": ["Widget", "Gadget", "Doohickey", "Thingamajig"],
        "price": [9.99, 19.99, 4.99, 14.99],
    })
    discounts = pd.DataFrame({
        "discount_id": [1, 2, 3],
        "price": [5.00, 8.00, 12.00],
    })
    return FrameQL({"products": products, "discounts": discounts})


class TestAnyAll:
    def test_all(self, any_all_engine):
        """price > ALL(discount prices) means price > max(discount prices) = 12."""
        result = any_all_engine.query("""
            SELECT name, price FROM products
            WHERE price > ALL (SELECT price FROM discounts)
        """)
        assert all(result["price"] > 12.0)
        assert set(result["name"]) == {"Gadget", "Thingamajig"}

    def test_any(self, any_all_engine):
        """price > ANY(discount prices) means price > min(discount prices) = 5."""
        result = any_all_engine.query("""
            SELECT name, price FROM products
            WHERE price > ANY (SELECT price FROM discounts)
        """)
        assert all(result["price"] > 5.0)
        assert set(result["name"]) == {"Widget", "Gadget", "Thingamajig"}

    def test_any_empty_subquery(self):
        """ANY over an empty subquery always returns False."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        empty = pd.DataFrame({"v": pd.Series([], dtype=float)})
        e = FrameQL({"t": df, "empty": empty})
        result = e.query("SELECT x FROM t WHERE x > ANY (SELECT v FROM empty)")
        assert len(result) == 0

    def test_all_empty_subquery(self):
        """ALL over an empty subquery always returns True (vacuous truth)."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        empty = pd.DataFrame({"v": pd.Series([], dtype=float)})
        e = FrameQL({"t": df, "empty": empty})
        result = e.query("SELECT x FROM t WHERE x > ALL (SELECT v FROM empty)")
        assert len(result) == 3  # all rows pass vacuous ALL


# ─────────────────────────────────────────────────────────────
# 15. Window Functions
# ─────────────────────────────────────────────────────────────

class TestWindowFunctions:
    def test_row_number_partition_order(self, payments_engine):
        """ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount)."""
        result = payments_engine.query("""
            SELECT user_id,
                   amount,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount) AS rn
            FROM payments
        """)
        assert "rn" in result.columns
        # Within each user partition, rn should start at 1
        for uid, grp in result.groupby("user_id"):
            sorted_grp = grp.sort_values("amount")
            assert list(sorted_grp["rn"]) == list(range(1, len(grp) + 1))

    def test_row_number_values(self, payments_engine):
        """Verify exact ROW_NUMBER values per user."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount) AS rn
            FROM payments
            ORDER BY user_id, amount
        """)
        # user_id=1: amounts 100→rn=1, 200→rn=2
        u1 = result[result["user_id"] == 1].sort_values("amount")
        assert list(u1["rn"]) == [1, 2]
        # user_id=2: amounts 50→rn=1, 150→rn=2
        u2 = result[result["user_id"] == 2].sort_values("amount")
        assert list(u2["rn"]) == [1, 2]
        # user_id=3: single row → rn=1
        u3 = result[result["user_id"] == 3]
        assert list(u3["rn"]) == [1]

    def test_sum_over_partition(self, payments_engine):
        """SUM(amount) OVER (PARTITION BY user_id) gives per-user total per row."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   SUM(amount) OVER (PARTITION BY user_id) AS user_total
            FROM payments
        """)
        assert "user_total" in result.columns
        # user_id=1 total = 300
        assert all(result[result["user_id"] == 1]["user_total"] == 300.0)
        # user_id=2 total = 200
        assert all(result[result["user_id"] == 2]["user_total"] == 200.0)
        # user_id=3 total = 300
        assert all(result[result["user_id"] == 3]["user_total"] == 300.0)

    def test_window_no_temp_cols_leaked(self, payments_engine):
        """Internal __win_* columns must not appear in SELECT * output."""
        result = payments_engine.query("""
            SELECT user_id,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount) AS rn
            FROM payments
        """)
        for col in result.columns:
            assert not str(col).startswith("__win_"), f"Window temp col leaked: {col}"


# ─────────────────────────────────────────────────────────────
# 16. Common Table Expressions (CTEs)
# ─────────────────────────────────────────────────────────────

class TestCTE:
    def test_simple_cte(self, payments_engine):
        """Basic CTE: compute totals then filter."""
        result = payments_engine.query("""
            WITH totals AS (
                SELECT user_id, SUM(amount) AS total
                FROM payments
                GROUP BY user_id
            )
            SELECT user_id, total FROM totals
            WHERE total > 100
        """)
        assert all(result["total"] > 100)
        assert len(result) == 3  # all users have total > 100

    def test_cte_filter_strict(self, payments_engine):
        """CTE with stricter filter."""
        result = payments_engine.query("""
            WITH totals AS (
                SELECT user_id, SUM(amount) AS total
                FROM payments
                GROUP BY user_id
            )
            SELECT user_id, total FROM totals
            WHERE total > 250
        """)
        # user1=300, user2=200, user3=300 → only user1 and user3 pass
        assert all(result["total"] > 250)
        assert len(result) == 2

    def test_chained_ctes(self, payments_engine):
        """Second CTE can reference first CTE."""
        result = payments_engine.query("""
            WITH totals AS (
                SELECT user_id, SUM(amount) AS total
                FROM payments
                GROUP BY user_id
            ),
            big_spenders AS (
                SELECT user_id, total FROM totals WHERE total >= 300
            )
            SELECT user_id FROM big_spenders
        """)
        assert len(result) == 2  # user_id 1 (300) and 3 (300)
        assert set(result["user_id"]) == {1, 3}

    def test_cte_cleanup(self, payments_engine):
        """CTE name must not pollute the engine's table registry after query."""
        payments_engine.query("""
            WITH temp_cte AS (SELECT user_id FROM payments)
            SELECT user_id FROM temp_cte
        """)
        assert "temp_cte" not in payments_engine.tables


# ─────────────────────────────────────────────────────────────
# Fixtures for new edge-case tests
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def ranked_engine():
    """Engine for top-N per group / CTE+window+join scenarios."""
    users = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
    })
    orders = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3],
        "amount": [50.0, 30.0, 70.0, 20.0, 90.0, 100.0],
    })
    return FrameQL({"users": users, "orders": orders})


@pytest.fixture
def nulls_engine():
    left = pd.DataFrame({"id": [1, 2, 3], "val": [10, None, 30]})
    right = pd.DataFrame({"id": [1, 2], "label": ["A", "B"]})
    return FrameQL({"left": left, "right": right})


# ─────────────────────────────────────────────────────────────
# 17. JOIN edge cases
# ─────────────────────────────────────────────────────────────

class TestJoinEdgeCases:
    def test_join_with_literal_in_on(self, ranked_engine):
        """ON clause with column=column AND column=literal must not raise KeyError."""
        result = ranked_engine.query("""
            WITH ranked AS (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY amount DESC
                       ) AS rn
                FROM orders
            )
            SELECT u.user_id, u.name, r.amount AS top_order
            FROM users u
            LEFT JOIN ranked r ON u.user_id = r.user_id AND r.rn = 1
            ORDER BY u.user_id
        """)
        assert len(result) == 4  # all users preserved by LEFT JOIN
        u1 = result[result["user_id"] == 1].iloc[0]
        assert u1["top_order"] == 70.0  # highest amount for user 1
        u2 = result[result["user_id"] == 2].iloc[0]
        assert u2["top_order"] == 90.0  # highest amount for user 2
        u3 = result[result["user_id"] == 3].iloc[0]
        assert u3["top_order"] == 100.0
        # David (user_id=4) has no orders → NULL
        u4 = result[result["user_id"] == 4].iloc[0]
        assert pd.isna(u4["top_order"])

    def test_left_join_preserves_unmatched(self, ranked_engine):
        """LEFT JOIN must keep rows from left table with no match on right."""
        result = ranked_engine.query("""
            SELECT u.user_id, u.name, o.amount
            FROM users u
            LEFT JOIN orders o ON u.user_id = o.user_id AND o.amount > 60
            ORDER BY u.user_id, o.amount
        """)
        # David (user_id=4) has no orders, must appear with NULL amount
        david = result[result["user_id"] == 4]
        assert len(david) >= 1
        assert pd.isna(david.iloc[0]["amount"])

    def test_inner_join_multiple_conditions(self, ranked_engine):
        """INNER JOIN ON col1 = col2 AND col3 > literal filters correctly."""
        result = ranked_engine.query("""
            SELECT u.name, o.amount
            FROM users u
            INNER JOIN orders o ON u.user_id = o.user_id AND o.amount > 50
            ORDER BY o.amount
        """)
        assert all(result["amount"] > 50)
        assert set(result["name"]).issubset({"Alice", "Bob", "Charlie"})

    def test_join_no_equi_keys_cross_then_filter(self):
        """ON with no equi-join keys falls back to cross join + filter."""
        a = pd.DataFrame({"x": [1, 2, 3]})
        b = pd.DataFrame({"y": [2, 4]})
        e = FrameQL({"a": a, "b": b})
        result = e.query("""
            SELECT a.x, b.y FROM a INNER JOIN b ON a.x < b.y
        """)
        assert len(result) > 0
        assert all(result["x"] < result["y"])

    def test_join_with_null_key(self, nulls_engine):
        """INNER JOIN on a column with NULLs excludes NULL-keyed rows."""
        result = nulls_engine.query("""
            SELECT l.id, r.label
            FROM left l
            INNER JOIN right r ON l.id = r.id
        """)
        # id=3 has no match in right; id=1 and id=2 match
        assert set(result["id"]) == {1, 2}

    def test_left_join_null_key_preserved(self, nulls_engine):
        """LEFT JOIN preserves all left rows even when left key has NULL."""
        result = nulls_engine.query("""
            SELECT l.id, l.val, r.label
            FROM left l
            LEFT JOIN right r ON l.id = r.id
        """)
        assert len(result) == 3
        # id=3 has no right match
        row3 = result[result["id"] == 3].iloc[0]
        assert pd.isna(row3["label"])

    def test_join_literal_string_condition(self, ranked_engine):
        """ON clause with string literal comparison is treated as residual filter."""
        # This just ensures no KeyError is raised
        result = ranked_engine.query("""
            WITH labeled AS (
                SELECT user_id, amount, 'order' AS kind FROM orders
            )
            SELECT u.name, l.amount
            FROM users u
            INNER JOIN labeled l ON u.user_id = l.user_id AND l.kind = 'order'
            ORDER BY u.name
        """)
        assert len(result) > 0
        assert all(result["amount"].notna())


# ─────────────────────────────────────────────────────────────
# 18. Window function edge cases (COUNT, MIN, MAX; no partition; etc.)
# ─────────────────────────────────────────────────────────────

class TestWindowFunctionEdgeCases:
    def test_count_over_partition(self, payments_engine):
        """COUNT(*) OVER (PARTITION BY user_id) gives per-partition row count."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   COUNT(*) OVER (PARTITION BY user_id) AS cnt
            FROM payments
        """)
        assert "cnt" in result.columns
        # user_id=1 has 2 payments, user_id=2 has 2, user_id=3 has 1
        assert all(result[result["user_id"] == 1]["cnt"] == 2)
        assert all(result[result["user_id"] == 2]["cnt"] == 2)
        assert all(result[result["user_id"] == 3]["cnt"] == 1)

    def test_min_over_partition(self, payments_engine):
        """MIN(amount) OVER (PARTITION BY user_id) gives per-partition minimum."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   MIN(amount) OVER (PARTITION BY user_id) AS min_amount
            FROM payments
        """)
        assert all(result[result["user_id"] == 1]["min_amount"] == 100.0)
        assert all(result[result["user_id"] == 2]["min_amount"] == 50.0)
        assert all(result[result["user_id"] == 3]["min_amount"] == 300.0)

    def test_max_over_partition(self, payments_engine):
        """MAX(amount) OVER (PARTITION BY user_id) gives per-partition maximum."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   MAX(amount) OVER (PARTITION BY user_id) AS max_amount
            FROM payments
        """)
        assert all(result[result["user_id"] == 1]["max_amount"] == 200.0)
        assert all(result[result["user_id"] == 2]["max_amount"] == 150.0)
        assert all(result[result["user_id"] == 3]["max_amount"] == 300.0)

    def test_row_number_no_partition(self, payments_engine):
        """ROW_NUMBER() with ORDER BY but no PARTITION BY numbers the whole result."""
        result = payments_engine.query("""
            SELECT payment_id, amount,
                   ROW_NUMBER() OVER (ORDER BY amount) AS rn
            FROM payments
        """)
        assert sorted(result["rn"].tolist()) == list(range(1, 6))
        # Smallest amount gets rn=1
        smallest_row = result[result["rn"] == 1].iloc[0]
        assert smallest_row["amount"] == result["amount"].min()

    def test_row_number_no_order(self, payments_engine):
        """ROW_NUMBER() with PARTITION BY but no ORDER BY assigns arbitrary order."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   ROW_NUMBER() OVER (PARTITION BY user_id) AS rn
            FROM payments
        """)
        # Each user's rows must have distinct row numbers starting at 1
        for uid, grp in result.groupby("user_id"):
            assert sorted(grp["rn"].tolist()) == list(range(1, len(grp) + 1))

    def test_row_number_preserves_original_order(self, payments_engine):
        """ROW_NUMBER computation must not globally reorder the output rows."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
            FROM payments
        """)
        # The result should have original DataFrame row order (not sorted by amount)
        # user_id values should NOT be fully sorted (original order: 1,1,2,2,3)
        assert list(result["user_id"]) == [1, 1, 2, 2, 3]

    def test_multiple_window_functions(self, payments_engine):
        """Multiple window functions in one SELECT are all computed correctly."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount) AS rn,
                   SUM(amount) OVER (PARTITION BY user_id) AS total,
                   MIN(amount) OVER (PARTITION BY user_id) AS mn
            FROM payments
        """)
        assert "rn" in result.columns
        assert "total" in result.columns
        assert "mn" in result.columns
        # Verify SUM for user_id=1
        assert all(result[result["user_id"] == 1]["total"] == 300.0)
        # Verify MIN for user_id=2 (50, 150 → min=50)
        assert all(result[result["user_id"] == 2]["mn"] == 50.0)

    def test_window_no_partition_no_order(self, payments_engine):
        """SUM() OVER () with no partition/order sums the entire column."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   SUM(amount) OVER () AS grand_total
            FROM payments
        """)
        assert all(result["grand_total"] == 800.0)

    def test_rank_with_ties(self):
        """RANK() assigns the same rank to tied rows and skips next ranks."""
        df = pd.DataFrame({"grp": [1, 1, 1, 1], "score": [90, 90, 80, 70]})
        e = FrameQL({"t": df})
        result = e.query("""
            SELECT grp, score,
                   RANK() OVER (PARTITION BY grp ORDER BY score DESC) AS rnk
            FROM t
        """)
        # Scores 90,90 tie at rank 1; 80 gets rank 3; 70 gets rank 4
        scores = result.sort_values("score", ascending=False)
        ranks = list(scores["rnk"])
        assert ranks[0] == ranks[1] == 1  # both 90s get rank 1
        assert ranks[2] == 3              # 80 skips rank 2
        assert ranks[3] == 4

    def test_dense_rank_with_ties(self):
        """DENSE_RANK() assigns same rank to ties but does NOT skip ranks."""
        df = pd.DataFrame({"grp": [1, 1, 1, 1], "score": [90, 90, 80, 70]})
        e = FrameQL({"t": df})
        result = e.query("""
            SELECT grp, score,
                   DENSE_RANK() OVER (PARTITION BY grp ORDER BY score DESC) AS drnk
            FROM t
        """)
        scores = result.sort_values("score", ascending=False)
        dranks = list(scores["drnk"])
        assert dranks[0] == dranks[1] == 1  # both 90s get rank 1
        assert dranks[2] == 2               # 80 gets rank 2 (not 3)
        assert dranks[3] == 3


# ─────────────────────────────────────────────────────────────
# 19. CTE edge cases
# ─────────────────────────────────────────────────────────────

class TestCTEEdgeCases:
    def test_cte_with_window_function(self, payments_engine):
        """CTE containing a window function is materialized correctly."""
        result = payments_engine.query("""
            WITH ranked AS (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY amount DESC
                       ) AS rn
                FROM payments
            )
            SELECT user_id, amount, rn FROM ranked
            WHERE rn = 1
            ORDER BY user_id
        """)
        # Each user's top payment should appear exactly once
        assert len(result) == 3
        assert list(result["rn"]) == [1, 1, 1]
        assert result[result["user_id"] == 1].iloc[0]["amount"] == 200.0
        assert result[result["user_id"] == 2].iloc[0]["amount"] == 150.0
        assert result[result["user_id"] == 3].iloc[0]["amount"] == 300.0

    def test_cte_used_in_join(self, ranked_engine):
        """CTE result is correctly joined on both table key and literal condition."""
        result = ranked_engine.query("""
            WITH top AS (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY amount DESC
                       ) AS rn
                FROM orders
            )
            SELECT u.name, t.amount
            FROM users u
            INNER JOIN top t ON u.user_id = t.user_id AND t.rn = 1
            ORDER BY u.name
        """)
        assert len(result) == 3  # users 1, 2, 3 have orders; David does not
        assert set(result["name"]) == {"Alice", "Bob", "Charlie"}
        # Alice's top order is 70
        alice = result[result["name"] == "Alice"].iloc[0]
        assert alice["amount"] == 70.0

    def test_nested_ctes(self, payments_engine):
        """Third-level CTE can reference second which references first."""
        result = payments_engine.query("""
            WITH totals AS (
                SELECT user_id, SUM(amount) AS total
                FROM payments GROUP BY user_id
            ),
            big AS (
                SELECT user_id, total FROM totals WHERE total >= 200
            ),
            named AS (
                SELECT user_id, total,
                       ROW_NUMBER() OVER (ORDER BY total DESC) AS rank_pos
                FROM big
            )
            SELECT user_id, total, rank_pos FROM named
            ORDER BY rank_pos
        """)
        assert len(result) >= 1
        # rank_pos=1 should be the highest total
        assert result.iloc[0]["total"] == result["total"].max()

    def test_cte_window_then_join_with_literal(self, ranked_engine):
        """Main test: CTE + ROW_NUMBER + LEFT JOIN ON col=col AND col=literal."""
        result = ranked_engine.query("""
            WITH ranked_orders AS (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY amount DESC
                       ) AS rn
                FROM orders
            )
            SELECT u.user_id, u.name, r.amount AS top_order
            FROM users u
            LEFT JOIN ranked_orders r
                ON u.user_id = r.user_id AND r.rn = 1
            ORDER BY u.user_id
        """)
        assert len(result) == 4
        u1 = result[result["user_id"] == 1].iloc[0]
        assert u1["top_order"] == 70.0
        u4 = result[result["user_id"] == 4].iloc[0]
        assert pd.isna(u4["top_order"])

    def test_cte_columns_accessible_in_outer_where(self, payments_engine):
        """CTE-computed columns (including window output) usable in outer WHERE."""
        result = payments_engine.query("""
            WITH w AS (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount) AS rn
                FROM payments
            )
            SELECT user_id, amount FROM w WHERE rn = 1
            ORDER BY user_id
        """)
        assert len(result) == 3
        # rn=1 within ORDER BY amount ASC means smallest amount per user
        assert result[result["user_id"] == 1].iloc[0]["amount"] == 100.0
        assert result[result["user_id"] == 2].iloc[0]["amount"] == 50.0


# ─────────────────────────────────────────────────────────────
# 20. Literal handling edge cases
# ─────────────────────────────────────────────────────────────

class TestLiteralHandling:
    def test_literal_in_join_on_not_treated_as_column(self, ranked_engine):
        """Integer literal in ON clause must never be treated as a column name."""
        # Should execute without KeyError
        result = ranked_engine.query("""
            WITH r AS (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
                FROM orders
            )
            SELECT u.user_id, r.amount
            FROM users u
            INNER JOIN r ON u.user_id = r.user_id AND r.rn = 1
        """)
        assert len(result) == 3  # users 1, 2, 3

    def test_string_literal_in_join_on(self):
        """String literal in ON clause filters correctly (not treated as column)."""
        t1 = pd.DataFrame({"id": [1, 2], "kind": ["a", "b"]})
        # t2 row id=1 has kind='a', t2 row id=2 has kind='b'
        t2 = pd.DataFrame({"id": [1, 2], "kind": ["a", "b"]})
        e = FrameQL({"t1": t1, "t2": t2})
        result = e.query("""
            SELECT t1.id FROM t1
            INNER JOIN t2 ON t1.id = t2.id AND t2.kind = 'a'
        """)
        # Only the row where t2.kind = 'a' (id=1) passes
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1

    def test_in_with_literals(self, payments_engine):
        """IN clause with literals works correctly."""
        result = payments_engine.query("""
            SELECT user_id FROM payments
            WHERE user_id IN (1, 3)
        """)
        assert set(result["user_id"]) == {1, 3}

    def test_comparison_with_zero(self):
        """Comparison with literal 0 does not confuse column resolution."""
        df = pd.DataFrame({"x": [0, 1, 2], "y": [10, 20, 30]})
        e = FrameQL({"t": df})
        result = e.query("SELECT x, y FROM t WHERE x = 0")
        assert len(result) == 1
        assert result.iloc[0]["y"] == 10


# ─────────────────────────────────────────────────────────────
# 21. Complex combined queries
# ─────────────────────────────────────────────────────────────

class TestComplexCombined:
    def test_top_n_per_group_full_pipeline(self, ranked_engine):
        """CTE + window + JOIN + ORDER BY all chained together correctly."""
        result = ranked_engine.query("""
            WITH ranked AS (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY amount DESC
                       ) AS rn
                FROM orders
            )
            SELECT u.user_id, u.name, r.amount AS best
            FROM users u
            LEFT JOIN ranked r ON u.user_id = r.user_id AND r.rn = 1
            ORDER BY u.user_id ASC
        """)
        # All 4 users appear; David is NULL
        assert len(result) == 4
        assert list(result["user_id"]) == [1, 2, 3, 4]
        assert result.iloc[0]["best"] == 70.0  # Alice's best
        assert pd.isna(result.iloc[3]["best"])  # David has no orders

    def test_cte_with_filter_and_window(self, payments_engine):
        """CTE filters data first, then window function runs on filtered set."""
        result = payments_engine.query("""
            WITH paid AS (
                SELECT user_id, amount
                FROM payments
                WHERE status = 'paid'
            ),
            ranked_paid AS (
                SELECT user_id, amount,
                       RANK() OVER (ORDER BY amount DESC) AS rnk
                FROM paid
            )
            SELECT user_id, amount, rnk
            FROM ranked_paid
            WHERE rnk = 1
        """)
        assert len(result) >= 1
        assert result.iloc[0]["rnk"] == 1
        assert result.iloc[0]["amount"] == result["amount"].max()

    def test_window_then_group_by_not_allowed_but_cte_workaround(self, payments_engine):
        """CTE materializes window results; outer query aggregates them."""
        result = payments_engine.query("""
            WITH w AS (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
                FROM payments
            )
            SELECT user_id, SUM(amount) AS total
            FROM w
            WHERE rn <= 1
            GROUP BY user_id
            ORDER BY user_id
        """)
        assert len(result) == 3
        # Each user contributes their top-1 payment
        assert result[result["user_id"] == 1].iloc[0]["total"] == 200.0
        assert result[result["user_id"] == 2].iloc[0]["total"] == 150.0
        assert result[result["user_id"] == 3].iloc[0]["total"] == 300.0

    def test_multiple_ctes_with_window_and_join(self, ranked_engine):
        """Two CTEs: one with window, one with aggregation, joined together."""
        result = ranked_engine.query("""
            WITH best AS (
                SELECT user_id, amount AS best_amount,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
                FROM orders
            ),
            totals AS (
                SELECT user_id, SUM(amount) AS total_amount
                FROM orders
                GROUP BY user_id
            )
            SELECT b.user_id, b.best_amount, t.total_amount
            FROM best b
            INNER JOIN totals t ON b.user_id = t.user_id AND b.rn = 1
            ORDER BY b.user_id
        """)
        assert len(result) == 3
        # user_id=1: best=70, total=150 (50+30+70)
        u1 = result[result["user_id"] == 1].iloc[0]
        assert u1["best_amount"] == 70.0
        assert u1["total_amount"] == 150.0

    def test_ranking_query_with_having(self, payments_engine):
        """GROUP BY + HAVING to find heavy-spending users, then rank by total."""
        result = payments_engine.query("""
            WITH heavy AS (
                SELECT user_id, SUM(amount) AS total
                FROM payments
                GROUP BY user_id
                HAVING SUM(amount) >= 200
            )
            SELECT user_id, total,
                   RANK() OVER (ORDER BY total DESC) AS spend_rank
            FROM heavy
            ORDER BY spend_rank
        """)
        assert len(result) >= 1
        totals = list(result["total"])
        assert totals == sorted(totals, reverse=True)
        assert result.iloc[0]["spend_rank"] == 1
