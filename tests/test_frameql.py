"""
FrameQL regression tests.

Run with:  pytest tests/test_frameql.py -v
"""

import pytest
import pandas as pd
import numpy as np
from frameql import FrameQL


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


# ═════════════════════════════════════════════════════════════
# 22. LEAD / LAG window functions
# ═════════════════════════════════════════════════════════════

class TestLeadLag:
    def test_lag_basic(self, payments_engine):
        """LAG returns the previous row's value within a partition."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   LAG(amount) OVER (PARTITION BY user_id ORDER BY amount) AS prev_amount
            FROM payments
        """)
        assert "prev_amount" in result.columns
        # For user_id=1 (amounts 100, 200 sorted ASC):
        # first row  → prev=NULL, second row → prev=100
        u1 = result[result["user_id"] == 1].sort_values("amount")
        assert pd.isna(u1.iloc[0]["prev_amount"])
        assert u1.iloc[1]["prev_amount"] == 100.0

    def test_lead_basic(self, payments_engine):
        """LEAD returns the next row's value within a partition."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   LEAD(amount) OVER (PARTITION BY user_id ORDER BY amount) AS next_amount
            FROM payments
        """)
        assert "next_amount" in result.columns
        u1 = result[result["user_id"] == 1].sort_values("amount")
        assert u1.iloc[0]["next_amount"] == 200.0   # 100 → 200
        assert pd.isna(u1.iloc[1]["next_amount"])   # 200 → NULL

    def test_lag_with_offset(self, payments_engine):
        """LAG(col, offset) skips multiple rows back."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   LAG(amount, 2) OVER (PARTITION BY user_id ORDER BY amount) AS two_back
            FROM payments
        """)
        # user_id=1 has only 2 rows; offset=2 means always NULL
        u1 = result[result["user_id"] == 1].sort_values("amount")
        assert all(pd.isna(u1["two_back"]))

    def test_lag_with_default(self, payments_engine):
        """LAG(col, 1, default) substitutes default when out of bounds."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   LAG(amount, 1, 0) OVER (PARTITION BY user_id ORDER BY amount) AS prev
            FROM payments
        """)
        # First row per partition should have default=0 instead of NULL
        u1 = result[result["user_id"] == 1].sort_values("amount")
        assert u1.iloc[0]["prev"] == 0

    def test_lead_with_default(self, payments_engine):
        """LEAD(col, 1, -1) substitutes -1 when out of bounds."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   LEAD(amount, 1, -1) OVER (PARTITION BY user_id ORDER BY amount) AS nxt
            FROM payments
        """)
        u1 = result[result["user_id"] == 1].sort_values("amount")
        assert u1.iloc[-1]["nxt"] == -1

    def test_lag_no_partition(self, payments_engine):
        """LAG without PARTITION BY operates over the entire result set."""
        result = payments_engine.query("""
            SELECT payment_id, amount,
                   LAG(amount) OVER (ORDER BY amount) AS prev
            FROM payments
        """)
        sorted_result = result.sort_values("amount").reset_index(drop=True)
        assert pd.isna(sorted_result.iloc[0]["prev"])   # first row has no predecessor
        assert sorted_result.iloc[1]["prev"] == sorted_result.iloc[0]["amount"]

    def test_lead_no_order(self, payments_engine):
        """LEAD without ORDER BY assigns arbitrary but non-crashing order."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   LEAD(amount) OVER (PARTITION BY user_id) AS nxt
            FROM payments
        """)
        assert "nxt" in result.columns
        assert len(result) == 5

    def test_lag_single_row_partition(self):
        """Single-row partition always returns NULL for LAG."""
        df = pd.DataFrame({"grp": [1, 2, 3], "val": [10, 20, 30]})
        e = FrameQL({"t": df})
        result = e.query("""
            SELECT grp, val,
                   LAG(val) OVER (PARTITION BY grp ORDER BY val) AS prev
            FROM t
        """)
        assert all(pd.isna(result["prev"]))

    def test_lag_preserves_original_row_order(self, payments_engine):
        """LAG computation must not permanently reorder the output rows."""
        result = payments_engine.query("""
            SELECT user_id, amount,
                   LAG(amount) OVER (PARTITION BY user_id ORDER BY amount DESC) AS prev
            FROM payments
        """)
        # Original data order: user_id = 1, 1, 2, 2, 3
        assert list(result["user_id"]) == [1, 1, 2, 2, 3]

    def test_lag_repeated_order_values(self):
        """Repeated ORDER BY values produce stable but implementation-defined LAG."""
        df = pd.DataFrame({
            "grp": [1, 1, 1],
            "score": [10, 10, 20],  # two tied rows at 10
        })
        e = FrameQL({"t": df})
        result = e.query("""
            SELECT grp, score,
                   LAG(score) OVER (PARTITION BY grp ORDER BY score) AS prev
            FROM t
        """)
        # At least one row should have prev = NULL (the first-in-partition row)
        assert result["prev"].isna().sum() >= 1

    def test_lead_lag_combined(self):
        """Both LEAD and LAG in the same SELECT are computed independently."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
        e = FrameQL({"t": df})
        result = e.query("""
            SELECT id, val,
                   LAG(val)  OVER (ORDER BY id) AS prev_val,
                   LEAD(val) OVER (ORDER BY id) AS next_val
            FROM t
        """)
        result = result.sort_values("id").reset_index(drop=True)
        assert pd.isna(result.iloc[0]["prev_val"])
        assert result.iloc[1]["prev_val"] == 10
        assert result.iloc[-1]["next_val"] is None or pd.isna(result.iloc[-1]["next_val"])
        assert result.iloc[-2]["next_val"] == 40

    def test_lag_in_cte(self, payments_engine):
        """LAG inside a CTE is materialized and accessible in outer query."""
        result = payments_engine.query("""
            WITH lagged AS (
                SELECT user_id, amount,
                       LAG(amount) OVER (PARTITION BY user_id ORDER BY amount) AS prev
                FROM payments
            )
            SELECT user_id, amount, prev
            FROM lagged
            WHERE prev IS NOT NULL
            ORDER BY user_id, amount
        """)
        assert len(result) > 0
        assert result["prev"].notna().all()


# ═════════════════════════════════════════════════════════════
# 23. Tuple / multi-column IN
# ═════════════════════════════════════════════════════════════

class TestTupleIn:
    def test_tuple_in_subquery(self, engine):
        """(user_id, amount) IN (SELECT …) returns matching rows."""
        result = engine.query("""
            SELECT user_id, name FROM users u
            WHERE (u.user_id) IN (SELECT user_id FROM orders)
        """)
        assert set(result["user_id"]) == {1, 2, 3}

    def test_two_column_tuple_in_subquery(self):
        """(a, b) IN (SELECT x, y …) performs tuple membership correctly."""
        left = pd.DataFrame({"id": [1, 2, 3], "code": ["A", "B", "C"]})
        right = pd.DataFrame({"id": [1, 3], "code": ["A", "C"]})
        e = FrameQL({"left": left, "right": right})
        result = e.query("""
            SELECT l.id FROM left l
            WHERE (l.id, l.code) IN (SELECT r.id, r.code FROM right r)
        """)
        assert set(result["id"]) == {1, 3}

    def test_tuple_in_excludes_non_matching(self):
        """Rows NOT in the subquery result are excluded."""
        left = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        right = pd.DataFrame({"x": [1, 2], "y": [10, 99]})  # (2,20) not in right
        e = FrameQL({"l": left, "r": right})
        result = e.query("""
            SELECT a FROM l
            WHERE (a, b) IN (SELECT x, y FROM r)
        """)
        # (1,10) → in, (2,20) → NOT in (right has (2,99)), (3,30) → NOT in
        assert list(result["a"]) == [1]

    def test_tuple_in_literal_list(self):
        """(a, b) IN ((v1, v2), (v3, v4)) with inline tuple literals."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        e = FrameQL({"t": df})
        result = e.query("""
            SELECT x, y FROM t
            WHERE (x, y) IN ((1, 'a'), (3, 'c'))
        """)
        assert set(result["x"]) == {1, 3}

    def test_tuple_in_null_on_left_no_match(self):
        """NULL on the left side of a tuple never matches."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [10, 20, 30]})
        right = pd.DataFrame({"x": [1, None, 3], "y": [10, 20, 30]})
        e = FrameQL({"l": df, "r": right})
        result = e.query("""
            SELECT a FROM l
            WHERE (a, b) IN (SELECT x, y FROM r)
        """)
        # NULL in left → no match; 1 and 3 match
        assert set(result["a"].dropna()) == {1, 3}
        assert result["a"].isna().sum() == 0  # NULL row excluded

    def test_tuple_in_null_on_right_no_match(self):
        """NULL in the right-side tuple means that row never contributes a match."""
        df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        right = pd.DataFrame({"x": [1, None], "y": [10, 20]})
        e = FrameQL({"l": df, "r": right})
        result = e.query("""
            SELECT a FROM l
            WHERE (a, b) IN (SELECT x, y FROM r)
        """)
        assert list(result["a"]) == [1]  # (2,20) doesn't match because right tuple has NULL

    def test_tuple_in_duplicates_in_subquery(self):
        """Duplicate tuples in subquery result don't affect membership semantics."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        right = pd.DataFrame({"x": [1, 1, 1], "y": [10, 10, 10]})  # duplicates
        e = FrameQL({"l": df, "r": right})
        result = e.query("""
            SELECT a FROM l
            WHERE (a, b) IN (SELECT x, y FROM r)
        """)
        assert list(result["a"]) == [1]

    def test_tuple_in_column_count_mismatch_raises(self):
        """Mismatched column counts between tuple and subquery raise ValueError."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        right = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        e = FrameQL({"l": df, "r": right})
        with pytest.raises(ValueError, match="column count mismatch"):
            e.query("""
                SELECT a FROM l
                WHERE (a, b) IN (SELECT x, y, z FROM r)
            """)


# ═════════════════════════════════════════════════════════════
# 24. Set operations (UNION / UNION ALL / INTERSECT / EXCEPT)
# ═════════════════════════════════════════════════════════════

@pytest.fixture
def set_engine():
    a = pd.DataFrame({"x": [1, 2, 3, 2], "y": ["a", "b", "c", "b"]})
    b = pd.DataFrame({"x": [2, 3, 4], "y": ["b", "c", "d"]})
    nums = pd.DataFrame({"n": [1, 2, 3, 4, 5]})
    return FrameQL({"a": a, "b": b, "nums": nums})


class TestSetOperations:
    def test_union_deduplicates(self, set_engine):
        """UNION removes duplicates across both sides."""
        result = set_engine.query("""
            SELECT x, y FROM a
            UNION
            SELECT x, y FROM b
        """)
        # Unique (x,y) pairs: (1,a),(2,b),(3,c),(4,d)
        assert len(result) == 4
        assert set(map(tuple, result.values.tolist())) == {(1,"a"),(2,"b"),(3,"c"),(4,"d")}

    def test_union_all_keeps_duplicates(self, set_engine):
        """UNION ALL preserves all rows including duplicates."""
        result = set_engine.query("""
            SELECT x FROM a
            UNION ALL
            SELECT x FROM b
        """)
        # a has 4 rows, b has 3 rows → 7 rows total
        assert len(result) == 7

    def test_union_deduplicates_within_side(self, set_engine):
        """UNION also deduplicates rows that appear multiple times within one side."""
        result = set_engine.query("""
            SELECT x, y FROM a
            UNION
            SELECT x, y FROM a
        """)
        # Unique rows in a: (1,a),(2,b),(3,c)
        assert len(result) == 3

    def test_intersect_returns_common_rows(self, set_engine):
        """INTERSECT returns only rows present in both queries."""
        result = set_engine.query("""
            SELECT x, y FROM a
            INTERSECT
            SELECT x, y FROM b
        """)
        assert set(map(tuple, result.values.tolist())) == {(2,"b"),(3,"c")}

    def test_intersect_empty(self, set_engine):
        """INTERSECT returns empty when no rows are common."""
        result = set_engine.query("""
            SELECT x FROM a WHERE x = 1
            INTERSECT
            SELECT x FROM b WHERE x = 4
        """)
        assert len(result) == 0

    def test_except_removes_right_rows(self, set_engine):
        """EXCEPT returns rows in left that are not in right."""
        result = set_engine.query("""
            SELECT x, y FROM a
            EXCEPT
            SELECT x, y FROM b
        """)
        # a has (1,a),(2,b),(3,c); b has (2,b),(3,c),(4,d) → only (1,a) remains
        assert set(map(tuple, result.values.tolist())) == {(1,"a")}

    def test_except_full_subtraction(self, set_engine):
        """EXCEPT with entire right side present in left → empty result."""
        result = set_engine.query("""
            SELECT x FROM b
            EXCEPT
            SELECT x FROM a
        """)
        # b has x=2,3,4; a has x=1,2,3 → only 4 not in a
        xs = set(result.values.flatten().tolist())
        assert xs == {4}

    def test_union_with_order_by(self, set_engine):
        """ORDER BY applies to the final UNION result."""
        result = set_engine.query("""
            SELECT x FROM a
            UNION
            SELECT x FROM b
            ORDER BY x DESC
        """)
        xs = list(result["x"])
        assert xs == sorted(xs, reverse=True)

    def test_union_with_limit(self, set_engine):
        """LIMIT applies after UNION deduplication."""
        result = set_engine.query("""
            SELECT x FROM a
            UNION
            SELECT x FROM b
            ORDER BY x
            LIMIT 2
        """)
        assert len(result) == 2
        assert list(result["x"]) == [1, 2]

    def test_union_position_based_columns(self):
        """Column names from left side take precedence; position-based matching."""
        left = pd.DataFrame({"col_a": [1, 2]})
        right = pd.DataFrame({"col_b": [3, 4]})
        e = FrameQL({"l": left, "r": right})
        result = e.query("SELECT col_a FROM l UNION ALL SELECT col_b FROM r")
        # Result columns should be named after left side
        assert "col_a" in result.columns
        assert set(result["col_a"]) == {1, 2, 3, 4}

    def test_union_column_count_mismatch_raises(self):
        """Column count mismatch between UNION sides raises ValueError."""
        left = pd.DataFrame({"a": [1], "b": [2]})
        right = pd.DataFrame({"x": [3]})
        e = FrameQL({"l": left, "r": right})
        with pytest.raises(ValueError, match="column count mismatch"):
            e.query("SELECT a, b FROM l UNION SELECT x FROM r")

    def test_union_with_null(self):
        """UNION correctly deduplicates rows containing NULL values."""
        df = pd.DataFrame({"v": [1, None, 3]})
        e = FrameQL({"t": df})
        result = e.query("SELECT v FROM t UNION SELECT v FROM t")
        # Should deduplicate; NULL appears once
        assert len(result) == 3

    def test_intersect_with_null(self):
        """INTERSECT treats NULL = NULL (set semantics)."""
        a = pd.DataFrame({"v": [1, None, 3]})
        b = pd.DataFrame({"v": [None, 3, 4]})
        e = FrameQL({"a": a, "b": b})
        result = e.query("SELECT v FROM a INTERSECT SELECT v FROM b")
        # NULL and 3 are in both
        assert len(result) == 2

    def test_except_with_null(self):
        """EXCEPT correctly excludes rows with matching NULL."""
        a = pd.DataFrame({"v": [1, None, 3]})
        b = pd.DataFrame({"v": [None]})
        e = FrameQL({"a": a, "b": b})
        result = e.query("SELECT v FROM a EXCEPT SELECT v FROM b")
        # NULL removed by EXCEPT
        non_null = result["v"].dropna()
        assert set(non_null) == {1, 3}
        assert result["v"].isna().sum() == 0

    def test_chained_union(self, set_engine):
        """Chained UNION operations work correctly."""
        result = set_engine.query("""
            SELECT x FROM a WHERE x = 1
            UNION
            SELECT x FROM a WHERE x = 2
            UNION
            SELECT x FROM b WHERE x = 4
        """)
        assert set(result["x"]) == {1, 2, 4}

    def test_cte_with_union(self, payments_engine):
        """CTE body may itself be a UNION."""
        result = payments_engine.query("""
            WITH combined AS (
                SELECT user_id, amount FROM payments WHERE status = 'paid'
                UNION ALL
                SELECT user_id, amount FROM payments WHERE status = 'pending'
            )
            SELECT COUNT(*) AS total_rows FROM combined
        """)
        assert result.iloc[0]["total_rows"] == 5  # all 5 rows (paid + pending)

    def test_union_inside_subquery(self, set_engine):
        """UNION result used as a subquery in FROM clause."""
        result = set_engine.query("""
            SELECT x FROM (
                SELECT x FROM a WHERE x < 3
                UNION
                SELECT x FROM b WHERE x > 2
            ) sub
            ORDER BY x
        """)
        xs = sorted(set(result["x"].tolist()))
        assert 1 in xs
        assert 3 in xs
        assert 4 in xs


# ═════════════════════════════════════════════════════════════
# 25. Combined: LEAD/LAG + Tuple IN + Set ops + CTEs
# ═════════════════════════════════════════════════════════════

class TestCombinedNewFeatures:
    def test_lag_in_cte_then_set_op(self, payments_engine):
        """CTE with LAG, then UNION result with another query."""
        result = payments_engine.query("""
            WITH lagged AS (
                SELECT user_id, amount,
                       LAG(amount) OVER (PARTITION BY user_id ORDER BY amount) AS prev
                FROM payments
            )
            SELECT user_id FROM lagged WHERE prev IS NOT NULL
            UNION
            SELECT user_id FROM lagged WHERE prev IS NULL
        """)
        # UNION of complementary sets = all distinct user_ids
        assert set(result["user_id"]) == {1, 2, 3}

    def test_tuple_in_with_cte(self, engine):
        """Tuple IN where right side is a CTE-backed subquery."""
        result = engine.query("""
            WITH valid_pairs AS (
                SELECT user_id, amount FROM orders WHERE amount > 20
            )
            SELECT o.user_id, o.amount
            FROM orders o
            WHERE (o.user_id, o.amount) IN (SELECT user_id, amount FROM valid_pairs)
            ORDER BY o.user_id, o.amount
        """)
        assert all(result["amount"] > 20)

    def test_set_op_with_window_in_subquery(self, payments_engine):
        """Window function inside a subquery used in a UNION branch."""
        result = payments_engine.query("""
            SELECT user_id, amount FROM (
                SELECT user_id, amount,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
                FROM payments
            ) ranked WHERE rn = 1
            UNION ALL
            SELECT user_id, amount FROM payments WHERE amount = 300
        """)
        assert len(result) > 0
        assert all(result["amount"].notna())

    def test_lead_then_intersect(self, payments_engine):
        """Rows selected by LEAD condition intersected with another filter."""
        result = payments_engine.query("""
            WITH with_lead AS (
                SELECT user_id, amount,
                       LEAD(amount) OVER (PARTITION BY user_id ORDER BY amount) AS nxt
                FROM payments
            )
            SELECT user_id FROM with_lead WHERE nxt IS NOT NULL
            INTERSECT
            SELECT user_id FROM payments WHERE status = 'paid'
        """)
        assert len(result) >= 1
        assert result["user_id"].notna().all()

    def test_except_after_window_filter(self, payments_engine):
        """EXCEPT removes window-function-filtered rows from a base set."""
        result = payments_engine.query("""
            SELECT user_id FROM payments
            EXCEPT
            SELECT user_id FROM (
                SELECT user_id,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
                FROM payments
            ) w WHERE rn = 1 AND user_id = 1
        """)
        # user_id=1's top row is excluded, but user_id=2,3 remain
        # (EXCEPT deduplicates, so user_id=1 may still appear from other rows
        #  unless we're only looking at distinct user_ids)
        assert len(result) >= 1
