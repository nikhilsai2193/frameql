"""
FrameQL DML tests: INSERT, UPDATE, DELETE.

Run with:  pytest tests/test_dml.py -v
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
        "user_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [90.0, 80.0, 70.0],
    })
    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "user_id": [1, 1, 2, 3],
        "amount": [50.0, 30.0, 20.0, 100.0],
        "status": ["paid", "paid", "pending", "paid"],
    })
    return FrameQL({"users": users, "orders": orders})


@pytest.fixture
def empty_engine():
    empty = pd.DataFrame({"id": pd.Series([], dtype=int), "val": pd.Series([], dtype=float)})
    return FrameQL({"t": empty})


# ─────────────────────────────────────────────────────────────
# INSERT tests
# ─────────────────────────────────────────────────────────────

class TestInsert:
    def test_insert_basic_values(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 60.0)")
        assert len(engine.tables["users"]) == 4
        last = engine.tables["users"].iloc[-1]
        assert last["user_id"] == 4
        assert last["name"] == "Dave"
        assert last["score"] == 60.0

    def test_insert_multiple_rows(self, engine):
        engine.execute(
            "INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 60.0), (5, 'Eve', 55.0)"
        )
        assert len(engine.tables["users"]) == 5
        names = list(engine.tables["users"]["name"])
        assert "Dave" in names
        assert "Eve" in names

    def test_insert_without_column_list(self, engine):
        engine.execute("INSERT INTO users VALUES (4, 'Dave', 60.0)")
        assert len(engine.tables["users"]) == 4
        assert engine.tables["users"].iloc[-1]["name"] == "Dave"

    def test_insert_with_null_value(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, NULL, 60.0)")
        row = engine.tables["users"].iloc[-1]
        assert pd.isna(row["name"])
        assert row["score"] == 60.0

    def test_insert_partial_columns_fills_nan(self, engine):
        engine.execute("INSERT INTO users (user_id, name) VALUES (4, 'Dave')")
        row = engine.tables["users"].iloc[-1]
        assert row["user_id"] == 4
        assert row["name"] == "Dave"
        assert pd.isna(row["score"])  # missing column → NaN

    def test_insert_integer_arithmetic_in_values(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (2 + 2, 'Dave', 10 * 5)")
        row = engine.tables["users"].iloc[-1]
        assert row["user_id"] == 4
        assert row["score"] == 50.0

    def test_insert_negative_value(self, engine):
        engine.execute("INSERT INTO orders (order_id, user_id, amount, status) VALUES (200, 1, -5.0, 'refund')")
        row = engine.tables["orders"].iloc[-1]
        assert row["amount"] == -5.0

    def test_insert_does_not_modify_other_tables(self, engine):
        original_orders = engine.tables["orders"].copy()
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 60.0)")
        pd.testing.assert_frame_equal(engine.tables["orders"], original_orders)

    def test_insert_preserves_existing_rows(self, engine):
        original_alice = engine.tables["users"].iloc[0].copy()
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 60.0)")
        assert engine.tables["users"].iloc[0]["name"] == original_alice["name"]
        assert engine.tables["users"].iloc[0]["score"] == original_alice["score"]

    def test_insert_from_select(self, engine):
        # Insert users with score > 80 as new "archive" table entries
        archive_df = pd.DataFrame({"user_id": pd.Series([], dtype=int), "name": pd.Series([], dtype=str)})
        engine.tables["archive"] = archive_df
        engine.execute("INSERT INTO archive (user_id, name) SELECT user_id, name FROM users WHERE score > 80")
        assert len(engine.tables["archive"]) == 1
        assert engine.tables["archive"].iloc[0]["name"] == "Alice"

    def test_insert_from_select_all_rows(self, engine):
        backup = pd.DataFrame({
            "user_id": pd.Series([], dtype=int),
            "name": pd.Series([], dtype=str),
            "score": pd.Series([], dtype=float),
        })
        engine.tables["backup"] = backup
        engine.execute("INSERT INTO backup (user_id, name, score) SELECT user_id, name, score FROM users")
        assert len(engine.tables["backup"]) == 3

    def test_insert_from_aggregate_select(self, engine):
        agg_table = pd.DataFrame({"user_id": pd.Series([], dtype=int), "total": pd.Series([], dtype=float)})
        engine.tables["user_totals"] = agg_table
        engine.execute(
            "INSERT INTO user_totals (user_id, total) "
            "SELECT user_id, SUM(amount) FROM orders GROUP BY user_id"
        )
        assert len(engine.tables["user_totals"]) == 3

    def test_insert_into_empty_table(self, empty_engine):
        empty_engine.execute("INSERT INTO t (id, val) VALUES (1, 42.0)")
        assert len(empty_engine.tables["t"]) == 1
        assert empty_engine.tables["t"].iloc[0]["val"] == 42.0

    def test_insert_nonexistent_table_raises(self, engine):
        with pytest.raises(ValueError, match="does not exist"):
            engine.execute("INSERT INTO nosuchname (x) VALUES (1)")

    def test_insert_column_count_mismatch_in_select_raises(self, engine):
        backup = pd.DataFrame({"user_id": pd.Series([], dtype=int), "name": pd.Series([], dtype=str)})
        engine.tables["backup"] = backup
        with pytest.raises(ValueError, match="column count mismatch"):
            engine.execute("INSERT INTO backup (user_id, name) SELECT user_id, name, score FROM users")

    def test_insert_then_query(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 45.0)")
        result = engine.query("SELECT name FROM users WHERE score < 50")
        assert "Dave" in list(result["name"])

    def test_insert_row_count_increments(self, engine):
        for i in range(5):
            engine.execute(f"INSERT INTO users (user_id, name, score) VALUES ({100+i}, 'User{i}', {i * 10.0})")
        assert len(engine.tables["users"]) == 8  # 3 original + 5 new

    def test_insert_string_with_spaces(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'John Doe', 88.0)")
        row = engine.tables["users"].iloc[-1]
        assert row["name"] == "John Doe"


# ─────────────────────────────────────────────────────────────
# UPDATE tests
# ─────────────────────────────────────────────────────────────

class TestUpdate:
    def test_update_single_column(self, engine):
        engine.execute("UPDATE users SET score = 99.0 WHERE user_id = 1")
        alice = engine.tables["users"][engine.tables["users"]["user_id"] == 1].iloc[0]
        assert alice["score"] == 99.0

    def test_update_does_not_affect_other_rows(self, engine):
        engine.execute("UPDATE users SET score = 99.0 WHERE user_id = 1")
        bob = engine.tables["users"][engine.tables["users"]["user_id"] == 2].iloc[0]
        assert bob["score"] == 80.0
        charlie = engine.tables["users"][engine.tables["users"]["user_id"] == 3].iloc[0]
        assert charlie["score"] == 70.0

    def test_update_all_rows_no_where(self, engine):
        engine.execute("UPDATE users SET score = 0.0")
        assert all(engine.tables["users"]["score"] == 0.0)

    def test_update_arithmetic_expression(self, engine):
        engine.execute("UPDATE orders SET amount = amount * 1.1 WHERE status = 'paid'")
        paid = engine.tables["orders"][engine.tables["orders"]["status"] == "paid"]
        assert all(abs(paid["amount"] - paid["amount"].round(1)) < 0.001)
        # order 101: 50*1.1=55, order 102: 30*1.1=33, order 104: 100*1.1=110
        row101 = engine.tables["orders"][engine.tables["orders"]["order_id"] == 101].iloc[0]
        assert abs(row101["amount"] - 55.0) < 0.001
        pending_row = engine.tables["orders"][engine.tables["orders"]["order_id"] == 103].iloc[0]
        assert pending_row["amount"] == 20.0  # unchanged

    def test_update_string_value(self, engine):
        engine.execute("UPDATE orders SET status = 'cancelled' WHERE order_id = 103")
        row = engine.tables["orders"][engine.tables["orders"]["order_id"] == 103].iloc[0]
        assert row["status"] == "cancelled"

    def test_update_multiple_columns(self, engine):
        engine.execute("UPDATE orders SET amount = 999.0, status = 'special' WHERE order_id = 101")
        row = engine.tables["orders"][engine.tables["orders"]["order_id"] == 101].iloc[0]
        assert row["amount"] == 999.0
        assert row["status"] == "special"

    def test_update_with_case_expression(self, engine):
        engine.execute("""
            UPDATE orders SET status = CASE
                WHEN amount >= 50 THEN 'big'
                ELSE 'small'
            END
        """)
        row_big = engine.tables["orders"][engine.tables["orders"]["order_id"] == 101].iloc[0]
        assert row_big["status"] == "big"
        row_small = engine.tables["orders"][engine.tables["orders"]["order_id"] == 103].iloc[0]
        assert row_small["status"] == "small"

    def test_update_no_matching_rows_noop(self, engine):
        original = engine.tables["orders"].copy()
        engine.execute("UPDATE orders SET amount = 999.0 WHERE user_id = 999")
        pd.testing.assert_frame_equal(engine.tables["orders"], original)

    def test_update_to_null(self, engine):
        engine.execute("UPDATE users SET score = NULL WHERE user_id = 3")
        row = engine.tables["users"][engine.tables["users"]["user_id"] == 3].iloc[0]
        assert pd.isna(row["score"])

    def test_update_with_subquery_scalar(self, engine):
        # Set all amounts to the overall average
        engine.execute("UPDATE orders SET amount = (SELECT AVG(amount) FROM orders)")
        avg = 50.0  # (50+30+20+100)/4 = 50
        assert all(abs(engine.tables["orders"]["amount"] - avg) < 0.001)

    def test_update_with_in_subquery(self, engine):
        engine.execute("""
            UPDATE orders SET status = 'vip'
            WHERE user_id IN (SELECT user_id FROM users WHERE score > 85)
        """)
        # Alice (score=90) is the only one with score > 85; user_id=1
        vip_rows = engine.tables["orders"][engine.tables["orders"]["status"] == "vip"]
        assert all(vip_rows["user_id"] == 1)

    def test_update_nonexistent_table_raises(self, engine):
        with pytest.raises(ValueError, match="does not exist"):
            engine.execute("UPDATE nosuchname SET x = 1")

    def test_update_then_query(self, engine):
        engine.execute("UPDATE orders SET amount = 0.0 WHERE user_id = 2")
        result = engine.query("SELECT SUM(amount) AS total FROM orders")
        # original: 50+30+20+100=200; user_id=2 had 20 → zeroed
        assert result.iloc[0]["total"] == 180.0

    def test_update_preserves_row_count(self, engine):
        original_count = len(engine.tables["orders"])
        engine.execute("UPDATE orders SET amount = 1.0")
        assert len(engine.tables["orders"]) == original_count

    def test_update_empty_table_noop(self, empty_engine):
        empty_engine.execute("UPDATE t SET val = 99.0")
        assert len(empty_engine.tables["t"]) == 0

    def test_update_compound_condition(self, engine):
        engine.execute("UPDATE orders SET amount = 0.0 WHERE status = 'paid' AND amount < 40")
        # order 102: paid, amount=30 → 0.0
        row102 = engine.tables["orders"][engine.tables["orders"]["order_id"] == 102].iloc[0]
        assert row102["amount"] == 0.0
        # order 101: paid, amount=50 → unchanged
        row101 = engine.tables["orders"][engine.tables["orders"]["order_id"] == 101].iloc[0]
        assert row101["amount"] == 50.0


# ─────────────────────────────────────────────────────────────
# DELETE tests
# ─────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_conditional(self, engine):
        engine.execute("DELETE FROM orders WHERE user_id = 2")
        assert all(engine.tables["orders"]["user_id"] != 2)

    def test_delete_all_rows(self, engine):
        engine.execute("DELETE FROM orders")
        assert len(engine.tables["orders"]) == 0

    def test_delete_all_preserves_schema(self, engine):
        engine.execute("DELETE FROM orders")
        df = engine.tables["orders"]
        assert list(df.columns) == ["order_id", "user_id", "amount", "status"]

    def test_delete_no_match_noop(self, engine):
        original = engine.tables["orders"].copy()
        engine.execute("DELETE FROM orders WHERE order_id = 9999")
        pd.testing.assert_frame_equal(engine.tables["orders"], original)

    def test_delete_preserves_other_rows(self, engine):
        engine.execute("DELETE FROM orders WHERE order_id = 101")
        remaining_ids = set(engine.tables["orders"]["order_id"])
        assert 101 not in remaining_ids
        assert {102, 103, 104} <= remaining_ids

    def test_delete_preserves_other_tables(self, engine):
        original_users = engine.tables["users"].copy()
        engine.execute("DELETE FROM orders WHERE order_id = 101")
        pd.testing.assert_frame_equal(engine.tables["users"], original_users)

    def test_delete_with_in_subquery(self, engine):
        engine.execute("""
            DELETE FROM orders WHERE user_id IN (
                SELECT user_id FROM users WHERE name = 'Bob'
            )
        """)
        # Bob is user_id=2; order 103 belongs to user_id=2
        remaining_user_ids = set(engine.tables["orders"]["user_id"])
        assert 2 not in remaining_user_ids

    def test_delete_with_compound_condition(self, engine):
        engine.execute("DELETE FROM orders WHERE status = 'paid' AND amount < 40")
        # order 102: paid, amount=30 → deleted
        # order 101: paid, amount=50 → kept
        remaining = engine.tables["orders"]
        assert 102 not in list(remaining["order_id"])
        assert 101 in list(remaining["order_id"])

    def test_delete_with_comparison(self, engine):
        engine.execute("DELETE FROM orders WHERE amount > 50")
        remaining = engine.tables["orders"]
        assert all(remaining["amount"] <= 50)

    def test_delete_from_empty_table(self, empty_engine):
        empty_engine.execute("DELETE FROM t WHERE id = 1")
        assert len(empty_engine.tables["t"]) == 0

    def test_delete_nonexistent_table_raises(self, engine):
        with pytest.raises(ValueError, match="does not exist"):
            engine.execute("DELETE FROM nosuchname WHERE x = 1")

    def test_delete_then_query(self, engine):
        engine.execute("DELETE FROM orders WHERE status = 'pending'")
        result = engine.query("SELECT status FROM orders")
        assert "pending" not in list(result["status"])

    def test_delete_row_count(self, engine):
        engine.execute("DELETE FROM orders WHERE user_id = 1")
        # user_id=1 has orders 101 and 102 → 2 rows deleted
        assert len(engine.tables["orders"]) == 2

    def test_delete_with_or_condition(self, engine):
        engine.execute("DELETE FROM orders WHERE order_id = 101 OR order_id = 103")
        remaining = set(engine.tables["orders"]["order_id"])
        assert remaining == {102, 104}

    def test_delete_with_not_condition(self, engine):
        engine.execute("DELETE FROM orders WHERE NOT status = 'paid'")
        # Only pending rows deleted (order 103)
        remaining = engine.tables["orders"]
        assert all(remaining["status"] == "paid")


# ─────────────────────────────────────────────────────────────
# NULL handling in DML
# ─────────────────────────────────────────────────────────────

class TestDMLNullHandling:
    def test_insert_null_then_is_null_query(self):
        df = pd.DataFrame({"id": [1, 2], "val": [10.0, 20.0]})
        e = FrameQL({"t": df})
        e.execute("INSERT INTO t (id, val) VALUES (3, NULL)")
        result = e.query("SELECT id FROM t WHERE val IS NULL")
        assert list(result["id"]) == [3]

    def test_update_null_field(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, None, 30.0]})
        e = FrameQL({"t": df})
        e.execute("UPDATE t SET val = 99.0 WHERE val IS NULL")
        assert e.tables["t"].iloc[1]["val"] == 99.0
        assert e.tables["t"].iloc[0]["val"] == 10.0

    def test_delete_where_is_null(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, None, 30.0]})
        e = FrameQL({"t": df})
        e.execute("DELETE FROM t WHERE val IS NULL")
        assert 2 not in list(e.tables["t"]["id"])
        assert len(e.tables["t"]) == 2

    def test_delete_where_is_not_null(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, None, 30.0]})
        e = FrameQL({"t": df})
        e.execute("DELETE FROM t WHERE val IS NOT NULL")
        assert len(e.tables["t"]) == 1
        assert pd.isna(e.tables["t"].iloc[0]["val"])

    def test_insert_partial_columns_null_fills(self, engine):
        engine.execute("INSERT INTO orders (order_id, user_id) VALUES (999, 1)")
        row = engine.tables["orders"].iloc[-1]
        assert row["order_id"] == 999
        assert pd.isna(row["amount"])
        assert pd.isna(row["status"])


# ─────────────────────────────────────────────────────────────
# Empty table edge cases
# ─────────────────────────────────────────────────────────────

class TestDMLEmptyTable:
    def test_insert_first_row(self, empty_engine):
        empty_engine.execute("INSERT INTO t (id, val) VALUES (1, 100.0)")
        assert len(empty_engine.tables["t"]) == 1
        assert empty_engine.tables["t"].iloc[0]["val"] == 100.0

    def test_update_empty_table(self, empty_engine):
        empty_engine.execute("UPDATE t SET val = 1.0")
        assert len(empty_engine.tables["t"]) == 0

    def test_delete_empty_table(self, empty_engine):
        empty_engine.execute("DELETE FROM t")
        assert len(empty_engine.tables["t"]) == 0

    def test_delete_empty_table_with_where(self, empty_engine):
        empty_engine.execute("DELETE FROM t WHERE id = 1")
        assert len(empty_engine.tables["t"]) == 0

    def test_insert_select_from_empty(self):
        src = pd.DataFrame({"id": pd.Series([], dtype=int), "val": pd.Series([], dtype=float)})
        dst = pd.DataFrame({"id": pd.Series([], dtype=int), "val": pd.Series([], dtype=float)})
        e = FrameQL({"src": src, "dst": dst})
        e.execute("INSERT INTO dst (id, val) SELECT id, val FROM src")
        assert len(e.tables["dst"]) == 0


# ─────────────────────────────────────────────────────────────
# Sequential DML operations
# ─────────────────────────────────────────────────────────────

class TestSequentialDML:
    def test_insert_then_update(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 50.0)")
        engine.execute("UPDATE users SET score = 95.0 WHERE name = 'Dave'")
        dave = engine.tables["users"][engine.tables["users"]["name"] == "Dave"].iloc[0]
        assert dave["score"] == 95.0

    def test_insert_then_delete(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 50.0)")
        assert len(engine.tables["users"]) == 4
        engine.execute("DELETE FROM users WHERE name = 'Dave'")
        assert len(engine.tables["users"]) == 3

    def test_update_then_delete(self, engine):
        engine.execute("UPDATE orders SET status = 'cancelled' WHERE amount < 30")
        engine.execute("DELETE FROM orders WHERE status = 'cancelled'")
        remaining = engine.tables["orders"]
        assert all(remaining["status"] != "cancelled")

    def test_full_lifecycle(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
        e = FrameQL({"t": df})

        # INSERT
        e.execute("INSERT INTO t (id, val) VALUES (4, 40.0)")
        assert len(e.tables["t"]) == 4

        # UPDATE
        e.execute("UPDATE t SET val = val * 2 WHERE id <= 2")
        assert e.tables["t"].iloc[0]["val"] == 20.0
        assert e.tables["t"].iloc[1]["val"] == 40.0
        assert e.tables["t"].iloc[2]["val"] == 30.0  # unchanged

        # DELETE
        e.execute("DELETE FROM t WHERE val < 25")
        remaining_ids = set(e.tables["t"]["id"])
        assert 1 not in remaining_ids  # 20 is still > 25? No, 20 < 25
        assert 3 in remaining_ids
        assert 4 in remaining_ids

        # SELECT verification
        result = e.query("SELECT SUM(val) AS total FROM t")
        expected = e.tables["t"]["val"].sum()
        assert result.iloc[0]["total"] == expected

    def test_multiple_inserts_accumulate(self, empty_engine):
        for i in range(5):
            empty_engine.execute(f"INSERT INTO t (id, val) VALUES ({i}, {float(i * 10)})")
        assert len(empty_engine.tables["t"]) == 5

    def test_delete_all_then_reinsert(self, engine):
        engine.execute("DELETE FROM users")
        assert len(engine.tables["users"]) == 0
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (1, 'New', 50.0)")
        assert len(engine.tables["users"]) == 1
        assert engine.tables["users"].iloc[0]["name"] == "New"

    def test_update_then_select_with_join(self, engine):
        engine.execute("UPDATE users SET score = 0.0 WHERE user_id = 2")
        result = engine.query("""
            SELECT u.name, u.score, SUM(o.amount) AS total
            FROM users u
            INNER JOIN orders o ON u.user_id = o.user_id
            GROUP BY u.name, u.score
            ORDER BY total DESC
        """)
        bob = result[result["name"] == "Bob"].iloc[0]
        assert bob["score"] == 0.0


# ─────────────────────────────────────────────────────────────
# Data integrity validation
# ─────────────────────────────────────────────────────────────

class TestDataIntegrity:
    def test_update_does_not_duplicate_rows(self, engine):
        original_count = len(engine.tables["users"])
        engine.execute("UPDATE users SET score = 100.0")
        assert len(engine.tables["users"]) == original_count

    def test_delete_does_not_duplicate_rows(self, engine):
        engine.execute("DELETE FROM orders WHERE order_id = 101")
        assert len(engine.tables["orders"]) == 3

    def test_insert_schema_consistent(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 60.0)")
        assert list(engine.tables["users"].columns) == ["user_id", "name", "score"]

    def test_update_only_targeted_column_changes(self, engine):
        engine.execute("UPDATE orders SET amount = 999.0 WHERE order_id = 101")
        row = engine.tables["orders"][engine.tables["orders"]["order_id"] == 101].iloc[0]
        assert row["amount"] == 999.0
        # Other columns unchanged
        assert row["user_id"] == 1
        assert row["status"] == "paid"

    def test_insert_index_is_reset(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 60.0)")
        idx = list(engine.tables["users"].index)
        assert idx == list(range(len(engine.tables["users"])))

    def test_delete_index_is_reset(self, engine):
        engine.execute("DELETE FROM orders WHERE order_id = 101")
        idx = list(engine.tables["orders"].index)
        assert idx == list(range(len(engine.tables["orders"])))

    def test_update_values_match_expectations(self, engine):
        engine.execute("UPDATE orders SET amount = amount + 5.0")
        expected = [55.0, 35.0, 25.0, 105.0]
        actual = list(engine.tables["orders"]["amount"])
        for a, e_val in zip(actual, expected):
            assert abs(a - e_val) < 0.001


# ─────────────────────────────────────────────────────────────
# execute() as SELECT router (backward compat)
# ─────────────────────────────────────────────────────────────

class TestExecuteSelectRouter:
    def test_execute_routes_select(self, engine):
        result = engine.execute("SELECT user_id, name FROM users WHERE user_id = 1")
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_execute_returns_none_for_insert(self, engine):
        result = engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'D', 0.0)")
        assert result is None

    def test_execute_returns_none_for_update(self, engine):
        result = engine.execute("UPDATE users SET score = 0.0")
        assert result is None

    def test_execute_returns_none_for_delete(self, engine):
        result = engine.execute("DELETE FROM users WHERE user_id = 1")
        assert result is None

    def test_execute_complex_select(self, engine):
        result = engine.execute("""
            SELECT u.name, SUM(o.amount) AS total
            FROM users u
            INNER JOIN orders o ON u.user_id = o.user_id
            GROUP BY u.name
            ORDER BY total DESC
        """)
        assert result is not None
        assert "name" in result.columns
        assert "total" in result.columns
        totals = list(result["total"])
        assert totals == sorted(totals, reverse=True)


# ─────────────────────────────────────────────────────────────
# DML with subqueries
# ─────────────────────────────────────────────────────────────

class TestDMLWithSubqueries:
    def test_delete_with_scalar_subquery_in_where(self, engine):
        # Delete orders whose amount is below the average
        avg = engine.tables["orders"]["amount"].mean()  # 50
        engine.execute("DELETE FROM orders WHERE amount < (SELECT AVG(amount) FROM orders)")
        remaining = engine.tables["orders"]
        assert all(remaining["amount"] >= avg)

    def test_update_with_scalar_subquery_in_set(self, engine):
        engine.execute("UPDATE orders SET amount = (SELECT MAX(amount) FROM orders) WHERE order_id = 103")
        row = engine.tables["orders"][engine.tables["orders"]["order_id"] == 103].iloc[0]
        assert row["amount"] == 100.0  # MAX was 100.0

    def test_delete_in_subquery(self, engine):
        engine.execute("""
            DELETE FROM orders
            WHERE user_id IN (SELECT user_id FROM users WHERE score < 75)
        """)
        # Charlie (score=70) → user_id=3 → order 104 deleted
        remaining_order_ids = set(engine.tables["orders"]["order_id"])
        assert 104 not in remaining_order_ids

    def test_insert_from_select_with_subquery(self, engine):
        log = pd.DataFrame({"user_id": pd.Series([], dtype=int), "name": pd.Series([], dtype=str)})
        engine.tables["log"] = log
        engine.execute("""
            INSERT INTO log (user_id, name)
            SELECT user_id, name FROM users
            WHERE score = (SELECT MAX(score) FROM users)
        """)
        assert len(engine.tables["log"]) == 1
        assert engine.tables["log"].iloc[0]["name"] == "Alice"


# ─────────────────────────────────────────────────────────────
# DML regression: existing SELECT features unaffected
# ─────────────────────────────────────────────────────────────

class TestDMLRegressionSelectUnaffected:
    """After DML operations, existing SELECT features must work correctly."""

    def test_join_still_works_after_insert(self, engine):
        engine.execute("INSERT INTO users (user_id, name, score) VALUES (4, 'Dave', 55.0)")
        result = engine.query("""
            SELECT u.name, o.amount
            FROM users u
            INNER JOIN orders o ON u.user_id = o.user_id
        """)
        assert len(result) == 4  # same as before; Dave has no orders
        assert "Dave" not in list(result["name"])

    def test_group_by_works_after_update(self, engine):
        engine.execute("UPDATE orders SET amount = amount * 2 WHERE user_id = 1")
        result = engine.query("SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id ORDER BY user_id")
        row1 = result[result["user_id"] == 1].iloc[0]
        assert row1["total"] == (50.0 + 30.0) * 2  # doubled

    def test_window_functions_work_after_delete(self, engine):
        engine.execute("DELETE FROM orders WHERE order_id = 101")
        result = engine.query("""
            SELECT user_id, amount,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount) AS rn
            FROM orders
        """)
        assert "rn" in result.columns
        for uid, grp in result.groupby("user_id"):
            assert sorted(grp["rn"].tolist()) == list(range(1, len(grp) + 1))

    def test_cte_works_after_dml(self, engine):
        engine.execute("UPDATE orders SET amount = 0.0 WHERE order_id = 103")
        result = engine.query("""
            WITH totals AS (
                SELECT user_id, SUM(amount) AS total
                FROM orders GROUP BY user_id
            )
            SELECT user_id, total FROM totals ORDER BY total DESC
        """)
        assert result.iloc[0]["total"] == result["total"].max()

    def test_subquery_works_after_dml(self, engine):
        engine.execute("DELETE FROM orders WHERE amount < 25")
        result = engine.query("""
            SELECT user_id, amount FROM orders
            WHERE amount > (SELECT AVG(amount) FROM orders)
        """)
        avg = engine.tables["orders"]["amount"].mean()
        assert all(result["amount"] > avg)
