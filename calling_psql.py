import pandas as pd
from FrameQL import FrameQL

# # Sample data
# import pandas as pd

# data = [
#     (1,"Alice","North","Laptop",1,900,"2025-01-10"),
#     (2,"Bob","South","Mouse",2,25,"2025-01-11"),
#     (3,"Alice","North","Keyboard",1,45,"2025-01-12"),
#     (4,"David","East","Monitor",2,200,"2025-01-13"),
#     (5,"Emma","West","Laptop",1,950,"2025-01-14"),
#     (6,"Bob","South","Laptop",1,880,"2025-01-15"),
#     (7,"Alice","North","Mouse",3,20,"2025-01-16"),
# ]

# df = pd.DataFrame(data, columns=[
#     "order_id","customer","region","product",
#     "quantity","price","order_date"
# ])


# # Create engine
# engine = PandasSQL({"df":df})

# # SQL query
# query = """
# SELECT d1.customer,
#        d1.region,
#        d1.price,

#        -- Customer-level average (correlated subquery)
#        (
#            SELECT AVG(d2.price)
#            FROM df d2
#            WHERE d2.customer = d1.customer
#        ) AS customer_avg,

#        -- Difference from customer's average
#        d1.price - (
#            SELECT AVG(d2.price)
#            FROM df d2
#            WHERE d2.customer = d1.customer
#        ) AS diff_from_customer_avg,

#        -- Global maximum price (independent subquery)
#        (
#            SELECT MAX(price)
#            FROM df
#        ) AS global_max

# FROM df d1
# WHERE d1.price > (
#     -- filter using another correlated subquery
#     SELECT AVG(d3.price)
#     FROM df d3
#     WHERE d3.region = d1.region
# )
# ORDER BY diff_from_customer_avg DESC;
# """

# # Execute
# result = engine.query(query)

# print(result)

# 1. Create Sample Data
df_users = pd.DataFrame({
    "user_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "David"]
})

df_orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104, 105],
    "user_id": [1, 1, 2, 3, 99], # Note: 99 doesn't exist in Users, 4 has no orders
    "amount": [50.0, 30.0, 20.0, 100.0, 15.0]
})

# 2. Initialize the Engine
engine = FrameQL({
    "users": df_users,
    "orders": df_orders
})

# 3. Test Cases

print("--- TEST 1: INNER JOIN ---")
# Only returns users who have orders
sql_inner="""
SELECT u1.user_id,
       u1.name,

       CASE
           WHEN (
               SELECT SUM(o1.amount)
               FROM orders o1
               WHERE o1.user_id = u1.user_id
           ) > 50 THEN 'high'

           WHEN (
               SELECT SUM(o2.amount)
               FROM orders o2
               WHERE o2.user_id = u1.user_id
           )  in (0) THEN 'no_orders'

           ELSE 'low'
       END AS spending_category

FROM users u1;
"""


print(engine.query(sql_inner))
