import pandas as pd
import numpy as np
import sqlglot
from sqlglot import exp
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

@dataclass
class QueryPlan:
    select: List
    where: object = None
    group_by_exprs: List = None
    having: object = None
    order_by: object = None
    limit: object = None
    resolution_map: Dict[str, str] = None
    distinct: bool = False

class FrameQL:
    AGG_MAP = {
        exp.Sum: "sum",
        exp.Min: "min",
        exp.Max: "max",
        exp.Avg: "mean",
        exp.Count: "count",
    }

    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self.tables = {k.lower(): v for k, v in tables.items()}

    def _resolve_col(self, df: Union[pd.DataFrame, pd.Series], name: str, table_alias: str = None) -> str:
        cols = df.index if isinstance(df, pd.Series) else df.columns
        name_lower = name.lower()
        
        # 1. Priority: Exact 'alias.column' (e.g., 'u.user_id')
        if table_alias:
            prefixed = f"{table_alias.lower()}.{name_lower}"
            for c in cols:
                if str(c).lower() == prefixed:
                    return c

        # 2. Direct Match (for temp columns or non-prefixed columns)
        if name in cols: return name
        
        # 3. Case-Insensitive Match
        for c in cols:
            if str(c).lower() == name_lower:
                return c

        # 4. Fuzzy/Suffix Match (CRITICAL for General SQL)
        # This allows 'amount' to find 'orders.amount' or 'o.amount'
        for c in cols:
            col_str = str(c).lower()
            if col_str.endswith(f".{name_lower}"):
                return c
                
        return name

    def query(self, sql: str, outer_scope: Dict[str, Any] = None):
        tree = sqlglot.parse_one(sql)
        outer_scope = outer_scope or {}
        
        # 1. Execute Joins to get the base DataFrame and the updated scope
        df_context, scope = self.execute_joins(tree, outer_scope)
        
        # 2. Structural Correlation Check for Subqueries
        from_node = tree.args.get("from") or tree.args.get("from_")
        local_aliases = set(scope.keys()) - set(outer_scope.keys())

        for subquery in list(tree.find_all(exp.Subquery)):
            if subquery.parent is from_node: continue
            
            is_correlated = False
            for col in subquery.find_all(exp.Column):
                col_table = col.args.get("table")
                if col_table:
                    col_table_name = col_table.this.lower()
                    if col_table_name in local_aliases or col_table_name in outer_scope:
                        is_correlated = True
                        break
            
            if not is_correlated:
                sub_result = self.query(subquery.this.sql(), outer_scope=scope)
                self._replace_subquery_with_literal(subquery, sub_result)

        # 3. Execute the rest of the plan
        plan = self.build_plan(tree)
        return self.execute_plan(df_context, plan, scope)

    def execute_joins(self, tree, outer_scope):
        from_node = tree.args.get("from") or tree.args.get("from_")
        if not from_node:
            return pd.DataFrame([{}], index=[0]), outer_scope.copy()

        tables = list(from_node.find_all(exp.Table))
        main_table = tables[0]
        main_real_name = main_table.this.this.lower()
        main_alias = main_table.alias.lower() if main_table.alias else main_real_name
        
        result_df = self.tables[main_real_name].copy()
        result_df.columns = [f"{main_alias}.{c}" for c in result_df.columns]
        
        scope = outer_scope.copy()
        scope[main_alias] = result_df
        existing_aliases = {main_alias}

        for join in tree.find_all(exp.Join):
            join_table = join.this
            join_real_name = join_table.this.this.lower()
            join_alias = join_table.alias.lower() if join_table.alias else join_real_name
            
            right_df = self.tables[join_real_name].copy()
            right_df.columns = [f"{join_alias}.{c}" for c in right_df.columns]
            
            side = str(join.args.get("side")).upper() if join.args.get("side") else ""
            how = "left" if "LEFT" in side else "right" if "RIGHT" in side else "inner"

            on_cond = join.args.get("on")
            if on_cond:
                # 1. Extract EQUALITY keys for the optimized Pandas merge
                left_keys, right_keys = self._extract_all_join_keys(
                    on_cond, existing_aliases, join_alias, result_df, right_df
                )
                
                # 2. Perform the Join
                if left_keys and right_keys:
                    result_df = pd.merge(result_df, right_df, left_on=left_keys, right_on=right_keys, how=how)
                else:
                    # If no equalities (e.g., ON u.age > o.min_age), we must cross join first
                    result_df = pd.merge(result_df, right_df, how='cross')
                
                # 3. Apply Remaining Filters (The "amount > 25" part)
                # We sync the scope temporarily so evaluate_expr can see the new combined columns
                temp_scope = scope.copy()
                temp_scope[join_alias] = result_df
                for alias in existing_aliases:
                    temp_scope[alias] = result_df
                
                # evaluate_expr will return a boolean mask for the whole merged DF
                mask = self.evaluate_expr(on_cond, temp_scope)
                result_df = result_df[mask].reset_index(drop=True)
            else:
                result_df = result_df.merge(right_df, how='cross')
            
            existing_aliases.add(join_alias)
            for alias in scope:
                if isinstance(scope[alias], pd.DataFrame): scope[alias] = result_df
            scope[join_alias] = result_df
                    
        return result_df, scope
    
    def _extract_all_join_keys(self, on_cond, existing_aliases, right_alias, left_df, right_df):
        left_keys = []
        right_keys = []
        
        # Rationale: Recursively find all "=" conditions within the "AND" blocks
        for eq_expr in on_cond.find_all(exp.EQ):
            l_col = eq_expr.left
            r_col = eq_expr.right
            
            # Determine which side of the "=" belongs to the right table
            # and which belongs to the already joined "left" tables.
            l_table = l_col.table.lower() if l_col.table else ""
            r_table = r_col.table.lower() if r_col.table else ""
            
            if r_table == right_alias:
                # Case: left_table.col = right_alias.col
                left_keys.append(self._resolve_col(left_df, l_col.name, l_table))
                right_keys.append(self._resolve_col(right_df, r_col.name, r_table))
            elif l_table == right_alias:
                # Case: right_alias.col = left_table.col
                left_keys.append(self._resolve_col(left_df, r_col.name, r_table))
                right_keys.append(self._resolve_col(right_df, l_col.name, l_table))
            else:
                # Fallback if table aliases are missing: try to find column in right_df
                # This makes the engine more resilient to simple "ON id = id"
                if f"{right_alias}.{r_col.name}" in right_df.columns:
                    right_keys.append(f"{right_alias}.{r_col.name}")
                    left_keys.append(self._resolve_col(left_df, l_col.name, l_table))
                else:
                    right_keys.append(f"{right_alias}.{l_col.name}")
                    left_keys.append(self._resolve_col(left_df, r_col.name, r_table))
                    
        return left_keys, right_keys
    
    def execute_case(self, expr, scope, resolution_map=None):
        conditions, choices = [], []

        for if_node in expr.args.get('ifs', []):
            cond_mask = self.eval_condition(if_node.this, scope, resolution_map)

            then_expr = if_node.args.get('true') or if_node.args.get('then')
            then_val = self.evaluate_expr(then_expr, scope, resolution_map)

            conditions.append(cond_mask)
            choices.append(then_val)

        # ELSE
        default_expr = expr.args.get('default')
        default_val = (
            self.evaluate_expr(default_expr, scope, resolution_map)
            if default_expr else None   # ✅ use None instead of np.nan
        )

        def to_object_array(x):
            if isinstance(x, (pd.Series, np.ndarray)):
                return x.astype(object)
            return x

        choices = [to_object_array(c) for c in choices]
        default_val = to_object_array(default_val)

        if all(isinstance(c, (pd.Series, np.ndarray)) for c in conditions):
            return np.select(conditions, choices, default=default_val)
        else:
            # scalar fallback (for row-wise execution)
            for cond, val in zip(conditions, choices):
                if cond:
                    return val
            return default_val

    def _replace_subquery_with_literal(self, subquery, sub_df):
        parent = subquery.parent
        if isinstance(parent, exp.In):
            values = sub_df.iloc[:, 0].tolist()
            replacement = [exp.Literal.number(v) if isinstance(v, (int, float, np.number)) 
                           else exp.Literal.string(str(v)) for v in values]
            parent.set("expressions", replacement)
            subquery.pop()
        else:
            if sub_df.empty:
                subquery.replace(exp.Null())
            else:
                val = sub_df.iloc[0, 0]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    replacement = exp.Literal.number(float(val))
                else:
                    replacement = exp.Literal.string(str(val))
                subquery.replace(replacement)

    def build_plan(self, tree):
        group_node = tree.args.get("group") or tree.args.get("group_")
        return QueryPlan(
            select=tree.expressions,
            where=tree.args.get("where"),
            group_by_exprs=group_node.expressions if group_node else [],
            having=tree.args.get("having"),
            order_by=tree.args.get("order"),
            limit=tree.args.get("limit"),
            resolution_map={},
            distinct=bool(tree.args.get("distinct"))
        )

    def evaluate_expr(self, expr, scope, resolution_map=None):
        if expr is None or isinstance(expr, exp.Null): 
            return None
        
        # 1. NORMALIZE THE KEY
        raw_sql = expr.sql()

        
        # 2. THE SHORTCUT
        if resolution_map and raw_sql in resolution_map:
            target_df = list(scope.values())[0]
            resolved_name = resolution_map[raw_sql]
            if resolved_name in target_df.columns:
                return target_df[resolved_name]

        # 3. SCALAR SUBQUERIES
        if isinstance(expr, exp.Subquery):
            sub_res = self.query(expr.this.sql(), outer_scope=scope)
            return sub_res.iloc[0, 0] if not sub_res.empty else None

        # 4. COLUMN RESOLUTION
        if isinstance(expr, exp.Column):
            table_alias = expr.table.lower() if expr.table else None
            col_name = expr.name
            target_df = list(scope.values())[0] 
            resolved_name = self._resolve_col(target_df, col_name, table_alias)
            return target_df[resolved_name]

                # --- NEW: COUNT(DISTINCT ...) ---
        if isinstance(expr, exp.Count) and isinstance(expr.this, exp.Distinct):
            inner_expr = expr.this.expressions[0]
            series = self.evaluate_expr(inner_expr, scope, resolution_map)

            
            if isinstance(series, np.ndarray):
                series = pd.Series(series)

            if not isinstance(series, pd.Series):
                return 1 if series is not None else 0

            return series.nunique(dropna=True)
                
        # 5. AGGREGATES
        if type(expr) in self.AGG_MAP:
            method = self.AGG_MAP[type(expr)]
            primary_df = list(scope.values())[-1] 
            if isinstance(expr.this, (exp.Star, exp.Literal)):
                return len(primary_df) if method == "count" else 0
            
            series = self.evaluate_expr(expr.this, scope, resolution_map)
            if not hasattr(series, method):
                return series
            return getattr(series, method)()

        # 6. LITERALS
        if isinstance(expr, exp.Literal):
            if expr.is_string: return expr.this
            # Handle numeric literals safely
            try:
                return float(expr.this) if "." in expr.this else int(expr.this)
            except ValueError:
                return expr.this
        
        # ADD THIS inside evaluate_expr (somewhere before math ops is fine)
        if isinstance(expr, exp.Case):
            return self.execute_case(expr, scope, resolution_map)

        # 7. LOGICAL OPERATORS (AND, OR)
        # Rationale: Pandas uses bitwise & and | for Series-wise boolean logic
        if isinstance(expr, (exp.And, exp.Or)):
            l = self.evaluate_expr(expr.left, scope, resolution_map)
            r = self.evaluate_expr(expr.right, scope, resolution_map)
            if isinstance(expr, exp.And): return l & r
            if isinstance(expr, exp.Or): return l | r

        # 8. COMPARISON OPERATORS (=, !=, >, <, >=, <=)
        if isinstance(expr, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            l = self.evaluate_expr(expr.left, scope, resolution_map)
            r = self.evaluate_expr(expr.right, scope, resolution_map)
            
            if isinstance(expr, exp.EQ): return l == r
            if isinstance(expr, exp.NEQ): return l != r
            if isinstance(expr, exp.GT): return l > r
            if isinstance(expr, exp.GTE): return l >= r
            if isinstance(expr, exp.LT): return l < r
            if isinstance(expr, exp.LTE): return l <= r

        # 9. UNARY OPERATORS (NOT, PAREN)
        if isinstance(expr, exp.Not):
            val = self.evaluate_expr(expr.this, scope, resolution_map)
            return ~val if hasattr(val, '__invert__') else not val
        
        if isinstance(expr, exp.Paren):
            return self.evaluate_expr(expr.this, scope, resolution_map)

        # 10. MATH OPERATIONS
        if isinstance(expr, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            l = self.evaluate_expr(expr.left, scope, resolution_map)
            r = self.evaluate_expr(expr.right, scope, resolution_map)
            
            if l is None or r is None: return None
            
            if isinstance(expr, exp.Add): return l + r
            if isinstance(expr, exp.Sub): return l - r
            if isinstance(expr, exp.Mul): return l * r
            if isinstance(expr, exp.Div): return l / r
            
        return expr.sql()

    def eval_condition(self, cond, scope, resolution_map=None):
        if isinstance(cond, exp.Paren):
            return self.eval_condition(cond.this, scope, resolution_map)
        
        # 1. Handle Logical Connectives (AND/OR)
        if isinstance(cond, (exp.And, exp.Or)):
            l = self.eval_condition(cond.left, scope, resolution_map)
            r = self.eval_condition(cond.right, scope, resolution_map)
            return (l & r) if isinstance(cond, exp.And) else (l | r)

        # 2. Handle IS NULL / IS NOT NULL
        if isinstance(cond, exp.Is):
            l_val = self.evaluate_expr(cond.left, scope, resolution_map)
            if isinstance(cond.args.get("expression"), exp.Null):
                return l_val.isna()
            return l_val == self.evaluate_expr(cond.args.get("expression"), scope, resolution_map)

        # 3. Row-Level Subquery Check (Correlation)
        primary_df = list(scope.values())[0]
        is_row_level = isinstance(primary_df, pd.Series)
        if not is_row_level and any(isinstance(node, exp.Subquery) for node in cond.find_all(exp.Subquery)):
            def row_eval(row):
                row_scope = scope.copy()
                first_alias = list(scope.keys())[0]
                row_scope[first_alias] = row
                return self.eval_condition(cond, row_scope, resolution_map)
            return primary_df.apply(row_eval, axis=1).astype(bool)

        # 4. Resolve the LEFT side safely
        # Based on your AST: In(this=Column(...))
        left_node = cond.this if isinstance(cond, exp.In) else cond.left
        l_val = self.evaluate_expr(left_node, scope, resolution_map)

        # 5. Resolve the RIGHT side safely
        if isinstance(cond, exp.In):
            # Based on your AST: In has a 'query' arg for Subqueries
            subquery_node = cond.args.get('query') or cond.args.get('field')
            
            if subquery_node:
                # This calls evaluate_expr which runs the subquery
                r_val = self.evaluate_expr(subquery_node, scope, resolution_map)
            else:
                # For static lists like IN (1, 2, 3)
                expr_list = cond.args.get('expressions', [])
                r_val = [self.evaluate_expr(e, scope, resolution_map) for e in expr_list]
        else:
            # Standard comparison operators
            r_node = cond.args.get('expression') or cond.args.get('right')
            r_val = self.evaluate_expr(r_node, scope, resolution_map)

        # 6. Comparison Logic
        if isinstance(cond, exp.In):
            # Flatten r_val if it's a Pandas object from the subquery
            if hasattr(r_val, 'tolist'): 
                r_val = r_val.tolist()
            elif hasattr(r_val, 'values'): 
                r_val = r_val.values.flatten()
            
            # Perform membership check
            return l_val.isin(r_val) if hasattr(l_val, 'isin') else l_val in r_val

        # Standard operators
        if isinstance(cond, exp.EQ): return l_val == r_val
        if isinstance(cond, exp.NEQ): return l_val != r_val
        if isinstance(cond, exp.GT): return l_val > r_val
        if isinstance(cond, exp.LT): return l_val < r_val
        if isinstance(cond, exp.GTE): return l_val >= r_val
        if isinstance(cond, exp.LTE): return l_val <= r_val
        
        return False
    
    def execute_plan(self, df, plan, scope):
        current_alias = list(scope.keys())[0] if scope else "df"
        scope[current_alias] = df

        # 1. WHERE: Filter raw rows
        if plan.where:
            mask = self.eval_condition(plan.where.this, scope)
            if not isinstance(mask, (pd.Series, np.ndarray)):
                mask = np.full(len(df), mask)
            df = df[mask.values if isinstance(mask, pd.Series) else mask]
            # Update scope so subsequent steps see filtered rows
            for k in scope: scope[k] = df

        # 2. GROUP BY: Aggregate data
        if plan.group_by_exprs:
            df = self.execute_groupby(df, plan, scope)
            # execute_groupby updates the scope internally to the new agg_df
        
        # 3. HAVING: Filter aggregated groups
        if plan.having:
            for k in scope: scope[k] = df
            # CRITICAL: Passing plan.resolution_map here!
            mask = self.eval_condition(plan.having.this, scope, plan.resolution_map)
            
            if not isinstance(mask, (pd.Series, np.ndarray)):
                mask = np.full(len(df), mask)
            df = df[mask.values if isinstance(mask, pd.Series) else mask]

        # 4. SELECT: Create final columns and aliases (e.g., 'total')
        # We run this for BOTH grouped and non-grouped queries now
        df = self.execute_select(df, plan, scope)
        
        if getattr(plan, "distinct", False):
            df = df.drop_duplicates().reset_index(drop=True)

        # Update scope one last time for Order By
        for k in scope: scope[k] = df

        # 5. ORDER BY: Sort using final names or aggregates
        if plan.order_by:
            df = self.execute_order_by(df, plan, scope)

        # 6. LIMIT: Constrain result size
        if plan.limit:
            limit_val = int(plan.limit.expression.this)
            df = df.head(limit_val)

        return df

    def execute_select(self, df, plan, scope):
        # 1. Preserve SELECT *
        if any(isinstance(e, exp.Star) for e in plan.select): 
            return df
            
        has_subquery = any(list(e.find_all(exp.Subquery)) for e in plan.select)

        if not has_subquery:
            # Vectorized execution (Standard)
            new_cols, order = {}, []
            for expr in plan.select:
                is_alias = isinstance(expr, exp.Alias)
                name = expr.alias if is_alias else expr.sql()
                val = self.evaluate_expr(expr.this if is_alias else expr, scope, plan.resolution_map)
                new_cols[name] = val
                order.append(name)
            
            is_all_scalars = all(not isinstance(v, (pd.Series, np.ndarray, list)) for v in new_cols.values())
            return pd.DataFrame(new_cols, index=df.index if not is_all_scalars else [0])[order]
        
        else:
            # Correlated Row-by-Row Execution
            def process_row(row):
                row_scope = scope.copy()
                for alias in row_scope:
                    row_scope[alias] = row
                
                row_results = {}
                for e in plan.select:
                    is_alias = isinstance(e, exp.Alias)
                    name = e.alias if is_alias else e.sql()
                    inner = e.this if is_alias else e
                    
                    val = self.evaluate_expr(inner, row_scope, plan.resolution_map)
                    
                    # --- FEATURE PRESERVATION: Scalar Extraction ---
                    # When a subquery like (SELECT COUNT(*)...) runs, it returns a 
                    # DataFrame or Series. We must extract the actual number.
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        val = val.iloc[0, 0] if not val.empty else 0
                    elif isinstance(val, np.ndarray):
                        val = val[0] if val.size > 0 else 0
                        
                    row_results[name] = val
                
                return pd.Series(row_results)

            # Apply row-by-row
            return df.apply(process_row, axis=1)
        
    def execute_groupby(self, df, plan, scope):
        group_cols, group_res = [], {}
        alias = list(scope.keys())[0]
        
        # 1. Resolve Grouping Columns
        for i, g_expr in enumerate(plan.group_by_exprs):
            raw_sql = g_expr.sql()
            if isinstance(g_expr, exp.Column):
                t_alias = g_expr.table.lower() if g_expr.table else None
                resolved_col = self._resolve_col(df, g_expr.name, t_alias)
                group_cols.append(resolved_col)
                plan.resolution_map[raw_sql] = resolved_col
            else:
                col_name = f"tmp_group_{i}"
                df[col_name] = self.evaluate_expr(g_expr, scope)
                group_cols.append(col_name)
                plan.resolution_map[raw_sql] = col_name

        # 2. Identify Aggregations (Includes SELECT, HAVING, and ORDER BY)
        agg_map = {}
        all_nodes = list(plan.select)
        if plan.having: 
            all_nodes.append(plan.having)
        if plan.order_by: 
            all_nodes.extend(plan.order_by.expressions)
        
        all_aggs = []
        for target in all_nodes:
            # Recursively find all aggregates (SUM, COUNT, etc.)
            for node in target.find_all(tuple(self.AGG_MAP.keys())): 
                all_aggs.append(node)
        
        # 3. Build Aggregation Map & Handle Aliases
        # We need to determine the final column names before we call .agg()
        named_aggs = {}
        for agg in all_aggs:
            raw_sql = agg.sql() # e.g., "SUM(amount)"
            
            if raw_sql not in agg_map:
                # Check if this aggregate has an alias in the SELECT clause
                alias_name = None
                for s_expr in plan.select:

                    if isinstance(s_expr, exp.Alias) and s_expr.this.sql() == raw_sql:
                        alias_name = s_expr.alias
                        break
                
                # The column name in the final DataFrame will be the Alias if it exists, 
                # otherwise it stays the raw SQL string.
                final_col_name = alias_name if alias_name else raw_sql
                
                # --- NEW: Detect DISTINCT ---
                is_distinct = isinstance(agg.this, exp.Distinct)

                # --- NEW: Extract actual expression ---
                if is_distinct:
                    if len(agg.this.expressions) != 1:
                        raise NotImplementedError("COUNT(DISTINCT col1, col2) not supported")
                    inner_expr = agg.this.expressions[0]
                else:
                    inner_expr = agg.this

                # --- NEW: Choose aggregation function ---
                if isinstance(agg, exp.Count) and is_distinct:
                    p_func = "nunique"
                else:
                    p_func = self.AGG_MAP[type(agg)]

                # --- REPLACE agg.this usage BELOW with inner_expr ---
                if isinstance(inner_expr, (exp.Star, exp.Literal)):
                    t_col = group_cols[0]
                else:
                    t_alias = inner_expr.table.lower() if isinstance(inner_expr, exp.Column) and inner_expr.table else None
                    inner_name = inner_expr.name if isinstance(inner_expr, exp.Column) else inner_expr.sql()
                    t_col = self._resolve_col(df, inner_name, t_alias)

                    if t_col not in df.columns:
                        tmp_pre = f"tmp_pre_{len(agg_map)}"
                        df[tmp_pre] = self.evaluate_expr(inner_expr, scope)
                        t_col = tmp_pre
                
                # Map the raw SQL string to the physical column name (e.g., "SUM(amount)" -> "total")
                plan.resolution_map[raw_sql] = final_col_name
                if alias_name:
                    plan.resolution_map[alias_name] = final_col_name
                
                named_aggs[final_col_name] = pd.NamedAgg(column=t_col, aggfunc=p_func)
                agg_map[raw_sql] = final_col_name

        # 4. Execute GroupBy and Update Scope
        agg_df = df.groupby(group_cols, as_index=False).agg(**named_aggs)
        
        for k in scope:
            scope[k] = agg_df

        return agg_df

    def execute_order_by(self, df, plan, scope):
        sort_cols, asc = [], []
        for i, expr in enumerate(plan.order_by.expressions):
            # expr is 'Ordered', expr.this is the actual column/agg
            actual_expr = expr.this 
            raw_sql = actual_expr.sql() 
        
            # Now the raw_sql will be "SUM(amount)", which matches your map!
            col_name = plan.resolution_map.get(raw_sql)

            print(f'{col_name} name in order by')

            print(df.columns)
            
            if col_name and col_name in df.columns:
                sort_cols.append(col_name)
            else:
                print("Inside else")
                    # Fallback
                tmp_name = f"tmp_sort_{i}"
                df[tmp_name] = self.evaluate_expr(actual_expr, scope, plan.resolution_map)
                sort_cols.append(tmp_name)

            
            asc.append(not expr.args.get("desc", False))
            
        df.sort_values(by=sort_cols, ascending=asc)
        return df.drop(columns=[c for c in df.columns if str(c).startswith("tmp_sort_")])
    
    def execute_limit(self, df, plan):
        try: return df.head(int(plan.limit.expression.this))
        except: return df
