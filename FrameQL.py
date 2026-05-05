"""
FrameQL — SQL queries on Pandas DataFrames.

Architecture
────────────
 1. build_logical_plan(tree, df) → QueryPlan
    - Scans all clauses upfront
    - Populates resolution_map before any execution
    - Records agg_specs so aggregates are computed exactly once

 2. execute_plan(df, plan, scope) → DataFrame
    - WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT
    - ORDER BY runs after SELECT so aliases are visible; falls back to
      pre-projection df for columns that were not projected
    - No recomputation of aggregates after GROUP BY
"""

import pandas as pd
import numpy as np
import sqlglot
from sqlglot import exp
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Logical Plan
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AggSpec:
    """Specification for a single aggregation used in groupby."""
    source_sql: str     # e.g. "SUM(o.amount)" — key for resolution_map
    func: str           # pandas aggfunc name: "sum", "mean", "count", …
    source_col: str     # physical column name, "__count_star__", or "__expr__"
    output_col: str     # column name in the post-groupby DataFrame


@dataclass
class WindowSpec:
    """Specification for a single window function."""
    source_sql: str          # full window expression SQL — key for resolution_map
    func: str                # "row_number", "rank", "dense_rank", "sum", "mean"
    func_arg: Optional[str]  # for SUM/AVG: resolved source column name
    partition_cols: List[str]
    order_cols: List[str]
    order_asc: List[bool]
    output_col: str          # internal temp column name in df


@dataclass
class QueryPlan:
    select: List
    where: Any = None
    group_by_exprs: List = field(default_factory=list)
    group_cols: List[str] = field(default_factory=list)   # resolved physical names
    having: Any = None
    order_by: Any = None
    limit: Optional[int] = None
    distinct: bool = False
    resolution_map: Dict[str, str] = field(default_factory=dict)
    agg_specs: List[AggSpec] = field(default_factory=list)
    window_specs: List[WindowSpec] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── public API ────────────────────────────────────────────────────────────

    def query(self, sql: str, outer_scope: Dict[str, Any] = None) -> pd.DataFrame:
        tree = sqlglot.parse_one(sql)
        outer_scope = outer_scope or {}

        # Execute CTEs in order and register results as temporary tables.
        cte_names: List[str] = []
        with_node = tree.args.get("with_") or tree.args.get("with")
        if with_node:
            for cte in with_node.expressions:
                cte_name = (cte.alias or "cte").lower()
                cte_df = self.query(cte.this.sql(), outer_scope=outer_scope)
                self.tables[cte_name] = cte_df
                cte_names.append(cte_name)

        try:
            df, scope = self._execute_joins(tree, outer_scope)

            # Replace uncorrelated subqueries with literals so they are not
            # re-executed row-by-row later.  Skip subqueries inside EXISTS,
            # ANY, or ALL — those need special runtime handling.
            from_node = tree.args.get("from") or tree.args.get("from_")
            local_aliases = set(scope.keys()) - set(outer_scope.keys())
            for subq in list(tree.find_all(exp.Subquery)):
                if subq.parent is from_node:
                    continue
                if isinstance(subq.parent, (exp.Exists, exp.Any, exp.All)):
                    continue
                if not self._is_correlated(subq, local_aliases, outer_scope):
                    sub_result = self.query(subq.this.sql(), outer_scope=scope)
                    self._replace_subquery_with_literal(subq, sub_result)

            plan = self._build_plan(tree, df)
            # Pass outer_scope so _execute_plan knows which scope keys must not
            # be overwritten when syncing the local DataFrame across pipeline stages.
            return self._execute_plan(df, plan, scope, outer_scope=outer_scope)
        finally:
            for name in cte_names:
                self.tables.pop(name, None)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_correlated(self, subquery, local_aliases, outer_scope) -> bool:
        for col in subquery.find_all(exp.Column):
            t = col.args.get("table")
            if t and t.this.lower() in (local_aliases | set(outer_scope.keys())):
                return True
        return False

    def _resolve_col(self, df, name: str, table_alias: str = None) -> str:
        """Return the actual column name in df matching (table_alias, name)."""
        cols = df.index if isinstance(df, pd.Series) else df.columns
        nl = name.lower()

        if table_alias:
            pref = f"{table_alias.lower()}.{nl}"
            for c in cols:
                if str(c).lower() == pref:
                    return c

        if name in cols:
            return name

        for c in cols:
            if str(c).lower() == nl:
                return c

        # suffix match: allows 'amount' to match 'orders.amount'
        for c in cols:
            if str(c).lower().endswith(f".{nl}"):
                return c

        return name  # not found — return as-is; KeyError will surface later

    def _replace_subquery_with_literal(self, subquery, sub_df: pd.DataFrame):
        parent = subquery.parent
        if isinstance(parent, exp.In):
            values = sub_df.iloc[:, 0].tolist()
            replacement = [
                exp.Literal.number(v) if isinstance(v, (int, float, np.number))
                else exp.Literal.string(str(v))
                for v in values
            ]
            parent.set("expressions", replacement)
            subquery.pop()
        else:
            if sub_df.empty:
                subquery.replace(exp.Null())
            else:
                val = sub_df.iloc[0, 0]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    subquery.replace(exp.Literal.number(float(val)))
                else:
                    subquery.replace(exp.Literal.string(str(val)))

    # ── JOIN execution ────────────────────────────────────────────────────────

    def _execute_joins(
        self, tree, outer_scope
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        from_node = tree.args.get("from") or tree.args.get("from_")
        if not from_node:
            return pd.DataFrame([{}], index=[0]), outer_scope.copy()

        scope = outer_scope.copy()

        # Handle subquery as FROM source  (e.g. FROM (SELECT ...) sub)
        from_item = from_node.this
        if isinstance(from_item, exp.Subquery):
            sub_alias = (from_item.alias or "sub").lower()
            sub_df = self.query(from_item.this.sql(), outer_scope=outer_scope)
            result_df = sub_df.copy()
            result_df.columns = [f"{sub_alias}.{c}" for c in result_df.columns]
            scope[sub_alias] = result_df
            return result_df, scope

        # Normal table FROM
        tables = list(from_node.find_all(exp.Table))
        mt = tables[0]
        main_real = mt.this.this.lower()
        main_alias = mt.alias.lower() if mt.alias else main_real

        result_df = self.tables[main_real].copy()
        result_df.columns = [f"{main_alias}.{c}" for c in result_df.columns]

        scope[main_alias] = result_df
        existing = {main_alias}

        for join in tree.find_all(exp.Join):
            jt = join.this
            jr = jt.this.this.lower()
            ja = jt.alias.lower() if jt.alias else jr

            right_df = self.tables[jr].copy()
            right_df.columns = [f"{ja}.{c}" for c in right_df.columns]

            side = str(join.args.get("side") or "").upper()
            how = "left" if "LEFT" in side else "right" if "RIGHT" in side else "inner"

            on_cond = join.args.get("on")
            if on_cond:
                lkeys, rkeys = self._extract_join_keys(
                    on_cond, existing, ja, result_df, right_df
                )
                if lkeys and rkeys:
                    result_df = pd.merge(
                        result_df, right_df, left_on=lkeys, right_on=rkeys, how=how
                    )
                else:
                    result_df = pd.merge(result_df, right_df, how="cross")

                tmp_scope = {**scope, ja: result_df}
                for a in existing:
                    tmp_scope[a] = result_df

                cond_mask = self._evaluate_expr(on_cond, tmp_scope)
                if not isinstance(cond_mask, pd.Series):
                    cond_mask = pd.Series(
                        [bool(cond_mask)] * len(result_df), index=result_df.index
                    )

                if how in ("left", "right"):
                    # Preserve unmatched rows (right join key is NaN for LEFT).
                    # Only apply the ON condition to matched rows so null rows
                    # from the outer join are not accidentally dropped.
                    ref_key = rkeys[0] if rkeys else None
                    if ref_key and ref_key in result_df.columns:
                        unmatched = result_df[ref_key].isna()
                        final_mask = unmatched | cond_mask.fillna(False)
                    else:
                        final_mask = cond_mask.fillna(True)
                    result_df = result_df[final_mask].reset_index(drop=True)
                else:
                    result_df = result_df[cond_mask.fillna(False)].reset_index(drop=True)
            else:
                result_df = result_df.merge(right_df, how="cross")

            existing.add(ja)
            for a in list(scope.keys()):
                if isinstance(scope[a], pd.DataFrame):
                    scope[a] = result_df
            scope[ja] = result_df

        return result_df, scope

    def _extract_join_keys(
        self, on_cond, existing, right_alias, left_df, right_df
    ) -> Tuple[List[str], List[str]]:
        lkeys, rkeys = [], []
        for eq in on_cond.find_all(exp.EQ):
            lc, rc = eq.left, eq.right
            # Only treat as equi-join key when BOTH sides are column references.
            # Literal comparisons (e.g. r.rn = 1) are residual conditions handled
            # as post-join filters, not as merge keys.
            if not isinstance(lc, exp.Column) or not isinstance(rc, exp.Column):
                continue
            lt = lc.table.lower() if getattr(lc, "table", None) else ""
            rt = rc.table.lower() if getattr(rc, "table", None) else ""
            ln = getattr(lc, "name", lc.sql())
            rn = getattr(rc, "name", rc.sql())
            if rt == right_alias:
                lkeys.append(self._resolve_col(left_df, ln, lt))
                rkeys.append(self._resolve_col(right_df, rn, rt))
            elif lt == right_alias:
                lkeys.append(self._resolve_col(left_df, rn, rt))
                rkeys.append(self._resolve_col(right_df, ln, lt))
            else:
                if f"{right_alias}.{rn}" in right_df.columns:
                    rkeys.append(f"{right_alias}.{rn}")
                    lkeys.append(self._resolve_col(left_df, ln, lt))
                else:
                    rkeys.append(f"{right_alias}.{ln}")
                    lkeys.append(self._resolve_col(left_df, rn, rt))
        return lkeys, rkeys

    # ── plan building ─────────────────────────────────────────────────────────

    def _build_plan(self, tree, df: pd.DataFrame) -> QueryPlan:
        """
        Build QueryPlan and populate resolution_map before any execution.

        resolution_map maps SQL expression strings → physical column names so
        that ORDER BY and HAVING can look up pre-computed aggregates without
        re-computing them.
        """
        group_node = tree.args.get("group") or tree.args.get("group_")
        limit_node = tree.args.get("limit")

        plan = QueryPlan(
            select=tree.expressions,
            where=tree.args.get("where"),
            group_by_exprs=group_node.expressions if group_node else [],
            having=tree.args.get("having"),
            order_by=tree.args.get("order"),
            limit=int(limit_node.expression.this) if limit_node else None,
            distinct=bool(tree.args.get("distinct")),
        )

        # Collect SELECT aliases: inner_sql → alias_name
        select_alias: Dict[str, str] = {}
        for s in plan.select:
            if isinstance(s, exp.Alias):
                select_alias[s.this.sql()] = s.alias
                plan.resolution_map[s.alias] = s.alias  # "total" → "total"

        # Scan SELECT + HAVING + ORDER BY for aggregate nodes
        scan_targets = list(plan.select)
        if plan.having:
            scan_targets.append(plan.having)
        if plan.order_by:
            scan_targets.extend(plan.order_by.expressions)

        seen: Dict[str, str] = {}
        for target in scan_targets:
            for agg in target.find_all(tuple(self.AGG_MAP.keys())):
                raw = agg.sql()
                if raw in seen:
                    continue

                alias = select_alias.get(raw)
                out_col = alias if alias else raw

                is_distinct = isinstance(agg.this, exp.Distinct)
                if isinstance(agg, exp.Count) and is_distinct:
                    func = "nunique"
                    if len(agg.this.expressions) != 1:
                        raise NotImplementedError(
                            "COUNT(DISTINCT col1, col2) not supported"
                        )
                    inner = agg.this.expressions[0]
                else:
                    func = self.AGG_MAP[type(agg)]
                    inner = agg.this

                if isinstance(inner, (exp.Star, exp.Literal)):
                    src = "__count_star__"
                elif isinstance(inner, exp.Column):
                    ta = inner.table.lower() if inner.table else None
                    src = self._resolve_col(df, inner.name, ta)
                    if src not in df.columns:
                        src = "__expr__"
                else:
                    src = "__expr__"

                plan.resolution_map[raw] = out_col
                if alias:
                    plan.resolution_map[alias] = out_col

                plan.agg_specs.append(
                    AggSpec(source_sql=raw, func=func, source_col=src, output_col=out_col)
                )
                seen[raw] = out_col

        # Scan SELECT for window functions
        _WIN_FUNCS = {
            exp.RowNumber: "row_number",
            exp.Rank: "rank",
            exp.DenseRank: "dense_rank",
            exp.Sum: "sum",
            exp.Avg: "mean",
            exp.Count: "count",
            exp.Min: "min",
            exp.Max: "max",
        }
        win_counter = 0
        for expr in plan.select:
            for win in expr.find_all(exp.Window):
                raw = win.sql()
                if raw in plan.resolution_map:
                    continue
                func_node = win.this
                func = _WIN_FUNCS.get(type(func_node))
                if func is None:
                    continue

                func_arg: Optional[str] = None
                if func in ("sum", "mean", "min", "max"):
                    if isinstance(func_node.this, exp.Column):
                        inner = func_node.this
                        ta = inner.table.lower() if inner.table else None
                        func_arg = self._resolve_col(df, inner.name, ta)
                elif func == "count":
                    inner = func_node.this
                    if isinstance(inner, (exp.Star, exp.Literal)) or inner is None:
                        func_arg = "__count_star__"
                    elif isinstance(inner, exp.Distinct):
                        func_arg = "__count_star__"
                    elif isinstance(inner, exp.Column):
                        ta = inner.table.lower() if inner.table else None
                        func_arg = self._resolve_col(df, inner.name, ta)
                    else:
                        func_arg = "__count_star__"

                # Resolve PARTITION BY columns
                partition_cols: List[str] = []
                pb_raw = win.args.get("partition_by") or []
                pb_list = pb_raw.expressions if hasattr(pb_raw, "expressions") else (
                    pb_raw if isinstance(pb_raw, list) else [pb_raw] if pb_raw else []
                )
                for p in pb_list:
                    if isinstance(p, exp.Column):
                        ta = p.table.lower() if p.table else None
                        partition_cols.append(self._resolve_col(df, p.name, ta))
                    else:
                        partition_cols.append(p.sql())

                # Resolve ORDER BY columns
                order_cols: List[str] = []
                order_asc: List[bool] = []
                order_node = win.args.get("order")
                if order_node:
                    for ord_expr in order_node.expressions:
                        actual = ord_expr.this
                        asc = not ord_expr.args.get("desc", False)
                        if isinstance(actual, exp.Column):
                            ta = actual.table.lower() if actual.table else None
                            order_cols.append(self._resolve_col(df, actual.name, ta))
                        else:
                            order_cols.append(actual.sql())
                        order_asc.append(asc)

                out_col = f"__win_{win_counter}__"
                win_counter += 1
                plan.resolution_map[raw] = out_col
                plan.window_specs.append(WindowSpec(
                    source_sql=raw,
                    func=func,
                    func_arg=func_arg,
                    partition_cols=partition_cols,
                    order_cols=order_cols,
                    order_asc=order_asc,
                    output_col=out_col,
                ))

        # Resolve GROUP BY columns to physical names
        for i, g in enumerate(plan.group_by_exprs):
            g_sql = g.sql()
            if isinstance(g, exp.Column):
                ta = g.table.lower() if g.table else None
                resolved = self._resolve_col(df, g.name, ta)
                plan.group_cols.append(resolved)
                plan.resolution_map[g_sql] = resolved
            else:
                tmp = f"__grp_{i}__"
                plan.group_cols.append(tmp)
                plan.resolution_map[g_sql] = tmp

        return plan

    # ── execution ─────────────────────────────────────────────────────────────

    def _execute_plan(
        self,
        df: pd.DataFrame,
        plan: QueryPlan,
        scope: Dict,
        outer_scope: Dict = None,
    ) -> pd.DataFrame:
        # Sync rule: update every scope entry whose current value is a DataFrame
        # (local tables and outer DataFrame references alike) but NEVER overwrite
        # entries that are pd.Series — those are outer-row bindings from correlated
        # subqueries and must stay fixed for the duration of this inner execution.
        def sync(d):
            for k in list(scope.keys()):
                if not isinstance(scope[k], pd.Series):
                    scope[k] = d

        # 1. WHERE
        if plan.where:
            sync(df)
            mask = self._eval_condition(plan.where.this, scope)
            if not isinstance(mask, (pd.Series, np.ndarray)):
                mask = np.full(len(df), bool(mask))
            vals = mask.values if isinstance(mask, pd.Series) else mask
            df = df[vals].reset_index(drop=True)
            sync(df)

        # 2. GROUP BY + aggregation
        if plan.group_by_exprs:
            df = self._execute_groupby(df, plan, scope)

        # 2.5 Window functions — computed after GROUP BY, before SELECT
        if plan.window_specs:
            df = self._execute_windows(df, plan, scope)
            sync(df)

        # 3. HAVING
        if plan.having:
            sync(df)
            mask = self._eval_condition(plan.having.this, scope, plan.resolution_map)
            if not isinstance(mask, (pd.Series, np.ndarray)):
                mask = np.full(len(df), bool(mask))
            vals = mask.values if isinstance(mask, pd.Series) else mask
            df = df[vals].reset_index(drop=True)

        # Keep pre-projection df for ORDER BY fallback (non-projected cols)
        pre_proj_df = df

        # 4. SELECT projection
        sync(df)
        df = self._execute_select(df, plan, scope)
        # Drop internal window temp columns (visible only when SELECT * is used)
        df = df.drop(
            columns=[c for c in df.columns if str(c).startswith("__win_")],
            errors="ignore",
        )

        # 5. DISTINCT
        if plan.distinct:
            df = df.drop_duplicates().reset_index(drop=True)

        # 6. ORDER BY — runs after SELECT so aliases are visible
        if plan.order_by:
            sync(df)
            pre_scope = {k: pre_proj_df for k in scope}  # for non-projected cols
            sort_cols: List[str] = []
            sort_asc: List[bool] = []

            for i, ord_node in enumerate(plan.order_by.expressions):
                actual = ord_node.this
                raw = actual.sql()
                asc = not ord_node.args.get("desc", False)

                # a) resolution_map → projected df (covers aliases + agg outputs)
                resolved = plan.resolution_map.get(raw)
                if resolved and resolved in df.columns:
                    sort_cols.append(resolved)
                    sort_asc.append(asc)
                    continue

                # b) direct column in projected df
                if isinstance(actual, exp.Column):
                    ta = actual.table.lower() if actual.table else None
                    col = self._resolve_col(df, actual.name, ta)
                    if col in df.columns:
                        sort_cols.append(col)
                        sort_asc.append(asc)
                        continue

                # c) direct name match in projected df
                if raw in df.columns:
                    sort_cols.append(raw)
                    sort_asc.append(asc)
                    continue

                # d) fallback: compute from pre-projection df
                tmp = f"__sort_{i}__"
                try:
                    val = self._evaluate_expr(actual, pre_scope, plan.resolution_map)
                    if isinstance(val, pd.Series):
                        df[tmp] = val.values[: len(df)]
                    else:
                        df[tmp] = val
                    sort_cols.append(tmp)
                    sort_asc.append(asc)
                except Exception:
                    pass  # skip unresolvable sort key

            if sort_cols:
                valid = [(c, a) for c, a in zip(sort_cols, sort_asc) if c in df.columns]
                if valid:
                    cols, ascs = zip(*valid)
                    df = df.sort_values(
                        by=list(cols), ascending=list(ascs)
                    ).reset_index(drop=True)
                df = df.drop(
                    columns=[c for c in df.columns if str(c).startswith("__sort_")],
                    errors="ignore",
                )

        # 7. LIMIT
        if plan.limit is not None:
            df = df.head(plan.limit)

        return df

    def _execute_groupby(
        self, df: pd.DataFrame, plan: QueryPlan, scope: Dict
    ) -> pd.DataFrame:
        # Compute expression-based group keys and attach as temp columns
        for i, g in enumerate(plan.group_by_exprs):
            if not isinstance(g, exp.Column):
                tmp = f"__grp_{i}__"
                val = self._evaluate_expr(g, scope)
                df = df.copy()
                df[tmp] = val.values if isinstance(val, pd.Series) else val

        named_aggs: Dict[str, pd.NamedAgg] = {}
        for spec in plan.agg_specs:
            if spec.output_col in named_aggs:
                continue  # already registered

            if spec.source_col == "__count_star__":
                src = plan.group_cols[0] if plan.group_cols else df.columns[0]
                named_aggs[spec.output_col] = pd.NamedAgg(column=src, aggfunc="count")

            elif spec.source_col == "__expr__" or spec.source_col not in df.columns:
                # Expression aggregate: evaluate the inner expression first
                agg_node = self._find_agg_node(spec.source_sql, plan)
                if agg_node is not None:
                    is_dist = isinstance(agg_node.this, exp.Distinct)
                    inner = (
                        agg_node.this.expressions[0] if is_dist else agg_node.this
                    )
                    tmp = f"__agg_tmp_{len(named_aggs)}__"
                    val = self._evaluate_expr(inner, scope)
                    df = df.copy()
                    df[tmp] = val.values if isinstance(val, pd.Series) else val
                    named_aggs[spec.output_col] = pd.NamedAgg(
                        column=tmp, aggfunc=spec.func
                    )
            else:
                named_aggs[spec.output_col] = pd.NamedAgg(
                    column=spec.source_col, aggfunc=spec.func
                )

        agg_df = df.groupby(plan.group_cols, as_index=False).agg(**named_aggs)

        # Drop all internal temp columns
        agg_df = agg_df.drop(
            columns=[c for c in agg_df.columns if str(c).startswith("__")],
            errors="ignore",
        )

        for k in list(scope.keys()):
            if not isinstance(scope[k], pd.Series):
                scope[k] = agg_df
        return agg_df

    def _find_agg_node(self, source_sql: str, plan: QueryPlan):
        """Locate the AST aggregate node whose .sql() == source_sql."""
        targets = list(plan.select)
        if plan.having:
            targets.append(plan.having)
        if plan.order_by:
            targets.extend(plan.order_by.expressions)
        for t in targets:
            for node in t.find_all(tuple(self.AGG_MAP.keys())):
                if node.sql() == source_sql:
                    return node
        return None

    def _execute_windows(
        self, df: pd.DataFrame, plan: QueryPlan, scope: Dict
    ) -> pd.DataFrame:
        df = df.copy()
        for spec in plan.window_specs:
            pc = spec.partition_cols
            oc = spec.order_cols
            oa = spec.order_asc

            if spec.func == "row_number":
                # Preserve original row order: sort to compute, then restore.
                orig_idx = "__win_orig_idx__"
                df[orig_idx] = range(len(df))
                if oc:
                    sort_cols = (pc if pc else []) + oc
                    sort_asc = ([True] * len(pc)) + oa
                    df = df.sort_values(sort_cols, ascending=sort_asc)
                if pc:
                    df[spec.output_col] = df.groupby(pc, sort=False).cumcount() + 1
                else:
                    df[spec.output_col] = range(1, len(df) + 1)
                df = (
                    df.sort_values(orig_idx)
                    .drop(columns=[orig_idx])
                    .reset_index(drop=True)
                )

            elif spec.func in ("rank", "dense_rank"):
                method = "min" if spec.func == "rank" else "dense"
                if oc:
                    col_to_rank = oc[0]
                    asc = oa[0] if oa else True
                    if pc:
                        df[spec.output_col] = df.groupby(pc)[col_to_rank].rank(
                            method=method, ascending=asc
                        )
                    else:
                        df[spec.output_col] = df[col_to_rank].rank(
                            method=method, ascending=asc
                        )
                else:
                    df[spec.output_col] = 1.0

            elif spec.func in ("sum", "mean", "min", "max") and spec.func_arg:
                if pc:
                    df[spec.output_col] = df.groupby(pc)[spec.func_arg].transform(
                        spec.func
                    )
                else:
                    agg_val = getattr(df[spec.func_arg], spec.func)()
                    df[spec.output_col] = agg_val

            elif spec.func == "count":
                src = spec.func_arg
                if pc:
                    ref_col = pc[0] if (src is None or src == "__count_star__") else src
                    df[spec.output_col] = df.groupby(pc)[ref_col].transform("count")
                else:
                    if src is None or src == "__count_star__":
                        df[spec.output_col] = len(df)
                    else:
                        df[spec.output_col] = df[src].count()

        return df

    def _execute_select(
        self,
        df: pd.DataFrame,
        plan: QueryPlan,
        scope: Dict,
        preserve: List[str] = None,
    ) -> pd.DataFrame:
        if any(isinstance(e, exp.Star) for e in plan.select):
            return df

        for k in list(scope.keys()):
            if not isinstance(scope[k], pd.Series):
                scope[k] = df

        has_subquery = any(list(e.find_all(exp.Subquery)) for e in plan.select)

        if not has_subquery:
            new_cols: Dict[str, Any] = {}
            order: List[str] = []
            for expr in plan.select:
                is_alias = isinstance(expr, exp.Alias)
                inner = expr.this if is_alias else expr
                if is_alias:
                    name = expr.alias
                elif isinstance(inner, exp.Column):
                    name = inner.name  # use bare column name, not "u.name"
                else:
                    name = inner.sql()
                val = self._evaluate_expr(inner, scope, plan.resolution_map)
                new_cols[name] = val
                order.append(name)

            all_scalar = all(
                not isinstance(v, (pd.Series, np.ndarray)) for v in new_cols.values()
            )
            # If all values are scalars but WHERE filtered the table to empty,
            # return an empty frame rather than a phantom one-row result.
            # Exception: aggregate queries (agg_specs non-empty) always return
            # one row even over an empty table (SQL standard for bare aggregates).
            if all_scalar and len(df) == 0 and not plan.agg_specs:
                return pd.DataFrame(columns=order)
            idx = [0] if all_scalar else df.index
            result = pd.DataFrame(new_cols, index=idx)[order]
        else:
            def process_row(row):
                rs = {k: row for k in scope}
                out: Dict[str, Any] = {}
                for e in plan.select:
                    ia = isinstance(e, exp.Alias)
                    inner = e.this if ia else e
                    if ia:
                        n = e.alias
                    elif isinstance(inner, exp.Column):
                        n = inner.name
                    else:
                        n = inner.sql()
                    v = self._evaluate_expr(inner, rs, plan.resolution_map)
                    if isinstance(v, (pd.DataFrame, pd.Series)):
                        v = v.iloc[0, 0] if not v.empty else 0
                    elif isinstance(v, np.ndarray):
                        v = v[0] if v.size > 0 else 0
                    out[n] = v
                return pd.Series(out)

            result = df.apply(process_row, axis=1)

        # Re-attach any temp columns that must survive into ORDER BY
        if preserve:
            for col in preserve:
                if col in df.columns and col not in result.columns:
                    result[col] = df[col].values

        return result

    # ── expression evaluation ─────────────────────────────────────────────────

    def _evaluate_expr(self, expr, scope, resolution_map=None):  # noqa: C901
        if expr is None or isinstance(expr, exp.Null):
            return None

        raw = expr.sql()

        # Fast path: resolution_map → pre-computed column in current df
        if resolution_map:
            mapped = resolution_map.get(raw)
            if mapped:
                target = list(scope.values())[0]
                if isinstance(target, pd.DataFrame) and mapped in target.columns:
                    return target[mapped]
                elif isinstance(target, pd.Series) and mapped in target.index:
                    return target[mapped]

        # Subquery
        if isinstance(expr, exp.Subquery):
            r = self.query(expr.this.sql(), outer_scope=scope)
            return r.iloc[0, 0] if not r.empty else None

        # Column reference
        if isinstance(expr, exp.Column):
            ta = expr.table.lower() if expr.table else None
            # When a table alias is given, look it up in scope directly so that
            # correlated outer references (e.g. p.user_id in a subquery where
            # scope["p"] is the current row) are resolved from the right source.
            if ta and ta in scope:
                target = scope[ta]
            else:
                # For unqualified columns, prefer the inner query's own DataFrame
                # over any outer-row Series injected via outer_scope.
                target = next(
                    (v for v in scope.values() if isinstance(v, pd.DataFrame)),
                    list(scope.values())[0],
                )
            col = self._resolve_col(target, expr.name, ta)
            if isinstance(target, pd.DataFrame):
                return target[col]
            return target.get(col) if isinstance(target, pd.Series) else None

        # COUNT(DISTINCT ...)
        if isinstance(expr, exp.Count) and isinstance(expr.this, exp.Distinct):
            inner = expr.this.expressions[0]
            s = self._evaluate_expr(inner, scope, resolution_map)
            if isinstance(s, np.ndarray):
                s = pd.Series(s)
            if not isinstance(s, pd.Series):
                return 1 if s is not None else 0
            return s.nunique(dropna=True)

        # Aggregates
        if type(expr) in self.AGG_MAP:
            method = self.AGG_MAP[type(expr)]
            primary = list(scope.values())[-1]
            if isinstance(expr.this, (exp.Star, exp.Literal)):
                n = len(primary) if isinstance(primary, pd.DataFrame) else 1
                return n if method == "count" else 0
            s = self._evaluate_expr(expr.this, scope, resolution_map)
            return getattr(s, method)() if hasattr(s, method) else s

        # Literal
        if isinstance(expr, exp.Literal):
            if expr.is_string:
                return expr.this
            try:
                return float(expr.this) if "." in expr.this else int(expr.this)
            except ValueError:
                return expr.this

        # CASE
        if isinstance(expr, exp.Case):
            return self._execute_case(expr, scope, resolution_map)

        # Boolean connectives
        if isinstance(expr, (exp.And, exp.Or)):
            l = self._evaluate_expr(expr.left, scope, resolution_map)
            r = self._evaluate_expr(expr.right, scope, resolution_map)
            return (l & r) if isinstance(expr, exp.And) else (l | r)

        # Comparisons
        if isinstance(expr, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            l = self._evaluate_expr(expr.left, scope, resolution_map)
            r = self._evaluate_expr(expr.right, scope, resolution_map)
            if isinstance(expr, exp.EQ):  return l == r
            if isinstance(expr, exp.NEQ): return l != r
            if isinstance(expr, exp.GT):  return l > r
            if isinstance(expr, exp.GTE): return l >= r
            if isinstance(expr, exp.LT):  return l < r
            if isinstance(expr, exp.LTE): return l <= r

        # Unary NOT
        if isinstance(expr, exp.Not):
            val = self._evaluate_expr(expr.this, scope, resolution_map)
            return ~val if hasattr(val, "__invert__") else not val

        # Parenthesis
        if isinstance(expr, exp.Paren):
            return self._evaluate_expr(expr.this, scope, resolution_map)

        # Arithmetic
        if isinstance(expr, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            l = self._evaluate_expr(expr.left, scope, resolution_map)
            r = self._evaluate_expr(expr.right, scope, resolution_map)
            if l is None or r is None:
                return None
            if isinstance(expr, exp.Add): return l + r
            if isinstance(expr, exp.Sub): return l - r
            if isinstance(expr, exp.Mul): return l * r
            if isinstance(expr, exp.Div): return l / r

        return expr.sql()

    def _eval_condition(self, cond, scope, resolution_map=None):
        if isinstance(cond, exp.Paren):
            return self._eval_condition(cond.this, scope, resolution_map)

        # AND / OR
        if isinstance(cond, (exp.And, exp.Or)):
            l = self._eval_condition(cond.left, scope, resolution_map)
            r = self._eval_condition(cond.right, scope, resolution_map)
            return (l & r) if isinstance(cond, exp.And) else (l | r)

        # NOT — handles both "NOT expr" and the IS NOT NULL case
        if isinstance(cond, exp.Not):
            inner = self._eval_condition(cond.this, scope, resolution_map)
            return ~inner if hasattr(inner, "__invert__") else not inner

        # IS NULL / IS NOT NULL
        if isinstance(cond, exp.Is):
            l_val = self._evaluate_expr(cond.left, scope, resolution_map)
            rhs = cond.args.get("expression")
            if isinstance(rhs, exp.Null):
                return l_val.isna() if hasattr(l_val, "isna") else l_val is None
            if isinstance(rhs, exp.Not) and isinstance(rhs.this, exp.Null):
                return l_val.notna() if hasattr(l_val, "notna") else l_val is not None
            return l_val == self._evaluate_expr(rhs, scope, resolution_map)

        # EXISTS / NOT EXISTS
        if isinstance(cond, exp.Exists):
            subq = cond.this
            sql = subq.this.sql() if isinstance(subq, exp.Subquery) else subq.sql()
            primary = list(scope.values())[0]
            is_row = isinstance(primary, pd.Series)
            if is_row:
                # Already in row-wise mode: execute subquery with this row as outer scope
                r = self.query(sql, outer_scope=scope)
                return len(r) > 0
            # Vectorised mode: check correlation and dispatch accordingly
            local_aliases = set(scope.keys())
            if self._is_correlated(subq, local_aliases, {}):
                def _row_exists(row):
                    rs = {k: row for k in scope}
                    return self._eval_condition(cond, rs, resolution_map)
                return primary.apply(_row_exists, axis=1).astype(bool)
            # Non-correlated: execute once and broadcast
            r = self.query(sql, outer_scope=scope)
            return np.full(len(primary), len(r) > 0)

        # Row-level evaluation for correlated subqueries
        primary = list(scope.values())[0]
        is_row = isinstance(primary, pd.Series)
        if not is_row and any(
            isinstance(n, exp.Subquery) for n in cond.find_all(exp.Subquery)
        ):
            def row_eval(row):
                rs = {k: row for k in scope}
                return self._eval_condition(cond, rs, resolution_map)

            return primary.apply(row_eval, axis=1).astype(bool)

        # Left operand
        left_node = cond.this if isinstance(cond, exp.In) else cond.left
        l_val = self._evaluate_expr(left_node, scope, resolution_map)

        # IN
        if isinstance(cond, exp.In):
            subq = cond.args.get("query") or cond.args.get("field")
            if subq:
                r_val = self._evaluate_expr(subq, scope, resolution_map)
            else:
                r_val = [
                    self._evaluate_expr(e, scope, resolution_map)
                    for e in cond.args.get("expressions", [])
                ]
            if hasattr(r_val, "tolist"):
                r_val = r_val.tolist()
            elif hasattr(r_val, "values"):
                r_val = r_val.values.flatten()
            return l_val.isin(r_val) if hasattr(l_val, "isin") else l_val in r_val

        # Standard comparisons
        r_node = cond.args.get("expression") or cond.args.get("right")

        # ANY / ALL quantifiers: x > ANY (subquery) / x > ALL (subquery)
        if isinstance(r_node, (exp.Any, exp.All)):
            subq = r_node.this
            sql = subq.this.sql() if isinstance(subq, exp.Subquery) else subq.sql()
            sub_df = self.query(sql, outer_scope=scope)
            vals = sub_df.iloc[:, 0].tolist() if not sub_df.empty else []
            is_any = isinstance(r_node, exp.Any)

            if not vals:
                # ANY over empty set → False; ALL over empty set → True
                result_scalar = not is_any
                if isinstance(l_val, pd.Series):
                    return pd.Series([result_scalar] * len(l_val), index=l_val.index)
                return result_scalar

            def _quantify(lv):
                results = []
                for v in vals:
                    if isinstance(cond, exp.GT):  results.append(lv > v)
                    elif isinstance(cond, exp.GTE): results.append(lv >= v)
                    elif isinstance(cond, exp.LT):  results.append(lv < v)
                    elif isinstance(cond, exp.LTE): results.append(lv <= v)
                    elif isinstance(cond, exp.EQ):  results.append(lv == v)
                    elif isinstance(cond, exp.NEQ): results.append(lv != v)
                return any(results) if is_any else all(results)

            if isinstance(l_val, pd.Series):
                return l_val.apply(_quantify)
            return _quantify(l_val)

        r_val = self._evaluate_expr(r_node, scope, resolution_map)

        if isinstance(cond, exp.EQ):  return l_val == r_val
        if isinstance(cond, exp.NEQ): return l_val != r_val
        if isinstance(cond, exp.GT):  return l_val > r_val
        if isinstance(cond, exp.LT):  return l_val < r_val
        if isinstance(cond, exp.GTE): return l_val >= r_val
        if isinstance(cond, exp.LTE): return l_val <= r_val

        return False

    def _execute_case(self, expr, scope, resolution_map=None):
        conditions, choices = [], []
        for if_node in expr.args.get("ifs", []):
            cond_mask = self._eval_condition(if_node.this, scope, resolution_map)
            then = if_node.args.get("true") or if_node.args.get("then")
            val = self._evaluate_expr(then, scope, resolution_map)
            conditions.append(cond_mask)
            choices.append(val)

        default_expr = expr.args.get("default")
        default_val = (
            self._evaluate_expr(default_expr, scope, resolution_map)
            if default_expr
            else None
        )

        def to_obj(x):
            return x.astype(object) if isinstance(x, (pd.Series, np.ndarray)) else x

        choices = [to_obj(c) for c in choices]
        default_val = to_obj(default_val)

        if conditions and all(
            isinstance(c, (pd.Series, np.ndarray)) for c in conditions
        ):
            return np.select(conditions, choices, default=default_val)

        # Scalar fallback (row-wise execution)
        for cond, val in zip(conditions, choices):
            if cond:
                return val
        return default_val
