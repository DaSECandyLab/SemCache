SELECT
  "ol"."l_orderkey" AS "l_orderkey",
  "ol"."l_linenumber" AS "l_linenumber",
  "s"."s_suppkey" AS "s_suppkey"
FROM "raw_part" AS "p"
JOIN (
  SELECT
    "ps"."ps_partkey" AS "ps_partkey",
    "ps"."ps_suppkey" AS "ps_suppkey",
    "ps"."ps_availqty" AS "ps_availqty",
    "ps"."ps_supplycost" AS "ps_supplycost",
    "s"."s_name" AS "s_name",
    "s"."s_acctbal" AS "s_acctbal",
    "s"."s_comment" AS "s_comment",
    "s"."s_nationkey" AS "s_nationkey",
    "n"."n_name" AS "nation_name",
    "r"."r_name" AS "region_name"
  FROM "raw_partsupp" AS "ps"
  JOIN "raw_supplier" AS "s"
    ON "s"."s_suppkey" = "ps"."ps_suppkey"
  JOIN "raw_nation" AS "n"
    ON "n"."n_nationkey" = "s"."s_nationkey"
  JOIN "raw_region" AS "r"
    ON "r"."r_regionkey" = "n"."n_regionkey"
  WHERE
    "ps"."ps_availqty" BETWEEN 10 AND 9000
    AND "ps"."ps_supplycost" BETWEEN 10 AND 950
    AND "r"."r_name" IN ('EUROPE', 'ASIA', 'AMERICA')
) AS "sb"
  ON "sb"."ps_partkey" = "p"."p_partkey"
JOIN (
  SELECT
    "l"."l_partkey" AS "l_partkey",
    "l"."l_suppkey" AS "l_suppkey",
    "l"."l_orderkey" AS "l_orderkey",
    "l"."l_quantity" AS "l_quantity",
    "l"."l_extendedprice" AS "l_extendedprice",
    "l"."l_discount" AS "l_discount",
    "o"."o_orderdate" AS "o_orderdate",
    "o"."o_orderpriority" AS "o_orderpriority",
    "o"."o_totalprice" AS "o_totalprice",
    "l"."l_linenumber" AS "l_linenumber"
  FROM "raw_lineitem" AS "l"
  JOIN "raw_orders" AS "o"
    ON "o"."o_orderkey" = "l"."l_orderkey"
  WHERE
    "o"."o_orderdate" >= CAST('1993-01-01' AS DATE)
    AND "o"."o_orderdate" < CAST('1997-01-01' AS DATE)
    AND "l"."l_quantity" BETWEEN 5 AND 45
    AND "l"."l_discount" BETWEEN 0.0 AND 0.1
) AS "ol"
  ON "ol"."l_partkey" = "sb"."ps_partkey" AND "ol"."l_suppkey" = "sb"."ps_suppkey"
JOIN "raw_supplier" AS "s"
  ON (
    "s"."s_suppkey" = "sb"."ps_suppkey"
    AND SEMANTIC_JOIN(
      'Is supplier balance {1} at least required line reserve {0}?',
      CAST((
        0.35 * "ol"."l_extendedprice"
      ) AS TEXT),
      CAST("s"."s_acctbal" AS TEXT)
    )
  )
  AND SEMANTIC_FILTER('Is supplier balance {0} between 2000 and 6000?', CAST("s"."s_acctbal" AS TEXT))
WHERE
  SEMANTIC_FILTER('Does part type {0} contain COPPER or BRASS?', CAST("p"."p_type" AS TEXT));
