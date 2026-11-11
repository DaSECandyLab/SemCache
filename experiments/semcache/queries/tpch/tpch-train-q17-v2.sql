SELECT
  "o"."o_orderkey" AS "o_orderkey",
  "l"."l_linenumber" AS "l_linenumber",
  "raw_customer"."c_custkey" AS "c_custkey"
FROM "raw_orders" AS "o"
JOIN "raw_lineitem" AS "l"
  ON "l"."l_orderkey" = "o"."o_orderkey"
  AND (
    "l"."l_shipdate" >= CAST('1994-01-01' AS DATE)
    AND "l"."l_shipdate" < CAST('1998-01-01' AS DATE)
    AND "l"."l_quantity" BETWEEN 6 AND 46
    AND "l"."l_discount" BETWEEN 0.0 AND 0.1
    AND "l"."l_shipmode" IN ('AIR', 'REG AIR', 'TRUCK', 'SHIP')
  )
JOIN (
  SELECT
    "p"."p_partkey" AS "p_partkey",
    "p"."p_name" AS "p_name",
    "p"."p_type" AS "p_type",
    "p"."p_brand" AS "p_brand",
    "s"."s_suppkey" AS "s_suppkey",
    "s"."s_name" AS "s_name",
    "n"."n_name" AS "supplier_nation"
  FROM "raw_part" AS "p"
  JOIN "raw_partsupp" AS "ps"
    ON "ps"."ps_partkey" = "p"."p_partkey"
  JOIN "raw_supplier" AS "s"
    ON "s"."s_suppkey" = "ps"."ps_suppkey"
  JOIN "raw_nation" AS "n"
    ON "n"."n_nationkey" = "s"."s_nationkey"
  WHERE
    "p"."p_size" BETWEEN 3 AND 47 AND "ps"."ps_supplycost" BETWEEN 10 AND 950
) AS "ps"
  ON "ps"."p_partkey" = "l"."l_partkey" AND "ps"."s_suppkey" = "l"."l_suppkey"
JOIN "raw_customer" AS "raw_customer"
  ON (
    "o"."o_custkey" = "raw_customer"."c_custkey"
    AND SEMANTIC_FILTER('Is customer segment {0} AUTOMOBILE?', CAST("raw_customer"."c_mktsegment" AS TEXT))
  )
  AND SEMANTIC_FILTER(
    'Does customer comment {0} mention requests or accounts?',
    CAST("raw_customer"."c_comment" AS TEXT)
  )
WHERE
  "o"."o_orderdate" >= CAST('1994-01-01' AS DATE)
  AND "o"."o_orderdate" < CAST('1998-01-01' AS DATE)
  AND "o"."o_orderstatus" IN ('O', 'F')
  AND "o"."o_totalprice" > 41700
  AND SEMANTIC_FILTER(
    'Does order comment {0} mention requests or accounts?',
    CAST("o"."o_comment" AS TEXT)
  );
