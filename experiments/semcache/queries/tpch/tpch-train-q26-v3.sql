SELECT
  "l"."l_orderkey" AS "l_orderkey",
  "l"."l_linenumber" AS "l_linenumber",
  "raw_customer"."c_custkey" AS "c_custkey"
FROM "raw_lineitem" AS "l"
JOIN "raw_orders" AS "o"
  ON "o"."o_orderkey" = "l"."l_orderkey"
JOIN "raw_partsupp" AS "ps"
  ON (
    "ps"."ps_partkey" = "l"."l_partkey" AND "ps"."ps_suppkey" = "l"."l_suppkey"
  )
  AND (
    "ps"."ps_supplycost" BETWEEN 10 AND 950 AND "ps"."ps_availqty" >= 50
  )
JOIN "raw_part" AS "p"
  ON "p"."p_partkey" = "l"."l_partkey"
  AND SEMANTIC_FILTER('Is part size {0} between 10 and 40?', CAST("p"."p_size" AS TEXT))
JOIN (
  SELECT
    "s"."s_suppkey" AS "s_suppkey",
    "s"."s_name" AS "s_name",
    "s"."s_acctbal" AS "s_acctbal",
    "s"."s_comment" AS "s_comment",
    "n"."n_name" AS "supplier_nation",
    "r"."r_name" AS "supplier_region"
  FROM "raw_supplier" AS "s"
  JOIN "raw_nation" AS "n"
    ON "n"."n_nationkey" = "s"."s_nationkey"
  JOIN "raw_region" AS "r"
    ON "r"."r_regionkey" = "n"."n_regionkey"
  WHERE
    "s"."s_acctbal" > 0
    AND SEMANTIC_FILTER(
      'Does supplier comment {0} contain regular or quick?',
      CAST("s"."s_comment" AS TEXT)
    )
) AS "s"
  ON "s"."s_suppkey" = "l"."l_suppkey"
JOIN "raw_customer" AS "raw_customer"
  ON "raw_customer"."c_custkey" = "o"."o_custkey"
  AND (
    SEMANTIC_FILTER(
      'Is customer balance {0} between 2000 and 6000?',
      CAST("raw_customer"."c_acctbal" AS TEXT)
    )
    AND "raw_customer"."c_acctbal" > -500
  )
WHERE
  "l"."l_shipdate" >= CAST('1995-01-01' AS DATE)
  AND "l"."l_shipdate" < CAST('1999-01-01' AS DATE)
  AND "l"."l_discount" BETWEEN 0.0 AND 0.1
  AND "l"."l_quantity" BETWEEN 7 AND 47
  AND "l"."l_shipmode" IN ('AIR', 'REG AIR', 'TRUCK', 'SHIP')
  AND "l"."l_shipinstruct" IN ('DELIVER IN PERSON', 'TAKE BACK RETURN')
  AND "o"."o_orderstatus" = 'O'
  AND "o"."o_orderpriority" IN ('1-URGENT', '2-HIGH')
  AND "o"."o_totalprice" > 52600
  AND SEMANTIC_FILTER(
    'Is order date {0} between 1995-01-01 and 1996-12-31?',
    CAST("o"."o_orderdate" AS TEXT)
  );
