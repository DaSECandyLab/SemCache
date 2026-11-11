SELECT
  "o"."o_orderkey" AS "o_orderkey",
  "l"."l_linenumber" AS "l_linenumber",
  "c"."c_custkey" AS "c_custkey"
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
JOIN "raw_customer" AS "c"
  ON (
    "c"."c_custkey" = "o"."o_custkey"
    AND SEMANTIC_JOIN(
      'Does customer balance {1} cover at least 4% of order total {0}?',
      CAST("o"."o_totalprice" AS TEXT),
      CAST("c"."c_acctbal" AS TEXT)
    )
  )
  AND SEMANTIC_FILTER(
    'Does customer comment {0} mention requests or accounts?',
    CAST("c"."c_comment" AS TEXT)
  )
WHERE
  SEMANTIC_FILTER(
    'Does order comment {0} mention requests or accounts?',
    CAST("o"."o_comment" AS TEXT)
  )
  AND (
    "o"."o_orderdate" >= CAST('1994-01-01' AS DATE)
    AND "o"."o_orderdate" < CAST('1998-01-01' AS DATE)
    AND "o"."o_totalprice" BETWEEN 20000 AND 280000
    AND "o"."o_orderstatus" = 'O'
    AND "o"."o_orderpriority" IN ('1-URGENT', '2-HIGH')
  );
