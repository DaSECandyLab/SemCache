SELECT
  "o"."o_orderkey" AS "o_orderkey",
  "l"."l_linenumber" AS "l_linenumber",
  "c"."c_custkey" AS "c_custkey"
FROM "raw_orders" AS "o"
JOIN "raw_lineitem" AS "l"
  ON "l"."l_orderkey" = "o"."o_orderkey"
  AND (
    "l"."l_shipdate" >= CAST('1993-01-01' AS DATE)
    AND "l"."l_shipdate" < CAST('1997-01-01' AS DATE)
    AND "l"."l_quantity" BETWEEN 5 AND 45
    AND "l"."l_discount" BETWEEN 0.0 AND 0.1
    AND "l"."l_shipmode" IN ('AIR', 'REG AIR', 'TRUCK', 'SHIP')
  )
JOIN "raw_customer" AS "c"
  ON (
    "c"."c_custkey" = "o"."o_custkey"
    AND SEMANTIC_JOIN(
      'Does customer balance {1} cover at least 3% of order total {0}?',
      CAST("o"."o_totalprice" AS TEXT),
      CAST("c"."c_acctbal" AS TEXT)
    )
  )
  AND SEMANTIC_FILTER('Is customer segment {0} BUILDING or MACHINERY?', CAST("c"."c_mktsegment" AS TEXT))
WHERE
  SEMANTIC_FILTER('Is order total {0} above 190000?', CAST("o"."o_totalprice" AS TEXT))
  AND (
    "o"."o_orderdate" >= CAST('1993-01-01' AS DATE)
    AND "o"."o_orderdate" < CAST('1997-01-01' AS DATE)
    AND "o"."o_totalprice" BETWEEN 20000 AND 280000
    AND "o"."o_orderstatus" = 'O'
    AND "o"."o_orderpriority" IN ('1-URGENT', '2-HIGH')
  );
