SELECT
  "b"."l_orderkey" AS "l_orderkey",
  "b"."l_linenumber" AS "l_linenumber",
  "raw_customer"."c_custkey" AS "c_custkey"
FROM "raw_part" AS "p"
JOIN (
  SELECT
    "l"."l_orderkey" AS "l_orderkey",
    "l"."l_linenumber" AS "l_linenumber",
    "l"."l_partkey" AS "l_partkey",
    "l"."l_suppkey" AS "l_suppkey",
    "l"."l_quantity" AS "l_quantity",
    "l"."l_extendedprice" AS "l_extendedprice",
    "l"."l_discount" AS "l_discount",
    "l"."l_shipmode" AS "l_shipmode",
    "l"."l_shipinstruct" AS "l_shipinstruct",
    "o"."o_orderdate" AS "o_orderdate",
    "o"."o_orderpriority" AS "o_orderpriority",
    "o"."o_totalprice" AS "o_totalprice",
    "o"."o_custkey" AS "o_custkey"
  FROM "raw_lineitem" AS "l"
  JOIN "raw_orders" AS "o"
    ON "o"."o_orderkey" = "l"."l_orderkey"
  WHERE
    "l"."l_shipdate" >= CAST('1993-01-01' AS DATE)
    AND "l"."l_shipdate" < CAST('1997-01-01' AS DATE)
    AND "l"."l_discount" BETWEEN 0.0 AND 0.1
    AND "l"."l_quantity" BETWEEN 5 AND 45
    AND "l"."l_shipmode" IN ('AIR', 'REG AIR', 'TRUCK', 'SHIP')
    AND "l"."l_shipinstruct" = 'DELIVER IN PERSON'
    AND "o"."o_orderstatus" = 'O'
    AND "o"."o_orderpriority" IN ('1-URGENT', '2-HIGH')
    AND "o"."o_totalprice" > 31600
) AS "b"
  ON "b"."l_partkey" = "p"."p_partkey"
JOIN "raw_customer" AS "raw_customer"
  ON (
    "raw_customer"."c_custkey" = "b"."o_custkey"
    AND SEMANTIC_FILTER(
      'Is customer balance {0} between 2000 and 6500?',
      CAST("raw_customer"."c_acctbal" AS TEXT)
    )
  )
  AND SEMANTIC_FILTER(
    'Is customer segment {0} BUILDING, MACHINERY or AUTOMOBILE?',
    CAST("raw_customer"."c_mktsegment" AS TEXT)
  )
WHERE
  SEMANTIC_FILTER('Does part type {0} contain COPPER, BRASS or TIN?', CAST("p"."p_type" AS TEXT));
