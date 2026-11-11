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
    "l"."l_shipdate" >= CAST('1995-01-01' AS DATE)
    AND "l"."l_shipdate" < CAST('1999-01-01' AS DATE)
    AND "l"."l_discount" BETWEEN 0.0 AND 0.1
    AND "l"."l_quantity" BETWEEN 7 AND 47
    AND "l"."l_shipmode" IN ('AIR', 'REG AIR', 'TRUCK', 'SHIP')
    AND "l"."l_shipinstruct" = 'DELIVER IN PERSON'
    AND "o"."o_orderstatus" = 'O'
    AND "o"."o_orderpriority" IN ('1-URGENT', '2-HIGH')
    AND "o"."o_totalprice" > 51600
) AS "b"
  ON "b"."l_partkey" = "p"."p_partkey"
JOIN "raw_customer" AS "raw_customer"
  ON (
    "raw_customer"."c_custkey" = "b"."o_custkey"
    AND SEMANTIC_FILTER(
      'Does customer comment {0} mention requests, accounts, unusual or special cases?',
      CAST("raw_customer"."c_comment" AS TEXT)
    )
  )
  AND SEMANTIC_FILTER(
    'Is customer balance {0} between 0 and 8000?',
    CAST("raw_customer"."c_acctbal" AS TEXT)
  )
WHERE
  SEMANTIC_FILTER(
    'Does part name {0} include red, orange, rose, coral or salmon?',
    CAST("p"."p_name" AS TEXT)
  )
  AND SEMANTIC_FILTER('Is part size {0} between 5 and 45?', CAST("p"."p_size" AS TEXT));
