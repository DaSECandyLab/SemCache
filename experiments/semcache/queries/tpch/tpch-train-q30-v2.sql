SELECT
  "olc"."l_orderkey" AS "l_orderkey",
  "olc"."l_linenumber" AS "l_linenumber",
  "s"."s_suppkey" AS "s_suppkey"
FROM "raw_part" AS "p"
JOIN (
  SELECT
    "l"."l_orderkey" AS "l_orderkey",
    "l"."l_partkey" AS "l_partkey",
    "l"."l_suppkey" AS "l_suppkey",
    "l"."l_quantity" AS "l_quantity",
    "l"."l_extendedprice" AS "l_extendedprice",
    "l"."l_discount" AS "l_discount",
    "l"."l_shipmode" AS "l_shipmode",
    "o"."o_orderdate" AS "o_orderdate",
    "o"."o_orderpriority" AS "o_orderpriority",
    "o"."o_totalprice" AS "o_totalprice",
    "c"."c_custkey" AS "c_custkey",
    "c"."c_name" AS "c_name",
    "c"."c_mktsegment" AS "c_mktsegment",
    "c"."c_acctbal" AS "c_acctbal",
    "l"."l_linenumber" AS "l_linenumber"
  FROM "raw_lineitem" AS "l"
  JOIN "raw_orders" AS "o"
    ON "o"."o_orderkey" = "l"."l_orderkey"
  JOIN "raw_customer" AS "c"
    ON "c"."c_custkey" = "o"."o_custkey"
  WHERE
    "l"."l_shipmode" IN ('AIR', 'REG AIR', 'TRUCK', 'SHIP')
    AND "l"."l_quantity" BETWEEN 6 AND 46
    AND "l"."l_discount" BETWEEN 0.0 AND 0.1
    AND "o"."o_orderdate" >= CAST('1994-01-01' AS DATE)
    AND "o"."o_orderdate" < CAST('1998-01-01' AS DATE)
    AND "o"."o_totalprice" BETWEEN 20000 AND 280000
    AND "c"."c_mktsegment" IN ('FURNITURE', 'HOUSEHOLD', 'MACHINERY', 'BUILDING')
) AS "olc"
  ON "olc"."l_partkey" = "p"."p_partkey"
JOIN "raw_supplier" AS "s"
  ON (
    "s"."s_suppkey" = "olc"."l_suppkey"
    AND SEMANTIC_JOIN(
      'Is supplier balance {1} at least required line reserve {0}?',
      CAST((
        0.15 * "olc"."l_extendedprice"
      ) AS TEXT),
      CAST("s"."s_acctbal" AS TEXT)
    )
  )
  AND SEMANTIC_FILTER(
    'Does supplier comment {0} contain unusual, special or pending?',
    CAST("s"."s_comment" AS TEXT)
  )
WHERE
  SEMANTIC_FILTER('Is packaging {0} a box or case?', CAST("p"."p_container" AS TEXT))
  AND (
    "p"."p_size" BETWEEN 3 AND 47
    AND "p"."p_container" IN (
      'SM BOX',
      'MED BOX',
      'LG BOX',
      'SM CASE',
      'MED CASE',
      'LG CASE',
      'SM BAG',
      'MED BAG'
    )
  );
