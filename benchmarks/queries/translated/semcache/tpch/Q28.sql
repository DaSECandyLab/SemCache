SELECT
  *
FROM (
  SELECT
    o.o_orderkey,
    o.o_orderpriority,
    o.o_totalprice,
    o.o_comment,
    o.o_orderdate,
    o.o_custkey,
    l.l_partkey,
    l.l_suppkey,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    l.l_shipmode
  FROM orders AS o
  JOIN lineitem AS l
    ON l.l_orderkey = o.o_orderkey
    AND (
      l.l_shipdate >= CAST('1995-01-01' AS DATE)
      AND l.l_shipdate < CAST('1996-07-01' AS DATE)
      AND l.l_quantity BETWEEN 18 AND 26
      AND l.l_discount BETWEEN 0.04 AND 0.07
      AND l.l_shipmode IN ('AIR', 'REG AIR')
    )
  WHERE
    SEMANTIC_FILTER(
      'Does this order look operationally significant based on priority label, total price, and comment content?
Order Priority: {0}
Total Price: {1}
Comment: {2}
Return a single YES or NO, with no other words. If uncertain, return YES.',
      CAST(o.o_orderpriority AS TEXT),
      CAST(o.o_totalprice AS TEXT),
      CAST(o.o_comment AS TEXT)
    )
    AND (
      o.o_orderdate >= CAST('1995-01-01' AS DATE)
      AND o.o_orderdate < CAST('1996-07-01' AS DATE)
      AND o.o_totalprice BETWEEN 100000 AND 220000
      AND o.o_orderstatus = 'O'
      AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
    )
) AS j
JOIN (
  SELECT
    c.c_custkey,
    c.c_name,
    c.c_address,
    c.c_nationkey,
    c.c_phone,
    c.c_acctbal,
    c.c_mktsegment,
    c.c_comment
  FROM customer AS c
  JOIN (
    SELECT DISTINCT
      o_custkey
    FROM orders AS o
    JOIN lineitem AS l
      ON l.l_orderkey = o.o_orderkey
      AND (
        l.l_shipdate >= CAST('1995-01-01' AS DATE)
        AND l.l_shipdate < CAST('1996-07-01' AS DATE)
        AND l.l_quantity BETWEEN 18 AND 26
        AND l.l_discount BETWEEN 0.04 AND 0.07
        AND l.l_shipmode IN ('AIR', 'REG AIR')
      )
    WHERE
      SEMANTIC_FILTER(
        'Does this order look operationally significant based on priority label, total price, and comment content?
Order Priority: {0}
Total Price: {1}
Comment: {2}
Return a single YES or NO, with no other words. If uncertain, return YES.',
        CAST(o.o_orderpriority AS TEXT),
        CAST(o.o_totalprice AS TEXT),
        CAST(o.o_comment AS TEXT)
      )
      AND (
        o.o_orderdate >= CAST('1995-01-01' AS DATE)
        AND o.o_orderdate < CAST('1996-07-01' AS DATE)
        AND o.o_totalprice BETWEEN 100000 AND 220000
        AND o.o_orderstatus = 'O'
        AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
      )
    ORDER BY
      o_custkey ASC
    LIMIT 60
  ) AS k
    ON k.o_custkey = c.c_custkey
  WHERE
    SEMANTIC_FILTER(
      'Does this customer look operationally strategic based on name style, market segment, and account balance?
Customer Name: {0}
Market Segment: {1}
Account Balance: {2}
Return a single YES or NO, with no other words. If uncertain, return YES.',
      CAST(c.c_name AS TEXT),
      CAST(c.c_mktsegment AS TEXT),
      CAST(c.c_acctbal AS TEXT)
    )
) AS cc
  ON cc.c_custkey = j.o_custkey
  AND SEMANTIC_JOIN(
    'Could this customer account be strategically important for order fulfillment?
Order Priority: {0}
Order Total Price: {1}
Lineitem Ship Mode: {2}
Customer Name: {3}
Market Segment: {4}
Account Balance: {5}
Return a single YES or NO, with no other words. If uncertain, return YES.',
    CAST(j.o_orderpriority AS TEXT),
    CAST(j.o_totalprice AS TEXT),
    CAST(j.l_shipmode AS TEXT),
    CAST(cc.c_name AS TEXT),
    CAST(cc.c_mktsegment AS TEXT),
    CAST(cc.c_acctbal AS TEXT)
  );
