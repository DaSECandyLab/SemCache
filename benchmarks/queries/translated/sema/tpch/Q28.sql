WITH joined AS (
  SELECT
    of.o_orderkey,
    of.o_orderpriority,
    of.o_totalprice,
    of.o_comment,
    of.o_orderdate,
    of.o_custkey,
    lf.l_partkey,
    lf.l_suppkey,
    lf.l_quantity,
    lf.l_extendedprice,
    lf.l_discount,
    lf.l_shipmode
  FROM (
    SELECT
      ob.o_orderkey,
      ob.o_orderpriority,
      ob.o_totalprice,
      ob.o_comment,
      ob.o_orderdate,
      ob.o_custkey
    FROM (
      SELECT
        o.o_orderkey,
        o.o_orderpriority,
        o.o_totalprice,
        o.o_comment,
        o.o_orderdate,
        o.o_custkey
      FROM orders AS o
      WHERE
        o.o_orderdate >= CAST('1995-01-01' AS DATE)
        AND o.o_orderdate < CAST('1996-07-01' AS DATE)
        AND o.o_totalprice BETWEEN 100000 AND 220000
        AND o.o_orderstatus = 'O'
        AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
    ) AS ob
    WHERE
      s'Does this order look operationally significant based on priority label, total price, and comment content?
Order Priority: {ob.o_orderpriority}
Total Price: {ob.o_totalprice}
Comment: {ob.o_comment}
Return a single YES or NO, with no other words. If uncertain, return YES.'
  ) AS of
  JOIN (
    SELECT
      l.l_orderkey,
      l.l_partkey,
      l.l_suppkey,
      l.l_quantity,
      l.l_extendedprice,
      l.l_discount,
      l.l_shipmode,
      l.l_shipdate
    FROM lineitem AS l
    WHERE
      l.l_shipdate >= CAST('1995-01-01' AS DATE)
      AND l.l_shipdate < CAST('1996-07-01' AS DATE)
      AND l.l_quantity BETWEEN 18 AND 26
      AND l.l_discount BETWEEN 0.04 AND 0.07
      AND l.l_shipmode IN ('AIR', 'REG AIR')
  ) AS lf
    ON lf.l_orderkey = of.o_orderkey
)
SELECT
  *
FROM joined AS j
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
    FROM joined
    ORDER BY
      o_custkey ASC
    LIMIT 60
  ) AS k
    ON k.o_custkey = c.c_custkey
  WHERE
    s'Does this customer look operationally strategic based on name style, market segment, and account balance?
Customer Name: {c.c_name}
Market Segment: {c.c_mktsegment}
Account Balance: {c.c_acctbal}
Return a single YES or NO, with no other words. If uncertain, return YES.'
) AS cc
  ON cc.c_custkey = j.o_custkey
  AND s'Could this customer account be strategically important for order fulfillment?
Order Priority: {j.o_orderpriority}
Order Total Price: {j.o_totalprice}
Lineitem Ship Mode: {j.l_shipmode}
Customer Name: {cc.c_name}
Market Segment: {cc.c_mktsegment}
Account Balance: {cc.c_acctbal}
Return a single YES or NO, with no other words. If uncertain, return YES.';
