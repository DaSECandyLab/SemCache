SELECT
  o.o_orderkey,
  o.o_custkey,
  o.o_orderdate,
  o.o_orderpriority,
  o.o_totalprice,
  l.l_partkey,
  l.l_suppkey,
  l.l_shipmode,
  l.l_shipinstruct,
  l.l_quantity,
  l.l_extendedprice,
  l.l_discount,
  l.l_tax,
  l.l_returnflag,
  p.p_name,
  p.p_type,
  p.p_brand,
  s.s_name,
  n.n_name AS supplier_nation,
  c.*
FROM orders AS o
JOIN lineitem AS l
  ON l.l_orderkey = o.o_orderkey
JOIN part AS p
  ON p.p_partkey = l.l_partkey
JOIN partsupp AS ps
  ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
JOIN supplier AS s
  ON s.s_suppkey = ps.ps_suppkey
JOIN nation AS n
  ON n.n_nationkey = s.s_nationkey
JOIN (
  SELECT
    *
  FROM customer
  ORDER BY
    c_custkey ASC
  LIMIT 120
) AS c
  ON c.c_custkey = o.o_custkey
WHERE
  (
    o.o_orderdate >= CAST('1993-01-01' AS DATE)
    AND o.o_orderdate < CAST('1997-01-01' AS DATE)
    AND o.o_orderstatus IN ('O', 'F')
    AND o.o_totalprice > 50000
    AND SEMANTIC_FILTER(
      'You are triaging enterprise orders for operations monitoring. From the priority label, total price, and order comment, does this order look important enough to deserve active follow-up by an operations manager?
Order Priority: {0}
Total Price: {1}
Comment: {2}
Return a single YES or NO, with no other words. If uncertain, return YES.',
      CAST(o.o_orderpriority AS TEXT),
      CAST(o.o_totalprice AS TEXT),
      CAST(o.o_comment AS TEXT)
    )
  )
  AND (
    l.l_shipdate >= CAST('1993-01-01' AS DATE)
    AND l.l_shipdate < CAST('1998-01-01' AS DATE)
    AND l.l_quantity BETWEEN 5 AND 40
    AND l.l_discount BETWEEN 0.01 AND 0.10
    AND l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK')
  )
  AND (
    p.p_size BETWEEN 5 AND 35 AND ps.ps_supplycost BETWEEN 10 AND 1000
  )
  AND (
    SEMANTIC_FILTER(
      'You are identifying high-value B2B customer profiles. From the customer name format, market segment, balance, and account comment, does this record look like a customer worth prioritizing in business account analytics?
Customer Name: {0}
Market Segment: {1}
Account Balance: {2}
Comment: {3}
Return a single YES or NO, with no other words. If uncertain, return NO.',
      CAST(c.c_name AS TEXT),
      CAST(c.c_mktsegment AS TEXT),
      CAST(c.c_acctbal AS TEXT),
      CAST(c.c_comment AS TEXT)
    )
  )
  AND (
    o.o_custkey = c.c_custkey
    AND SEMANTIC_FILTER(
      'You are an operations manager assessing whether a high-value customer is associated with an important order that requires active follow-up. The customer has the following profile:
Customer Name: {0}
Market Segment: {1}
Account Balance: {2}
Comment: {3})',
      CAST(c.c_name AS TEXT),
      CAST(c.c_mktsegment AS TEXT),
      CAST(c.c_acctbal AS TEXT),
      CAST(c.c_comment AS TEXT)
    )
  );
