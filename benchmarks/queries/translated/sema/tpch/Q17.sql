WITH customer_sample AS MATERIALIZED (
  SELECT
    *
  FROM customer
  ORDER BY
    c_custkey ASC
  LIMIT 120
)
SELECT
  *
FROM (
  SELECT
    of.o_orderkey,
    of.o_custkey,
    of.o_orderdate,
    of.o_orderpriority,
    of.o_totalprice,
    lf.l_partkey,
    lf.l_suppkey,
    lf.l_shipmode,
    lf.l_shipinstruct,
    lf.l_quantity,
    lf.l_extendedprice,
    lf.l_discount,
    lf.l_tax,
    lf.l_returnflag,
    ps.p_name,
    ps.p_type,
    ps.p_brand,
    ps.s_name,
    ps.supplier_nation
  FROM (
    SELECT
      o.o_orderkey,
      o.o_custkey,
      o.o_orderdate,
      o.o_orderpriority,
      o.o_totalprice,
      o.o_comment
    FROM orders AS o
    WHERE
      o.o_orderdate >= CAST('1993-01-01' AS DATE)
      AND o.o_orderdate < CAST('1997-01-01' AS DATE)
      AND o.o_orderstatus IN ('O', 'F')
      AND o.o_totalprice > 50000
      AND s'You are triaging enterprise orders for operations monitoring. From the priority label, total price, and order comment, does this order look important enough to deserve active follow-up by an operations manager?
Order Priority: {o.o_orderpriority}
Total Price: {o.o_totalprice}
Comment: {o.o_comment}
Return a single YES or NO, with no other words. If uncertain, return YES.'
  ) AS of
  JOIN (
    SELECT
      l.l_orderkey,
      l.l_partkey,
      l.l_suppkey,
      l.l_shipmode,
      l.l_shipinstruct,
      l.l_quantity,
      l.l_extendedprice,
      l.l_discount,
      l.l_tax,
      l.l_returnflag
    FROM lineitem AS l
    WHERE
      l.l_shipdate >= CAST('1993-01-01' AS DATE)
      AND l.l_shipdate < CAST('1998-01-01' AS DATE)
      AND l.l_quantity BETWEEN 5 AND 40
      AND l.l_discount BETWEEN 0.01 AND 0.10
      AND l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK')
  ) AS lf
    ON lf.l_orderkey = of.o_orderkey
  JOIN (
    SELECT
      p.p_partkey,
      p.p_name,
      p.p_type,
      p.p_brand,
      s.s_suppkey,
      s.s_name,
      n.n_name AS supplier_nation
    FROM part AS p
    JOIN partsupp AS ps
      ON ps.ps_partkey = p.p_partkey
    JOIN supplier AS s
      ON s.s_suppkey = ps.ps_suppkey
    JOIN nation AS n
      ON n.n_nationkey = s.s_nationkey
    WHERE
      p.p_size BETWEEN 5 AND 35 AND ps.ps_supplycost BETWEEN 10 AND 1000
  ) AS ps
    ON ps.p_partkey = lf.l_partkey AND ps.s_suppkey = lf.l_suppkey
) AS j
JOIN (
  SELECT
    *
  FROM customer_sample AS c
  WHERE
    s'You are identifying high-value B2B customer profiles. From the customer name format, market segment, balance, and account comment, does this record look like a customer worth prioritizing in business account analytics?
Customer Name: {c.c_name}
Market Segment: {c.c_mktsegment}
Account Balance: {c.c_acctbal}
Comment: {c.c_comment}
Return a single YES or NO, with no other words. If uncertain, return NO.'
) AS c
  ON j.o_custkey = c.c_custkey
  AND s'You are an operations manager assessing whether a high-value customer is associated with an important order that requires active follow-up. The customer has the following profile:
Customer Name: {c.c_name}
Market Segment: {c.c_mktsegment}
Account Balance: {c.c_acctbal}
Comment: {c.c_comment})';
