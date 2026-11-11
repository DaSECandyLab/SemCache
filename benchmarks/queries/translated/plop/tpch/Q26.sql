WITH customer_sample AS MATERIALIZED (
  SELECT
    *
  FROM customer
  WHERE
    c_acctbal > 500 AND c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY')
), a_candidates AS (
  SELECT
    p.p_partkey,
    p.p_name,
    p.p_type,
    p.p_brand
  FROM part AS p
  WHERE
    SEMANTIC(
      'You are helping a retail analytics team pick products for customer demand analysis. From the name, type, and brand, does this item read like a marketable finished good rather than an upstream industrial input?
Part Name: {p.p_name}
Part Type: {p.p_type}
Brand: {p.p_brand}
Return a single YES or NO, with no other words. If uncertain, return YES.'
    )
), s_candidates AS (
  SELECT
    s.s_suppkey,
    s.s_name,
    s.s_acctbal,
    s.s_comment,
    n.n_name AS supplier_nation,
    r.r_name AS supplier_region
  FROM supplier AS s
  JOIN nation AS n
    ON n.n_nationkey = s.s_nationkey
  JOIN region AS r
    ON r.r_regionkey = n.n_regionkey
  WHERE
    s.s_acctbal > 0
    AND SEMANTIC(
      'You are shortlisting suppliers for stable operations. Based on supplier name style, account balance, comment, nation, and region, does this supplier look commercially dependable?
Supplier Name: {s.s_name}
Account Balance: {s.s_acctbal}
Comment: {s.s_comment}
Nation: {n.n_name}
Region: {r.r_name}
Return a single YES or NO, with no other words. If uncertain, return YES.'
    )
)
SELECT
  b.l_orderkey,
  b.l_linenumber,
  b.l_quantity,
  b.l_extendedprice,
  b.l_discount,
  b.l_tax,
  b.l_shipmode,
  b.l_shipinstruct,
  a.p_name,
  a.p_type,
  a.p_brand,
  s.s_name,
  s.supplier_nation,
  s.supplier_region,
  ps.ps_supplycost,
  ps.ps_availqty,
  c.c_custkey AS context_custkey,
  c.c_name,
  c.c_mktsegment,
  c.c_acctbal
FROM (
  SELECT
    l.l_orderkey,
    l.l_linenumber,
    l.l_partkey,
    l.l_suppkey,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    l.l_tax,
    l.l_shipmode,
    l.l_shipinstruct
  FROM lineitem AS l
  JOIN orders AS o
    ON o.o_orderkey = l.l_orderkey
  WHERE
    l.l_shipdate >= CAST('1995-01-01' AS DATE)
    AND l.l_shipdate < CAST('1997-01-01' AS DATE)
    AND l.l_discount BETWEEN 0.05 AND 0.08
    AND l.l_quantity BETWEEN 16 AND 24
    AND l.l_shipmode IN ('AIR', 'REG AIR')
    AND l.l_shipinstruct = 'DELIVER IN PERSON'
    AND o.o_orderstatus = 'O'
    AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
    AND o.o_totalprice > 90000
    AND SEMANTIC(
      'You are triaging enterprise orders for operations monitoring. From priority, total price, and comment, should this order be considered operationally important?
Order Priority: {o.o_orderpriority}
Total Price: {o.o_totalprice}
Order Comment: {o.o_comment}
Return a single YES or NO, with no other words. If uncertain, return YES.'
    )
) AS b
JOIN (
  SELECT
    ps.ps_partkey,
    ps.ps_suppkey,
    ps.ps_supplycost,
    ps.ps_availqty
  FROM partsupp AS ps
  JOIN a_candidates AS a
    ON a.p_partkey = ps.ps_partkey
  JOIN s_candidates AS s
    ON s.s_suppkey = ps.ps_suppkey
  WHERE
    ps.ps_supplycost BETWEEN 80 AND 900 AND ps.ps_availqty >= 50
) AS ps
  ON ps.ps_partkey = b.l_partkey AND ps.ps_suppkey = b.l_suppkey
JOIN a_candidates AS a
  ON a.p_partkey = b.l_partkey
JOIN s_candidates AS s
  ON s.s_suppkey = b.l_suppkey
CROSS JOIN (
  SELECT
    *
  FROM customer_sample AS c
  WHERE
    SEMANTIC(
      'You are tagging accounts for B2B revenue analysis. Based on customer name, market segment, account balance, and profile comment, does this customer look business-like and operationally relevant?
Customer Name: {c.c_name}
Market Segment: {c.c_mktsegment}
Account Balance: {c.c_acctbal}
Comment: {c.c_comment}
Return a single YES or NO, with no other words. If uncertain, return NO.'
    )
) AS c;
