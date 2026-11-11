SELECT
  l.l_orderkey,
  l.l_linenumber,
  l.l_quantity,
  l.l_extendedprice,
  l.l_discount,
  l.l_tax,
  l.l_shipmode,
  l.l_shipinstruct,
  p.p_name,
  p.p_type,
  p.p_brand,
  s.s_name,
  n.n_name AS supplier_nation,
  r.r_name AS supplier_region,
  ps.ps_supplycost,
  ps.ps_availqty,
  c.c_custkey AS context_custkey,
  c.c_name,
  c.c_mktsegment,
  c.c_acctbal
FROM lineitem AS l
JOIN orders AS o
  ON o.o_orderkey = l.l_orderkey
JOIN part AS p
  ON p.p_partkey = l.l_partkey
JOIN partsupp AS ps
  ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
JOIN supplier AS s
  ON s.s_suppkey = l.l_suppkey
JOIN nation AS n
  ON n.n_nationkey = s.s_nationkey
JOIN region AS r
  ON r.r_regionkey = n.n_regionkey
CROSS JOIN customer AS c
WHERE
  (
    c_acctbal > 500 AND c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY')
  )
  AND (
    SEMANTIC_FILTER(
      'You are helping a retail analytics team pick products for customer demand analysis. From the name, type, and brand, does this item read like a marketable finished good rather than an upstream industrial input?
Part Name: {0}
Part Type: {1}
Brand: {2}
Return a single YES or NO, with no other words. If uncertain, return YES.',
      CAST(p.p_name AS TEXT),
      CAST(p.p_type AS TEXT),
      CAST(p.p_brand AS TEXT)
    )
  )
  AND (
    s.s_acctbal > 0
    AND SEMANTIC_FILTER(
      'You are shortlisting suppliers for stable operations. Based on supplier name style, account balance, comment, nation, and region, does this supplier look commercially dependable?
Supplier Name: {0}
Account Balance: {1}
Comment: {2}
Nation: {3}
Region: {4}
Return a single YES or NO, with no other words. If uncertain, return YES.',
      CAST(s.s_name AS TEXT),
      CAST(s.s_acctbal AS TEXT),
      CAST(s.s_comment AS TEXT),
      CAST(n.n_name AS TEXT),
      CAST(r.r_name AS TEXT)
    )
  )
  AND (
    l.l_shipdate >= CAST('1995-01-01' AS DATE)
    AND l.l_shipdate < CAST('1997-01-01' AS DATE)
    AND l.l_discount BETWEEN 0.05 AND 0.08
    AND l.l_quantity BETWEEN 16 AND 24
    AND l.l_shipmode IN ('AIR', 'REG AIR')
    AND l.l_shipinstruct = 'DELIVER IN PERSON'
    AND o.o_orderstatus = 'O'
    AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
    AND o.o_totalprice > 90000
    AND SEMANTIC_FILTER(
      'You are triaging enterprise orders for operations monitoring. From priority, total price, and comment, should this order be considered operationally important?
Order Priority: {0}
Total Price: {1}
Order Comment: {2}
Return a single YES or NO, with no other words. If uncertain, return YES.',
      CAST(o.o_orderpriority AS TEXT),
      CAST(o.o_totalprice AS TEXT),
      CAST(o.o_comment AS TEXT)
    )
  )
  AND (
    ps.ps_supplycost BETWEEN 80 AND 900 AND ps.ps_availqty >= 50
  )
  AND (
    SEMANTIC_FILTER(
      'You are tagging accounts for B2B revenue analysis. Based on customer name, market segment, account balance, and profile comment, does this customer look business-like and operationally relevant?
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
  );
