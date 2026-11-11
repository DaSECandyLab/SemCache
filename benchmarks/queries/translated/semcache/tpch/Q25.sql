SELECT
  p.p_partkey,
  p.p_name,
  p.p_type,
  p.p_brand,
  l.l_orderkey,
  l.l_linenumber,
  l.l_suppkey,
  l.l_quantity,
  l.l_extendedprice,
  l.l_discount,
  l.l_shipmode,
  l.l_shipinstruct,
  o.o_orderdate,
  o.o_orderpriority,
  o.o_totalprice,
  c.*
FROM part AS p
JOIN lineitem AS l
  ON l.l_partkey = p.p_partkey
JOIN orders AS o
  ON o.o_orderkey = l.l_orderkey
CROSS JOIN (
  SELECT
    *
  FROM customer
  ORDER BY
    c_custkey ASC
  LIMIT 100
) AS c
WHERE
  (
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
    l.l_shipdate >= CAST('1995-01-01' AS DATE)
    AND l.l_shipdate < CAST('1997-01-01' AS DATE)
    AND l.l_discount BETWEEN 0.05 AND 0.07
    AND l.l_quantity BETWEEN 18 AND 22
    AND l.l_shipmode IN ('AIR', 'REG AIR')
    AND l.l_shipinstruct = 'DELIVER IN PERSON'
    AND o.o_orderstatus = 'O'
    AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
    AND o.o_totalprice > 100000
  )
  AND (
    SEMANTIC_FILTER(
      'You are tagging accounts for B2B revenue analysis. Based on this customer name pattern, market segment, and balance level, does this profile look more like a business customer likely to place larger operational orders?
Customer Name: {0}
Market Segment: {1}
Account Balance: {2}
Return a single YES or NO, with no other words. If uncertain, return NO.',
      CAST(c.c_name AS TEXT),
      CAST(c.c_mktsegment AS TEXT),
      CAST(c.c_acctbal AS TEXT)
    )
  )
  AND (
    SEMANTIC_FILTER(
      'You are validating whether this customer profile is a business-relevant account for enterprise analysis.
Customer Name: {0}
Market Segment: {1}
Account Balance: {2}
Return a single YES or NO, with no other words. If uncertain, return NO.',
      CAST(c.c_name AS TEXT),
      CAST(c.c_mktsegment AS TEXT),
      CAST(c.c_acctbal AS TEXT)
    )
  );
