WITH customer_sample AS MATERIALIZED (
  SELECT
    *
  FROM customer
  ORDER BY
    c_custkey ASC
  LIMIT 100
)
SELECT
  *
FROM (
  SELECT
    a.p_partkey,
    a.p_name,
    a.p_type,
    a.p_brand,
    b.l_orderkey,
    b.l_linenumber,
    b.l_suppkey,
    b.l_quantity,
    b.l_extendedprice,
    b.l_discount,
    b.l_shipmode,
    b.l_shipinstruct,
    b.o_orderdate,
    b.o_orderpriority,
    b.o_totalprice
  FROM (
    SELECT
      p.p_partkey,
      p.p_name,
      p.p_type,
      p.p_brand
    FROM part AS p
    WHERE
      s'You are helping a retail analytics team pick products for customer demand analysis. From the name, type, and brand, does this item read like a marketable finished good rather than an upstream industrial input?
Part Name: {p.p_name}
Part Type: {p.p_type}
Brand: {p.p_brand}
Return a single YES or NO, with no other words. If uncertain, return YES.'
  ) AS a
  JOIN (
    SELECT
      l.l_orderkey,
      l.l_linenumber,
      l.l_partkey,
      l.l_suppkey,
      l.l_quantity,
      l.l_extendedprice,
      l.l_discount,
      l.l_shipmode,
      l.l_shipinstruct,
      o.o_orderdate,
      o.o_orderpriority,
      o.o_totalprice
    FROM lineitem AS l
    JOIN orders AS o
      ON o.o_orderkey = l.l_orderkey
    WHERE
      l.l_shipdate >= CAST('1995-01-01' AS DATE)
      AND l.l_shipdate < CAST('1997-01-01' AS DATE)
      AND l.l_discount BETWEEN 0.05 AND 0.07
      AND l.l_quantity BETWEEN 18 AND 22
      AND l.l_shipmode IN ('AIR', 'REG AIR')
      AND l.l_shipinstruct = 'DELIVER IN PERSON'
      AND o.o_orderstatus = 'O'
      AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
      AND o.o_totalprice > 100000
  ) AS b
    ON b.l_partkey = a.p_partkey
) AS ab
JOIN (
  SELECT
    *
  FROM customer_sample AS c
  WHERE
    s'You are tagging accounts for B2B revenue analysis. Based on this customer name pattern, market segment, and balance level, does this profile look more like a business customer likely to place larger operational orders?
Customer Name: {c.c_name}
Market Segment: {c.c_mktsegment}
Account Balance: {c.c_acctbal}
Return a single YES or NO, with no other words. If uncertain, return NO.'
) AS c
  ON s'You are validating whether this customer profile is a business-relevant account for enterprise analysis.
Customer Name: {c.c_name}
Market Segment: {c.c_mktsegment}
Account Balance: {c.c_acctbal}
Return a single YES or NO, with no other words. If uncertain, return NO.';
