WITH orders_lineitem_customer AS (
  SELECT
    l.l_orderkey,
    l.l_partkey,
    l.l_suppkey,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    l.l_shipmode,
    o.o_orderdate,
    o.o_orderpriority,
    o.o_totalprice,
    c.c_custkey,
    c.c_name,
    c.c_mktsegment,
    c.c_acctbal
  FROM lineitem AS l
  JOIN orders AS o
    ON o.o_orderkey = l.l_orderkey
  JOIN customer AS c
    ON c.c_custkey = o.o_custkey
  WHERE
    l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK')
    AND l.l_quantity BETWEEN 12 AND 30
    AND l.l_discount BETWEEN 0.04 AND 0.07
    AND o.o_orderdate >= CAST('1996-01-01' AS DATE)
    AND o.o_orderdate < CAST('1998-01-01' AS DATE)
    AND o.o_totalprice BETWEEN 60000 AND 170000
    AND c.c_mktsegment IN ('FURNITURE', 'HOUSEHOLD', 'MACHINERY', 'BUILDING')
), joined AS (
  SELECT
    pf.p_partkey,
    pf.p_name,
    pf.p_type,
    pf.p_container,
    olc.l_orderkey,
    olc.l_suppkey,
    olc.l_quantity,
    olc.l_extendedprice,
    olc.l_discount,
    olc.l_shipmode,
    olc.o_orderdate,
    olc.o_orderpriority,
    olc.o_totalprice,
    olc.c_custkey,
    olc.c_name,
    olc.c_mktsegment,
    olc.c_acctbal
  FROM (
    SELECT
      pb.p_partkey,
      pb.p_name,
      pb.p_type,
      pb.p_container,
      pb.p_size
    FROM (
      SELECT
        p.p_partkey,
        p.p_name,
        p.p_type,
        p.p_container,
        p.p_size
      FROM part AS p
      JOIN (
        SELECT DISTINCT
          l_partkey
        FROM orders_lineitem_customer
        ORDER BY
          l_partkey ASC
        LIMIT 200
      ) AS k
        ON k.l_partkey = p.p_partkey
      WHERE
        p.p_size BETWEEN 5 AND 28
        AND p.p_container IN ('SM BOX', 'MED BOX', 'LG BOX', 'SM CASE')
    ) AS pb
    WHERE
      s'Does this part look suitable for customer-facing distribution based on name, type, and packaging?
Part Name: {pb.p_name}
Part Type: {pb.p_type}
Container: {pb.p_container}
Return a single YES or NO, with no other words. If uncertain, return YES.'
  ) AS pf
  JOIN orders_lineitem_customer AS olc
    ON olc.l_partkey = pf.p_partkey
)
SELECT
  *
FROM joined AS j
JOIN (
  SELECT
    s.s_suppkey,
    s.s_name,
    s.s_address,
    s.s_nationkey,
    s.s_phone,
    s.s_acctbal,
    s.s_comment
  FROM supplier AS s
  JOIN (
    SELECT DISTINCT
      l_suppkey
    FROM joined
    ORDER BY
      l_suppkey ASC
    LIMIT 40
  ) AS k
    ON k.l_suppkey = s.s_suppkey
  WHERE
    s'Does this supplier appear qualified for consumer-facing distribution partnerships?
Supplier Name: {s.s_name}
Account Balance: {s.s_acctbal}
Comment: {s.s_comment}
Return a single YES or NO, with no other words. If uncertain, return YES.'
) AS sc
  ON sc.s_suppkey = j.l_suppkey
  AND s'Would this supplier be suitable for retail distribution of this product?
Part Type: {j.p_type}
Ship Mode: {j.l_shipmode}
Customer Segment: {j.c_mktsegment}
Supplier Name: {sc.s_name}
Account Balance: {sc.s_acctbal}
Comment: {sc.s_comment}
Return a single YES or NO, with no other words. If uncertain, return YES.';
