WITH joined AS (
  SELECT
    pf.p_partkey,
    pf.p_name,
    pf.p_type,
    pf.p_brand,
    sb.ps_suppkey,
    sb.s_name,
    sb.s_acctbal,
    sb.s_comment,
    sb.nation_name,
    sb.region_name,
    sb.ps_availqty,
    sb.ps_supplycost,
    ol.l_orderkey,
    ol.l_quantity,
    ol.l_extendedprice,
    ol.o_orderdate,
    ol.o_orderpriority,
    ol.o_totalprice
  FROM (
    SELECT
      p.p_partkey,
      p.p_name,
      p.p_type,
      p.p_brand,
      p.p_size
    FROM part AS p
    WHERE
      s'You are screening parts for strategic sourcing analysis. Looking at part name, type, brand, and size, does this look like a differentiated component where supplier quality and pricing strategy would matter?
Part Name: {p.p_name}
Part Type: {p.p_type}
Brand: {p.p_brand}
Size: {p.p_size}
Return a single YES or NO, with no other words. If uncertain, return YES.'
  ) AS pf
  JOIN (
    SELECT
      ps.ps_partkey,
      ps.ps_suppkey,
      ps.ps_availqty,
      ps.ps_supplycost,
      s.s_name,
      s.s_acctbal,
      s.s_comment,
      s.s_nationkey,
      n.n_name AS nation_name,
      r.r_name AS region_name
    FROM partsupp AS ps
    JOIN supplier AS s
      ON s.s_suppkey = ps.ps_suppkey
    JOIN nation AS n
      ON n.n_nationkey = s.s_nationkey
    JOIN region AS r
      ON r.r_regionkey = n.n_regionkey
    WHERE
      ps.ps_availqty BETWEEN 10 AND 600
      AND ps.ps_supplycost BETWEEN 20 AND 900
      AND r.r_name IN ('EUROPE', 'ASIA', 'AMERICA')
  ) AS sb
    ON sb.ps_partkey = pf.p_partkey
  JOIN (
    SELECT
      l.l_partkey,
      l.l_suppkey,
      l.l_orderkey,
      l.l_quantity,
      l.l_extendedprice,
      l.l_discount,
      o.o_orderdate,
      o.o_orderpriority,
      o.o_totalprice
    FROM lineitem AS l
    JOIN orders AS o
      ON o.o_orderkey = l.l_orderkey
    WHERE
      o.o_orderdate >= CAST('1994-01-01' AS DATE)
      AND o.o_orderdate < CAST('1997-01-01' AS DATE)
      AND l.l_quantity BETWEEN 8 AND 35
      AND l.l_discount BETWEEN 0.00 AND 0.08
  ) AS ol
    ON ol.l_partkey = sb.ps_partkey AND ol.l_suppkey = sb.ps_suppkey
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
      ps_suppkey
    FROM joined
    ORDER BY
      ps_suppkey ASC
    LIMIT 80
  ) AS k
    ON k.ps_suppkey = s.s_suppkey
  WHERE
    s'You are shortlisting suppliers for long-term procurement partnerships. Based on supplier identity style, account balance, and profile comment tone, does this supplier appear commercially dependable enough for preferred-vendor consideration?
Supplier Name: {s.s_name}
Account Balance: {s.s_acctbal}
Comment: {s.s_comment}
Return a single YES or NO, with no other words. If uncertain, return YES.'
) AS sc
  ON sc.s_suppkey = j.ps_suppkey
  AND s'You are validating whether this supplier profile is suitable for preferred-vendor sourcing.
Part Type: {j.p_type}
Region: {j.region_name}
Supplier Name: {sc.s_name}
Account Balance: {sc.s_acctbal}
Comment: {sc.s_comment}
Return a single YES or NO, with no other words. If uncertain, return YES.';
