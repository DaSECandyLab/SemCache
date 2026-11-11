"""abacus: tpch/Q16; common numeric v4; explicitly rewritten workload.
Source: original/tpch/plop/Q16.sql
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'abacus'
QUERY_ID = 'tpch/Q16'
SOURCE = 'original/tpch/plop/Q16.sql'

RELATIONAL_SQL = "WITH customer_sample AS MATERIALIZED (\n  SELECT\n    *\n  FROM customer\n  ORDER BY\n    c_custkey ASC\n  LIMIT 100\n)\nSELECT\n  *\nFROM (\n  SELECT\n    a.p_partkey,\n    a.p_name,\n    a.p_type,\n    a.p_brand,\n    b.l_orderkey,\n    b.l_linenumber,\n    b.l_suppkey,\n    b.l_quantity,\n    b.l_extendedprice,\n    b.l_discount,\n    b.l_shipmode,\n    b.l_shipinstruct,\n    b.o_orderdate,\n    b.o_orderpriority,\n    b.o_totalprice\n  FROM (\n    SELECT\n      p.p_partkey,\n      p.p_name,\n      p.p_type,\n      p.p_brand\n    FROM part AS p\n    WHERE\n      SEMANTIC_FILTER(\n        'Task: classify whether this part should be included in customer-facing demand analysis.\nRule: YES only if the part sounds like a finished good that could appear in commercial product demand; NO for raw/industrial-looking components.\nPart Name: {0}\nPart Type: {1}\nBrand: {2}\nReturn YES or NO only. If uncertain, return NO.',\n        CAST(p.p_name AS TEXT),\n        CAST(p.p_type AS TEXT),\n        CAST(p.p_brand AS TEXT)\n      )\n  ) AS a\n  JOIN (\n    SELECT\n      l.l_orderkey,\n      l.l_linenumber,\n      l.l_partkey,\n      l.l_suppkey,\n      l.l_quantity,\n      l.l_extendedprice,\n      l.l_discount,\n      l.l_shipmode,\n      l.l_shipinstruct,\n      o.o_orderdate,\n      o.o_orderpriority,\n      o.o_totalprice\n    FROM lineitem AS l\n    JOIN orders AS o\n      ON o.o_orderkey = l.l_orderkey\n    WHERE\n      l.l_shipdate >= CAST('1995-01-01' AS DATE)\n      AND l.l_shipdate < CAST('1997-01-01' AS DATE)\n      AND l.l_discount BETWEEN 0.05 AND 0.07\n      AND l.l_quantity BETWEEN 18 AND 22\n      AND l.l_shipmode IN ('AIR', 'REG AIR')\n      AND l.l_shipinstruct = 'DELIVER IN PERSON'\n      AND o.o_orderstatus = 'O'\n      AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')\n      AND o.o_totalprice > 100000\n  ) AS b\n    ON b.l_partkey = a.p_partkey\n) AS ab\nJOIN (\n  SELECT\n    *\n  FROM customer_sample AS c\n  WHERE\n    SEMANTIC_FILTER(\n      'Task: decide if this is an enterprise-oriented customer profile for B2B order analysis.\nRule: YES only for profiles that look commercially active (business segment + meaningful balance); otherwise NO.\nCustomer Name: {0}\nMarket Segment: {1}\nAccount Balance: {2}\nReturn YES or NO only. If uncertain, return NO.',\n      CAST(c.c_name AS TEXT),\n      CAST(c.c_mktsegment AS TEXT),\n      CAST(c.c_acctbal AS TEXT)\n    )\n) AS c\n  ON SEMANTIC_FILTER(\n    'You are validating whether this customer profile is a business-relevant account for enterprise analysis.\nCustomer Name: {0}\nMarket Segment: {1}\nAccount Balance: {2}\nReturn a single YES or NO, with no other words. If uncertain, return NO.',\n    CAST(c.c_name AS TEXT),\n    CAST(c.c_mktsegment AS TEXT),\n    CAST(c.c_acctbal AS TEXT)\n  );\n"

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_relational
    return run_relational(SYSTEM, RELATIONAL_SQL, tables, job, semantic=semantic)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
