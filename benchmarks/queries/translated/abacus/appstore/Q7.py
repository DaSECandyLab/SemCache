"""abacus: appstore/Q7; common numeric v4; explicitly rewritten workload.
Source: original/appstore/sema/Q7.sql
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'abacus'
QUERY_ID = 'appstore/Q7'
SOURCE = 'original/appstore/sema/Q7.sql'

RELATIONAL_SQL = "SELECT\n  ur.rowid AS id\nFROM user_reviews AS ur\nJOIN playstore AS p\n  ON ur.App = p.App\nWHERE\n  p.Category = 'TOOLS'\n  AND SEMANTIC_FILTER(\n    '{0} is a complete sentence and has informative content.',\n    CAST(ur.Translated_Review AS TEXT)\n  )\n  AND SEMANTIC_FILTER('{0} is a positive user review.', CAST(ur.Translated_Review AS TEXT))\n  AND SEMANTIC_FILTER('{0} shows that this app is a useful tool.', CAST(ur.Translated_Review AS TEXT));\n"

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_relational
    return run_relational(SYSTEM, RELATIONAL_SQL, tables, job, semantic=semantic)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
