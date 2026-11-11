"""nirvana: estate/Q9; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q9.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'nirvana'
QUERY_ID = 'estate/Q9'
SOURCE = 'original/estate/nirvana/Q9.py'
AGGREGATE = {'function': 'AVG', 'column': 'Price'}

def build(tables):
    import nirvana as nv
    df = nv.DataFrame(tables['nirvana_estate'])
    df.semantic_map(user_instruction='Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.', input_columns=['Details'], output_columns=['Price'])
    df.semantic_filter(user_instruction='Whether the estate has more than 3 bedrooms', input_columns=['Title'])
    df.semantic_filter(user_instruction='Whether the estate has less than 6 bedrooms.', input_columns=['Title'])
    return df

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_nirvana
    return run_nirvana(build(tables), job, aggregate=AGGREGATE)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
