"""nirvana: estate/Q11; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q11.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'nirvana'
QUERY_ID = 'estate/Q11'
SOURCE = 'original/estate/nirvana/Q11.py'
AGGREGATE = {'function': 'MIN', 'column': 'Price'}

def build(tables):
    import nirvana as nv
    df = nv.DataFrame(tables['nirvana_estate'])
    df.semantic_map(user_instruction='Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.', input_columns=['Details'], output_columns=['Price'])
    df.semantic_map(user_instruction='Extract the amenities from the estate details.', input_columns=['Details'], output_columns=['Amenities'])
    df.semantic_filter(user_instruction='Is there a swimming pool in the estate.', input_columns=['Amenities'])
    return df

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_nirvana
    return run_nirvana(build(tables), job, aggregate=AGGREGATE)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
