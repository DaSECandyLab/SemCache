"""abacus: estate/Q10; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q10.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'abacus'
QUERY_ID = 'estate/Q10'
SOURCE = 'original/estate/nirvana/Q10.py'
AGGREGATE = {'function': 'MIN', 'column': 'Price'}

def build(tables):
    from experiments.validation.native_api import load_adapter
    ds = load_adapter("abacus").source_dataset(tables['nirvana_estate'], QUERY_ID)
    ds = ds.sem_map(cols=[{'name': 'Price', 'type': float | None, 'desc': 'Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.'}], desc='Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.', depends_on=['Details'])
    ds = ds.sem_filter('Whether the estate has more than 3 bedrooms.', depends_on=['Title'])
    ds = ds.sem_filter('Whether the estate has less than 6 bedrooms.', depends_on=['Title'])
    ds = ds.sem_filter('Whether the estate is a detached duplex.', depends_on=['Title'])
    return ds

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_abacus
    return run_abacus(build(tables), job, aggregate=AGGREGATE)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
