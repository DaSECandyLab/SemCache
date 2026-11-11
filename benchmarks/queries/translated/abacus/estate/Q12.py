"""abacus: estate/Q12; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q12.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'abacus'
QUERY_ID = 'estate/Q12'
SOURCE = 'original/estate/nirvana/Q12.py'
AGGREGATE = {'function': 'AVG', 'column': 'Price'}

def build(tables):
    from experiments.validation.native_api import load_adapter
    ds = load_adapter("abacus").source_dataset(tables['nirvana_estate'], QUERY_ID)
    ds = ds.sem_map(cols=[{'name': 'Price', 'type': float | None, 'desc': 'Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.'}], desc='Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.', depends_on=['Details'])
    ds = ds.sem_map(cols=[{'name': 'Amenities', 'type': str | None, 'desc': 'Extract the amenities from the estate details.'}], desc='Extract the amenities from the estate details.', depends_on=['Details'])
    ds = ds.sem_filter('Is there a swimming pool in the estate.', depends_on=['Amenities'])
    ds = ds.sem_filter('Is there a gym in the estate.', depends_on=['Amenities'])
    ds = ds.sem_filter('Is the estate located in Lekki, Lagos.', depends_on=['Location'])
    return ds

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_abacus
    return run_abacus(build(tables), job, aggregate=AGGREGATE)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
