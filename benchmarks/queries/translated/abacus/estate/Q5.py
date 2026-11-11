"""abacus: estate/Q5; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q5.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'abacus'
QUERY_ID = 'estate/Q5'
SOURCE = 'original/estate/nirvana/Q5.py'

def build(tables):
    from experiments.validation.native_api import load_adapter
    ds = load_adapter("abacus").source_dataset(tables['nirvana_estate'], QUERY_ID)
    ds = ds.sem_filter('Whether the estate has more than 3 bedrooms', depends_on=['Title'])
    ds = ds.sem_map(cols=[{'name': 'Amenities', 'type': str | None, 'desc': 'Extract Amenities of the estate from the estate details.'}], desc='Extract Amenities of the estate from the estate details.', depends_on=['Details'])
    ds = ds.sem_filter('Whether the estate has less than 6 bedrooms.', depends_on=['Title'])
    return ds

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_abacus
    return run_abacus(build(tables), job, reduction=False)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
