"""lotus: estate/Q5; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q5.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'lotus'
QUERY_ID = 'estate/Q5'
SOURCE = 'original/estate/nirvana/Q5.py'

def build(tables, model=None):
    import pandas as pd
    import lotus
    if model is not None: lotus.settings.configure(lm=model)
    df = tables['nirvana_estate'].copy()
    df = df.sem_filter('Whether the estate has more than 3 bedrooms\nInputs: Title: {Title}')
    df = df.sem_map('Extract Amenities of the estate from the estate details.\nInputs: Details: {Details}', suffix='Amenities')
    df = df.sem_filter('Whether the estate has less than 6 bedrooms.\nInputs: Title: {Title}')
    return df

def run(tables, job, model=None):
    from experiments.validation.faithful_runtime import run_lotus
    return run_lotus(build(tables, model=model), job)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
