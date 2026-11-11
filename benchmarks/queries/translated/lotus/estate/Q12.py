"""lotus: estate/Q12; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q12.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'lotus'
QUERY_ID = 'estate/Q12'
SOURCE = 'original/estate/nirvana/Q12.py'
AGGREGATE = {'function': 'AVG', 'column': 'Price'}

def build(tables, model=None):
    import pandas as pd
    import lotus
    if model is not None: lotus.settings.configure(lm=model)
    df = tables['nirvana_estate'].copy()
    df = df.sem_map('Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.\nInputs: Details: {Details}', suffix='Price')
    df = df.sem_map('Extract the amenities from the estate details.\nInputs: Details: {Details}', suffix='Amenities')
    df = df.sem_filter('Is there a swimming pool in the estate.\nInputs: Amenities: {Amenities}')
    df = df.sem_filter('Is there a gym in the estate.\nInputs: Amenities: {Amenities}')
    df = df.sem_filter('Is the estate located in Lekki, Lagos.\nInputs: Location: {Location}')
    from experiments.validation.faithful_runtime import aggregate_frame
    df = aggregate_frame(df, AGGREGATE)
    return df

def run(tables, job, model=None):
    from experiments.validation.faithful_runtime import run_lotus
    return run_lotus(build(tables, model=model), job)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
