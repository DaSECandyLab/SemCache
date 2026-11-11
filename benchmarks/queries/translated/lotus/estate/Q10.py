"""lotus: estate/Q10; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q10.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'lotus'
QUERY_ID = 'estate/Q10'
SOURCE = 'original/estate/nirvana/Q10.py'
AGGREGATE = {'function': 'MIN', 'column': 'Price'}

def build(tables, model=None):
    import pandas as pd
    import lotus
    if model is not None: lotus.settings.configure(lm=model)
    df = tables['nirvana_estate'].copy()
    df = df.sem_map('Extract the house price from the detail about the estate. Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. Expand thousand/k, million/m and billion/b to their numeric multipliers. This dataset uses NGN when currency is omitted. If no unique price is stated, the price is a range, or a different currency is stated, return null. Do not infer a price from bedroom counts or other unrelated numbers.\nInputs: Details: {Details}', suffix='Price')
    df = df.sem_filter('Whether the estate has more than 3 bedrooms.\nInputs: Title: {Title}')
    df = df.sem_filter('Whether the estate has less than 6 bedrooms.\nInputs: Title: {Title}')
    df = df.sem_filter('Whether the estate is a detached duplex.\nInputs: Title: {Title}')
    from experiments.validation.faithful_runtime import aggregate_frame
    df = aggregate_frame(df, AGGREGATE)
    return df

def run(tables, job, model=None):
    from experiments.validation.faithful_runtime import run_lotus
    return run_lotus(build(tables, model=model), job)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
