"""nirvana: estate/Q5; common numeric v4; explicitly rewritten workload.
Source: original/estate/nirvana/Q5.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'nirvana'
QUERY_ID = 'estate/Q5'
SOURCE = 'original/estate/nirvana/Q5.py'

def build(tables):
    import nirvana as nv
    df = nv.DataFrame(tables['nirvana_estate'])
    df.semantic_filter(user_instruction='Whether the estate has more than 3 bedrooms', input_columns=['Title'])
    df.semantic_map(user_instruction='Extract Amenities of the estate from the estate details.', input_columns=['Details'], output_columns=['Amenities'])
    df.semantic_filter(user_instruction='Whether the estate has less than 6 bedrooms.', input_columns=['Title'])
    return df

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_nirvana
    return run_nirvana(build(tables), job)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
