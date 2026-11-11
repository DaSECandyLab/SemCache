"""nirvana: movies/Q11; common numeric v4; explicitly rewritten workload.
Source: original/movies/nirvana/Q11.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'nirvana'
QUERY_ID = 'movies/Q11'
SOURCE = 'original/movies/nirvana/Q11.py'
AGGREGATE = {'function': 'AVG', 'column': 'RuntimeMinutes'}

def build(tables):
    import nirvana as nv
    df = nv.DataFrame(tables['nirvana_movies'])
    df.semantic_map(user_instruction='According to the movie plot, extract the genre(s) of each movie.', input_columns=['Plot'], output_columns=['Genre'])
    df.semantic_filter(user_instruction='The rating is higher than 9.', input_columns=['IMDB_rating'])
    df.semantic_filter(user_instruction='The movie belongs to crime movies.', input_columns=['Genre'])
    df.semantic_map(user_instruction='Extract the movie runtime as a single numeric duration in minutes. Convert explicitly stated hours and minutes to total minutes; a bare numeric runtime means minutes. If the runtime is missing, ambiguous, or a range, return null. Return no unit suffix.', input_columns=['Runtime'], output_columns=['RuntimeMinutes'])
    return df

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_nirvana
    return run_nirvana(build(tables), job, aggregate=AGGREGATE)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
