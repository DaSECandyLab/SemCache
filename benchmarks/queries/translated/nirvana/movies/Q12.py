"""nirvana: movies/Q12; common numeric v4; explicitly rewritten workload.
Source: original/movies/nirvana/Q12.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'nirvana'
QUERY_ID = 'movies/Q12'
SOURCE = 'original/movies/nirvana/Q12.py'

def build(tables):
    import nirvana as nv
    df = nv.DataFrame(tables['nirvana_movies'])
    df.semantic_map(user_instruction='According to the movie plot, extract the genre(s) of each movie.', input_columns=['Plot'], output_columns=['Genre'])
    df.semantic_filter(user_instruction='The rating is higher than 9.', input_columns=['IMDB_rating'])
    df.semantic_filter(user_instruction='The movie belongs to crime movies.', input_columns=['Genre'])
    df.semantic_map(user_instruction='Extract the main character from the movie plot.', input_columns=['Plot'], output_columns=['Character'])
    return df

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_nirvana
    return run_nirvana(build(tables), job)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
