"""Sema Movies Q10: three native semantic filters, then ordinary output projection."""
SYSTEM = 'nirvana'
QUERY_ID = 'movies_sema/Q10'
SOURCE = 'original/movies/sema/Q10.sql'

def build(tables):
    from experiments.validation.faithful_runtime import nirvana_frame
    df = nirvana_frame(tables['movie'])
    df.semantic_filter(user_instruction='The overview suggests that the movie is an action movie.', input_columns=['overview'])
    df.semantic_filter(user_instruction='The overview shows that the movie focuses on a central role.', input_columns=['overview'])
    df.semantic_filter(user_instruction='The overview shows that the movie is based on historical events ', input_columns=['overview'])
    return df

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_nirvana, output_columns
    output, cost, seconds = run_nirvana(build(tables), job)
    return output_columns(output, ['movie_id']), cost, seconds
