"""Sema Movies Q10: three native semantic filters, then ordinary output projection."""
SYSTEM = 'abacus'
QUERY_ID = 'movies_sema/Q10'
SOURCE = 'original/movies/sema/Q10.sql'

def build(tables):
    from experiments.validation.native_api import load_adapter
    ds = load_adapter('abacus').source_dataset(tables['movie'], QUERY_ID)
    ds = ds.sem_filter('The overview suggests that the movie is an action movie.', depends_on=['overview'])
    ds = ds.sem_filter('The overview shows that the movie focuses on a central role.', depends_on=['overview'])
    ds = ds.sem_filter('The overview shows that the movie is based on historical events ', depends_on=['overview'])
    return ds

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_abacus, output_columns
    return output_columns(run_abacus(build(tables), job), ['movie_id'])
