"""abacus: movies/Q12; common numeric v4; explicitly rewritten workload.
Source: original/movies/nirvana/Q12.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'abacus'
QUERY_ID = 'movies/Q12'
SOURCE = 'original/movies/nirvana/Q12.py'

def build(tables):
    from experiments.validation.native_api import load_adapter
    ds = load_adapter("abacus").source_dataset(tables['nirvana_movies'], QUERY_ID)
    ds = ds.sem_map(cols=[{'name': 'Genre', 'type': str | None, 'desc': 'According to the movie plot, extract the genre(s) of each movie.'}], desc='According to the movie plot, extract the genre(s) of each movie.', depends_on=['Plot'])
    ds = ds.sem_filter('The rating is higher than 9.', depends_on=['IMDB_rating'])
    ds = ds.sem_filter('The movie belongs to crime movies.', depends_on=['Genre'])
    ds = ds.sem_map(cols=[{'name': 'Character', 'type': str | None, 'desc': 'Extract the main character from the movie plot.'}], desc='Extract the main character from the movie plot.', depends_on=['Plot'])
    return ds

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_abacus
    return run_abacus(build(tables), job, reduction=False)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
