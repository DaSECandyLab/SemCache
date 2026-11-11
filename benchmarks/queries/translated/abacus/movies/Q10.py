"""abacus: movies/Q10; common numeric v4; explicitly rewritten workload.
Source: original/movies/nirvana/Q10.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'abacus'
QUERY_ID = 'movies/Q10'
SOURCE = 'original/movies/nirvana/Q10.py'
AGGREGATE = {'function': 'COUNT', 'column': None}

def build(tables):
    from experiments.validation.native_api import load_adapter
    ds = load_adapter("abacus").source_dataset(tables['nirvana_movies'], QUERY_ID)
    ds = ds.sem_map(cols=[{'name': 'Genre', 'type': str | None, 'desc': 'According to the movie plot, extract the genre(s) of each movie.'}], desc='According to the movie plot, extract the genre(s) of each movie.', depends_on=['Plot'])
    ds = ds.sem_filter('The rating is higher than 8.5.', depends_on=['IMDB_rating'])
    ds = ds.sem_filter('The rating is lower than 9.', depends_on=['IMDB_rating'])
    ds = ds.sem_filter('The movie belongs to crime movies.', depends_on=['Genre'])
    return ds

def run(tables, job, semantic=None):
    from experiments.validation.faithful_runtime import run_abacus
    return run_abacus(build(tables), job, aggregate=AGGREGATE)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
