"""lotus: movies/Q12; common numeric v4; explicitly rewritten workload.
Source: original/movies/nirvana/Q12.py
Run via experiments.validation.faithful_runtime; model settings are external.
"""
SYSTEM = 'lotus'
QUERY_ID = 'movies/Q12'
SOURCE = 'original/movies/nirvana/Q12.py'

def build(tables, model=None):
    import pandas as pd
    import lotus
    if model is not None: lotus.settings.configure(lm=model)
    df = tables['nirvana_movies'].copy()
    df = df.sem_map('According to the movie plot, extract the genre(s) of each movie.\nInputs: Plot: {Plot}', suffix='Genre')
    df = df.sem_filter('The rating is higher than 9.\nInputs: IMDB_rating: {IMDB_rating}')
    df = df.sem_filter('The movie belongs to crime movies.\nInputs: Genre: {Genre}')
    df = df.sem_map('Extract the main character from the movie plot.\nInputs: Plot: {Plot}', suffix='Character')
    return df

def run(tables, job, model=None):
    from experiments.validation.faithful_runtime import run_lotus
    return run_lotus(build(tables, model=model), job)

if __name__ == '__main__':
    from experiments.validation.faithful_runtime import query_main
    query_main(globals())
