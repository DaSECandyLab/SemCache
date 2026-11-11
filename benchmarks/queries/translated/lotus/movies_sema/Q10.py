"""Sema Movies Q10: three native LOTUS filters."""
SYSTEM = 'lotus'
QUERY_ID = 'movies_sema/Q10'
SOURCE = 'original/movies/sema/Q10.sql'

def build(tables, model=None):
    import lotus
    if model is not None: lotus.settings.configure(lm=model)
    df = tables['movie'].copy()
    df = df.sem_filter('The {overview} suggests that the movie is an action movie.')
    df = df.sem_filter('The {overview} shows that the movie focuses on a central role.')
    df = df.sem_filter('The {overview} shows that the movie is based on historical events ')
    return df.loc[:, ['movie_id']].copy()

def run(tables, job, model=None):
    return build(tables, model=model)
