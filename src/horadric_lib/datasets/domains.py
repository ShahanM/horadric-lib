"""Domain configurations for frequently used datasets."""

MOVIELENS_DOMAIN = {
    'user_col': 'userId',
    'item_col': 'movieId',
    'rating_col': 'rating',
    'time_col': 'timestamp',
    'title_col': 'title',
    'metadata_cols': ['genres'],
}

MIND_DOMAIN = {
    'user_col': 'userId',
    'item_col': 'itemId',
    'rating_col': 'rating',
    'time_col': 'timestamp',
    'title_col': 'title',
    'metadata_cols': ['genres'],
}


def get_domain_config(dataset_name: str) -> dict | None:
    domains = {
        'movielens': MOVIELENS_DOMAIN,
        'mind': MIND_DOMAIN,
    }
    return domains.get(dataset_name.lower())
