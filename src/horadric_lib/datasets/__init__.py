try:
    import binpickle
    import datasets
    import pandas
    import sklearn
except ImportError as e:
    raise ImportError(
        'The datasets manager require extra dependencies. '
        "Please re-install using: uv add 'horadric-lib[datasets] @ git+...'"
    ) from e
