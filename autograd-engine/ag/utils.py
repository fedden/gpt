"""Utility functions."""


def flatten(container):
    """Flatten a nested container."""
    for i in container:
        if isinstance(i, (list, tuple)):
            yield from flatten(i)
        else:
            yield i
