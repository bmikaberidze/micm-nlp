from .string_f1 import StringF1


def _metric(*args, **kwargs):
    return StringF1()
