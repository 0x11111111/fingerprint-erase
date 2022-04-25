from types import SimpleNamespace


def sn2dict(sn):
    d = dict()
    if isinstance(sn, SimpleNamespace):
        for k, v in sn.__dict__.items():
            if isinstance(v, SimpleNamespace):
                d[k] = sn2dict(v)
            else:
                d[k] = v

    return d
