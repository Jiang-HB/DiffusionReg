class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        if key.startswith('__'):
            raise AttributeError
        return self.get(key, None)

    def __setattr__(self, key, value):
        if key.startswith('__'):
            raise AttributeError("Cannot set magic attribute '{}'".format(key))
        self[key] = value