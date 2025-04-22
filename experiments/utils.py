import inspect

def autoassign(init):
    def wrapper(self, *args, **kwargs):
        bound = inspect.signature(init).bind(self, *args, **kwargs)
        bound.apply_defaults()
        for name, value in list(bound.arguments.items())[1:]:  # skip self
            setattr(self, name, value)
        init(self, *args, **kwargs)
    return wrapper