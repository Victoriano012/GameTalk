import inspect

def autoassign(init):
    def wrapper(self, *args, **kwargs):
        bound = inspect.signature(init).bind(self, *args, **kwargs)
        bound.apply_defaults()
        for name, value in list(bound.arguments.items())[1:]:  # skip self
            setattr(self, name, value)
        init(self, *args, **kwargs)
    return wrapper
    
def masked_call(cls, queries, mask, unpack=True):
    filtered_inputs = [q for q, m in zip(queries, mask) if m] # Extract elements where mask is 1
    if filtered_inputs == []:
        return [""]*len(queries)
    
    filtered_outputs = cls(filtered_inputs)                   # Call cls once with all necessary elements
    if not unpack:
        return filtered_outputs
    
    output_iter = iter(filtered_outputs)                      # Iterator to retrieve processed elements
    return [next(output_iter) if m else "" for m in mask]