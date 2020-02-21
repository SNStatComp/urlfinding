from yaml import load, FullLoader

def get_config(file):
    with open(file, 'r') as f:
        config = load(f, Loader=FullLoader)
    mapping = {val:key for key, val in config['input']['columns'].items() if (val is not None) and (not isinstance(val, list))}
    computed = {key:val for key, val in config['input']['columns'].items() if isinstance(val, list)}
    search = config['search']
    features = config['features']
    train = config['train']
    return mapping, computed, search, features, train
