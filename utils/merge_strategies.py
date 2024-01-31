def get_layer_list(model, strategy, original_list):
    if strategy == 'all':
        layer_list = original_list
    elif strategy == 'nofc':
        layer_list = [name for name in original_list if 'out' not in name]
    return layer_list