def get_layer_list(model, strategy, original_list):
    if strategy == 'all':
        layer_list = original_list
    elif strategy == 'noembed':
        layer_list = [name for name in original_list if not ('embedding' in name.split('.') or 'fc' in name.split('.'))]
    elif strategy == 'nomerge':
        layer_list = []
    elif strategy == 'finetuning':
        layer_list = [name for name in original_list if 'embedding' in name.split('.') or 'fc' in name]
    elif strategy == 'affinetoo':
        layer_list = [name for name in original_list if 'embedding' in name.split('.') or 'fc' in name or 'affine' in name.split('.')]
    elif strategy == 'onlyfc':
        layer_list = [name for name in original_list if 'fc' in name]
    elif strategy == 'onlyemb':
        layer_list = [name for name in original_list if 'embedding' in name.split('.')]
    elif strategy == 'l4-fc':
        layer_list = [name for name in original_list if 'fc' in name or 'layer4' in name.split('.')]
    return layer_list