import torch

def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


def load_model(model, model_load_path):

    weights_state_dict = torch.load(model_load_path)

    weights_dict = {}
    for k, v in weights_state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
        
    model.load_state_dict(weights_dict)
    
    return model

def preprocess_state_dict(state_dict):
    
    processed_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        processed_state_dict[new_k] = v

    return processed_state_dict

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num
