import torch
import os


def update_kl_weight(alpha, current_epoch, total_epochs):
    return max((1-alpha) * (1 - current_epoch / total_epochs), 0.2)

def update_supCon_weight(alpha, current_epoch, total_epochs):
    return min(alpha * (current_epoch / total_epochs), 0.8)


import contextlib
@contextlib.contextmanager
def dummy_context():
    '''default context manager that does nothing.'''
    yield

def getBinaryTensor(inputTensor, boundary):
    one = torch.ones_like(inputTensor)
    zero = torch.zeros_like(inputTensor)
    return torch.where(inputTensor > boundary, one, zero)


def normalized_labels_va(unnormalized_labels):
    mean_vals = unnormalized_labels.mean(dim=0)[0]
    std_vals = unnormalized_labels.std(dim=0)[0]
    normalized_labels = (unnormalized_labels - mean_vals) / std_vals
    return normalized_labels

def save_checkpoint(fabric, choice_modality, model, args, curr_time):
    save_state = {'model': model.state_dict(), 'config': args}
    save_path = os.path.join(args.save_model_path, 'best_model_{}_{}.pth').format(choice_modality, curr_time)
    fabric.print('Saving best_model:'+save_path)
    torch.save(save_state, save_path)

def load_checkpoint(fabric, choice_modality, save_Model_path, best_model_time):
    save_model_name = 'best_model_{}_{}.pth'.format(choice_modality, best_model_time)
    load_path = os.path.join(save_Model_path, save_model_name)
    fabric.print("*"*50)
    fabric.print('Prepare to evaluate the best model:'+save_model_name)
    checkpoint = torch.load(load_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    del checkpoint
    torch.cuda.empty_cache()
    return checkpoint_model

