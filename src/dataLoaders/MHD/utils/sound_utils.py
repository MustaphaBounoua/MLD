import torch

def unstack_tensor(tensor, dim=0):

    tensor_lst = []
    for i in range(tensor.size(dim)):
        tensor_lst.append(tensor[i])
    tensor_unstack = torch.cat(tensor_lst, dim=0)
    return tensor_unstack