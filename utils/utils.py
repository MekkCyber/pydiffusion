import torch

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1 ,t) # out = a[t] in the case of a 1D tensor a
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def identity(t, *args, **kwargs):
    return t

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        # Tensor.uniform_ : Fills tensor with numbers sampled from the continuous uniform distribution:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
    
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cycle(dl):
    while True:
        for data in dl:
            yield data

def function_convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image