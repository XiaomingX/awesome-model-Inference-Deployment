import os

def check_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            os.makedirs(path)


def torchtensor2numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    pass
