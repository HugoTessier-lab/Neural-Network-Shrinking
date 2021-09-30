import torch
from semantic_segmentation.networks import hrnet
import time


def test_network(model, name):
    begin = time.time()
    model.eval()
    i = torch.rand(1, 3, 512, 1024)
    out = torch.Tensor([])
    try:
        out = model(i)
        assert out.shape[-2:] == i.shape[-2:]
    except AssertionError:
        print(f'\tError when testing {name} : output shape {out.shape[-2:]} is not input shape {i.shape[-2:]}')
    except Exception as e:
        print(f'\tError when testing {name} : {str(e)}')
    else:
        print(f'\t{name} OK')
    finally:
        print(f'\t\tRuntime : {round(time.time() - begin, 1)}s')


def test_hrnet():
    print('\nTesting HRNet standalone networks...\n')
    model_name = 'HRNet-18'
    try:
        model = hrnet.hrnet18(True, gates=False, mappers=False)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-18-gates'
    try:
        model = hrnet.hrnet18(True, gates=True, mappers=False)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-18-mappers'
    try:
        model = hrnet.hrnet18(True, gates=False, mappers=True)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-18-gates-mappers'
    try:
        model = hrnet.hrnet18(True, gates=True, mappers=True)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-32'
    try:
        model = hrnet.hrnet32(True, gates=False, mappers=False)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-32-gates'
    try:
        model = hrnet.hrnet32(True, gates=True, mappers=False)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-32-mappers'
    try:
        model = hrnet.hrnet32(True, gates=False, mappers=True)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-32-gates-mappers'
    try:
        model = hrnet.hrnet32(True, gates=True, mappers=True)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-48'
    try:
        model = hrnet.hrnet48(True, gates=False, mappers=False)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-48-gates'
    try:
        model = hrnet.hrnet48(True, gates=True, mappers=False)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-48-mappers'
    try:
        model = hrnet.hrnet48(True, gates=False, mappers=True)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)
    model_name = 'HRNet-48-gates-mappers'
    try:
        model = hrnet.hrnet48(True, gates=True, mappers=True)
    except Exception as e:
        print(f'\tError when instantiating network {model_name} : {str(e)}')
    else:
        test_network(model, model_name)


if __name__ == '__main__':
    with torch.no_grad():
        # HRNET
        test_hrnet()
