from .SGNet import SGNet
from .SGNet_CVAE import SGNet_CVAE

_META_ARCHITECTURES = {
    'SGNet':SGNet,
    'SGNet_CVAE':SGNet_CVAE,
}


def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
