from .jaad_data_layer import JAADDataLayer
from .pie_data_layer import PIEDataLayer
from .ethucy_data_layer import ETHUCYDataLayer

def build_dataset(args, phase):
    print(args.dataset)
    if args.dataset in ['JAAD']:
        data_layer = JAADDataLayer
    elif args.dataset in ['PIE']:
        data_layer = PIEDataLayer
    elif args.dataset in ['ETH', 'HOTEL','UNIV', 'ZARA1', 'ZARA2']:
        data_layer = ETHUCYDataLayer
    return data_layer(args, phase)