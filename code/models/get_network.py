from .model import ResNet

def create_network(args):
    resnet_layer_settings = {'50':  [3, 4, 6, 3], 
                             '101': [3, 4, 23, 3]}
    if args.encoder == 'resnet50':
        setttings = resnet_layer_settings['50']
    elif args.encoder == 'resnet101':
        setttings = resnet_layer_settings['101']
    else:
        raise RuntimeError('network not found.' +
                           'The network must be either of resnet50 or resnet101.')

    net = ResNet(args.min_depth, args.max_depth, args.classes, args.classifier, args.inference, args.decoder, setttings)
    return net