
def parse_opv2v_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/OPV2Va",
            "meta": "/home/yuan/data/OPV2Va/meta"
        },
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = True
    return cfgs

def parse_v2vreal_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/data/OPV2Va",
            "meta": "/home/data/OPV2Va/meta"
        },
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = True
    return cfgs


def parse_opv2vt_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/OPV2Vt",
            "meta": "/home/yuan/data/OPV2Vt/meta"
        },
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['map_path'] = path_map[name].get('map', None)
    cfgs['DATASET']['enable_split_sub_folder'] = True
    return cfgs


def parse_dairv2xt_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/DairV2Xt",
            "meta": "/home/yuan/data/DairV2Xt/meta"
        },
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = False
    return cfgs


def parse_dairv2x_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/DairV2Xt",
            "meta": "/home/yuan/data/DairV2Xt/meta-coalign"
        },
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = False
    return cfgs


def parse_paths(cfgs):
    if 'opv2v' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_opv2v_paths(cfgs)
    elif 'v2vreal' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_v2vreal_paths(cfgs)
    elif 'opv2vt' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_opv2vt_paths(cfgs)
    elif 'dairv2xt' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_dairv2xt_paths(cfgs)
    elif 'dairv2x' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_dairv2x_paths(cfgs)
    return cfgs