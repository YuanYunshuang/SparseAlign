import socket

MACHINE_NAME = socket.gethostname() if socket.gethostname() in ["ISI", "mars"] else "default"
DATA_ROOT = {
    "default": "/data/to/datasets/COOD_datasets",
    "ISI": "/home/yuan/data/COOD_datasets",
    "mars": "/koko/datasets",
}[MACHINE_NAME]


def parse_opv2v_paths(cfgs):
    path_map = {
        "ISI": {
            "data": f"{DATA_ROOT}/OPV2Va",
            "meta": f"{DATA_ROOT}/OPV2Va/meta"
        },
    }
    cfgs['DATASET']['data_path'] = path_map[MACHINE_NAME]['data']
    cfgs['DATASET']['meta_path'] = path_map[MACHINE_NAME]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = True
    return cfgs

def parse_opv2vt_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": f"{DATA_ROOT}/OPV2Vt",
            "meta": f"{DATA_ROOT}/OPV2Vt/meta"
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
        "mars": {
            "data": "/koko/datasets/DairV2Xt",
            "meta": "/koko/datasets/DairV2Xt/meta-coalign"
        },
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = False
    return cfgs


def parse_paths_(cfgs):
    if 'opv2v' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_opv2v_paths(cfgs)
    elif 'opv2vt' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_opv2vt_paths(cfgs)
    elif 'dairv2xt' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_dairv2xt_paths(cfgs)
    elif 'dairv2x' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_dairv2x_paths(cfgs)
    return cfgs


def parse_paths(cfgs):
    if 'opv2v' == cfgs['DATASET']['data_path'].lower():
        cfgs['DATASET']['data_path'] = f"{DATA_ROOT}/OPV2Va"
        cfgs['DATASET']['meta_path'] = f"{DATA_ROOT}/OPV2Va/meta"
        cfgs['DATASET']['enable_split_sub_folder'] = True
    elif 'opv2vt' == cfgs['DATASET']['data_path'].lower():
        cfgs['DATASET']['data_path'] = f"{DATA_ROOT}/OPV2Vt"
        cfgs['DATASET']['meta_path'] = f"{DATA_ROOT}/OPV2Vt/meta"
        cfgs['DATASET']['enable_split_sub_folder'] = True
    elif 'dairv2xt' == cfgs['DATASET']['data_path'].lower():
        cfgs['DATASET']['data_path'] = f"{DATA_ROOT}/DairV2Xt"
        cfgs['DATASET']['meta_path'] = f"{DATA_ROOT}/DairV2Xt/meta"
        cfgs['DATASET']['enable_split_sub_folder'] = False
    elif 'dairv2x' == cfgs['DATASET']['data_path'].lower():
        cfgs['DATASET']['data_path'] = f"{DATA_ROOT}/DairV2Xt"
        cfgs['DATASET']['meta_path'] = f"{DATA_ROOT}/DairV2Xt/meta-coalign"
        cfgs['DATASET']['enable_split_sub_folder'] = False
    return cfgs