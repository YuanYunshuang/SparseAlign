
LABEL_COLORS = {
    'Unlabeled': (0, 0, 0),  # 0 Unlabeled
    'Buildings': (70, 70, 70),  # 1 Buildings
    'Fences': (100, 40, 40),  # 2 Fences
    'Other': (55, 90, 80),  # 3 Other
    'Pedestrians': (220, 20, 60),  # 4 Pedestrians
    'Poles': (153, 153, 153),  # 5 Poles
    'RoadLines': (157, 234, 50),  # 6 RoadLines
    'Roads': (128, 64, 128),  # 7 Roads
    'Sidewalks': (244, 35, 232),  # 8 Sidewalks
    'Vegetation': (107, 142, 35),  # 9 Vegetation
    'Vehicles': (0, 0, 142),  # 10 Vehicles
    'Walls': (102, 102, 156),  # 11 Walls
    'TrafficSign': (220, 220, 0),  # 12 TrafficSign
    'Sky': (70, 130, 180),  # 13 Sky
    'Ground': (81, 0, 81),  # 14 Ground
    'Bridge': (150, 100, 100),  # 15 Bridge
    'Railtrack': (230, 150, 140),  # 16 Railtrack
    'GuardRail': (180, 165, 180),  # 17 GuardRail
    'TrafficLight': (250, 170, 30),  # 18 TrafficLight
    'Static': (110, 190, 160),  # 19 Static
    'Dynamic': (170, 120, 50),  # 20 Dynamic
    'Water': (45, 60, 150),  # 21 Water
    'Terrain': (145, 170, 100)  # 22 Terrain
}


VALID_CLS_nuscenes = [
    [24],  # 1 drivable surface
    [17, 19, 20],  # 2 car
    [15, 16],  # 3 bus
    [18],  # 4 construction_vehicle
    [21],  # 5 motorcycle
    [14],  # 6 bicycle
    [22],  # 7 trailer
    [23],  # 8 truck
    [2, 3, 4, 5, 6, 7, 8],  # 9 pedestrian
    [12],  # 10 traffic_cone
    [25],  # 11 other_flat
    [26],  # 12 sidewalk
    [27],  # 13 terrain
    [28],  # 14 manmade
    [30],  # 15 vegetation
    [9],  # 16 barrier
]

CoSenseBenchmarks = {
    'CenterPoints': {
        0: [
            'vehicle.car',
        ],
        1: [
            'vehicle.truck',
        ],
        2: [
            'vehicle.bus',
        ],
        3: [
            'vehicle.motorcycle',
        ],
        4: [
            'vehicle.cyclist'
        ],
        5: [
            'human.pedestrian',
        ]
    },
    'Car': {
        0: ['vehicle.car']
    },
    'Detection3Dpseudo4WheelVehicle': {
        0: [
            'vehicle.car',
            'vehicle.van',
            # 'vehicle.truck',
            # 'vehicle.bus',
        ]  # four-wheel-vehicle
    },
    'Detection3DpseudoVehicle': {
        0: [
            'vehicle.car',
            'vehicle.van',
            'vehicle.truck',
            'vehicle.bus',
        ], # four-wheel-vehicle
        1: [
            'vehicle.motorcycle',
            'vehicle.cyclist',
            'vehicle.scooter'
        ]  # two-wheel-vehicle
    },
    'Detection3DpseudoAll': {
        0: [
            'vehicle.car',
            'vehicle.van',
            'vehicle.truck',
            # 'vehicle.bus',
        ],  # four-wheel-vehicle
        1: [
            'vehicle.motorcycle',
            'vehicle.cyclist',
            'vehicle.scooter'
        ],  # two-wheel-vehicle
        2: [
            'human.pedestrian',
            'human.wheelchair',
            'human.sitting'
        ]  # human
    }
}

OPV2V_TOWN_DICTIONARY = {
    '2021_08_20_21_48_35': 'Town06',
    '2021_08_18_19_48_05': 'Town06',
    '2021_08_20_21_10_24': 'Town06',
    '2021_08_21_09_28_12': 'Town06',
    '2021_08_22_07_52_02': 'Town05',
    '2021_08_22_09_08_29': 'Town05',
    '2021_08_22_21_41_24': 'Town05',
    '2021_08_23_12_58_19': 'Town05',
    '2021_08_23_15_19_19': 'Town04',
    '2021_08_23_16_06_26': 'Town04',
    '2021_08_23_17_22_47': 'Town04',
    '2021_08_23_21_07_10': 'Town10HD',
    '2021_08_23_21_47_19': 'Town10HD',
    '2021_08_24_07_45_41': 'Town10HD',
    '2021_08_24_11_37_54': 'Town07',
    '2021_08_24_20_09_18': 'Town04',
    '2021_08_24_20_49_54': 'Town04',
    '2021_08_24_21_29_28': 'Town04',
    '2021_08_16_22_26_54': 'Town06',
    '2021_08_18_09_02_56': 'Town06',
    '2021_08_18_18_33_56': 'Town06',
    '2021_08_18_21_38_28': 'Town06',
    '2021_08_18_22_16_12': 'Town06',
    '2021_08_18_23_23_19': 'Town06',
    '2021_08_19_15_07_39': 'Town06',
    '2021_08_20_16_20_46': 'Town06',
    '2021_08_20_20_39_00': 'Town06',
    '2021_08_20_21_00_19': 'Town06',
    '2021_08_21_09_09_41': 'Town06',
    '2021_08_21_15_41_04': 'Town05',
    '2021_08_21_16_08_42': 'Town05',
    '2021_08_21_17_00_32': 'Town05',
    '2021_08_21_21_35_56': 'Town05',
    '2021_08_21_22_21_37': 'Town05',
    '2021_08_22_06_43_37': 'Town05',
    '2021_08_22_07_24_12': 'Town05',
    '2021_08_22_08_39_02': 'Town05',
    '2021_08_22_09_43_53': 'Town05',
    '2021_08_22_10_10_40': 'Town05',
    '2021_08_22_10_46_58': 'Town06',
    '2021_08_22_11_29_38': 'Town06',
    '2021_08_22_22_30_58': 'Town05',
    '2021_08_23_10_47_16': 'Town04',
    '2021_08_23_11_06_41': 'Town05',
    '2021_08_23_11_22_46': 'Town04',
    '2021_08_23_12_13_48': 'Town05',
    '2021_08_23_13_10_47': 'Town05',
    '2021_08_23_16_42_39': 'Town04',
    '2021_08_23_17_07_55': 'Town04',
    '2021_08_23_19_27_57': 'Town10HD',
    '2021_08_23_20_47_11': 'Town10HD',
    '2021_08_23_22_31_01': 'Town10HD',
    '2021_08_23_23_08_17': 'Town10HD',
    '2021_08_24_09_25_42': 'Town07',
    '2021_08_24_09_58_32': 'Town07',
    '2021_08_24_12_19_30': 'Town07',
    '2021_09_09_13_20_58': 'Town03',
    '2021_09_09_19_27_35': 'Town01',
    '2021_09_10_12_07_11': 'Town04',
    '2021_09_09_23_21_21': 'Town03',
    '2021_08_21_17_30_41': 'Town05',
    '2021_08_22_13_37_16': 'Town06',
    '2021_08_22_22_01_17': 'Town05',
    '2021_08_23_10_51_24': 'Town05',
    '2021_08_23_13_17_21': 'Town05',
    '2021_08_23_19_42_07': 'Town10HD',
    '2021_09_09_22_21_11': 'Town02',
    '2021_09_11_00_33_16': 'Town10HD',
    '2021_08_18_19_11_02': 'Town06'
}