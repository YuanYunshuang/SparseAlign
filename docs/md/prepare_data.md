# Prepare Datasets
## Download 
```shell
cd CoSense3D/cosense3d/tools
# bash download.sh [DATASET_NAME] [OUTPUT_DIR]
bash download.sh OPV2Vt data_path/OPV2Vt
bash download.sh DairV2Xt data_path/DairV2Xt
bash download.sh OPV2V data_path/OPV2V
```

### OPV2Vt 
```shell
├── train
    ├── 2021_08_16_22_26_54
        │── 10710
            │── 000070.ply
            ├── 000072.ply
            ├── ...
        │── ...
├── test
   ├── 2021_08_23_21_07_10
        │── 160
            │── 000071.ply
            ├── 000073.ply
            ├── ...
        │── ...
   ├── meta
        ├── 2021_09_10_12_07_11.json
        ├── test.txt
        └── train.txt

```


### OPV2V
```shell
├── train
    ├── 2021_08_16_22_26_54
        │── 10710
            │── 000070.bin
            ├── 000070.yaml
            ├── ...
        │── ...
├── test
   ├── 2021_08_23_21_07_10
        │── 160
            │── 000071.bin
            ├── 000071.yaml
            ├── ...
        │── ...
├── meta
    ├── 2021_09_10_12_07_11.json
    ├── 2021_09_10_12_07_15.json
    ├── ...
    ├── test.txt
    └── train.txt

```

## DairV2Xt and DairV2X

Download [DAIR-V2X-C](https://thudair.baai.ac.cn/coop-dtest) dataset and the new generated meta data (will be available at the publication) and extract and structure them as following.

```shell
├── cooperative-vehicle-infrastructure
  |── 2021_08_16_22_26_54
  |── ...
├── cooperative-vehicle-infrastructure-infrastructure-side-image
├── cooperative-vehicle-infrastructure-infrastructure-side-velodyne
├── cooperative-vehicle-infrastructure-vehicle-side-image
├── cooperative-vehicle-infrastructure-vehicle-side-velodyne
├── dairv2xt-meta
    ├── 2021_09_10_12_07_11.json
    ├── 2021_09_10_12_07_15.json
    ├── ...
    ├── test.txt
    └── train.txt
├── meta-coalign
    ├── 2021_09_10_12_07_11.json
    ├── 2021_09_10_12_07_15.json
    ├── ...
    ├── test.txt
    └── train.txt
```