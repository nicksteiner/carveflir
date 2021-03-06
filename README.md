# Airborne FLIR Processing Library
## Carbon in the Arctic Reservoir Experiment (CARVE)

## Quick-Start

1. Directory tree should be linked to look like this:

![dir-tree](https://docs.google.com/drawings/d/18p-AXV8w7iyfeQAAEaTy8ESl8FSVwra0EGLsGOpWzjQ/pub?w=600&h=607)

2. Initialize the SQLite database (may take a while):

```#!bash
$ python ./flir.py --write_database

```
3. Write the NETCDF product

```#!bash 
$ python ./flir.py --netcdf --product L1A
```
The ```flir.py``` script will write netcdf files to the ```./dat``` folder
