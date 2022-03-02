#!usr/bin/env python
# -*- coding utf-8 -*-
"""
Project:    bs4
File:       vasprun_data.py
author:     Liu Duan-Yang
Email:      liuduanyangwj@qq.om
time:       2022/2/17  19:47
IDE:        PyCharm
"""

is_cupy = False
float_accuracy = 'single'
# import os
from bs4 import BeautifulSoup as Bs
import pandas as pd
import numpy as np
from . import chemical_element_table
from chemical_element_table import ChemicalElementTable

if float_accuracy.lower() == 'single':
    float_accuracy = np.float32
elif float_accuracy.lower() == 'double':
    float_accuracy = np.float64
if is_cupy:
    import cupy as cp
else:
    cp = np

# CSS提取原子类型信息的路径, 可以根据以后的版本修改
atomtype_info_path = "modeling > atominfo > array[name='atomtypes'] > set > rc > c"
incar_info_path = "modeling > incar"
parameters_info_path = "modeling > parameters"
init_cell_path = "modeling > structure[name='initialpos'] > crystal > varray[name='basis'] > v"
cells_info_path = "modeling > calculation > structure > crystal > varray[name='basis'] > v"
position_path = "modeling > calculation > structure > varray[name='positions'] > v"
forces_path = "modeling > calculation >  varray[name='forces'] > v"


class VasprunData:
    def __init__(self, vasprun_file_path):
        # SSH提取原子类型信息的路径, 可以根据以后的版本修改
        self.atomtype_info_path = atomtype_info_path
        self.incar_info_path = incar_info_path
        self.parameters_info_path = parameters_info_path
        self.init_cell_path = init_cell_path
        self.cells_info_path = cells_info_path
        self.position_path = position_path
        self.forces_path = forces_path

        self.soup = Bs(open(vasprun_file_path, encoding="utf-8"), "html.parser")

        self.incar_dict = {}
        self.parameters_dict = {}
        # 临时存储原子种类的位置, 具体如何优化, 以后再优化 (未实现)
        self.atom_symbol_list = []
        # 存储forces信息 格式为一三维数组 50*64*3 依次为MD数，原子数，三个维度
        self.track_poscar_info_list = []
        # 存储position信息 格式为一三维数组 50*64*3
        self.position_list = []

        self.array_dict = {}

        self.num_atom = 0  # 原子数
        self.num_image = 0  # MD数
        self.init_cell = None

        self.track_incar_info()
        self.track_poscar_info()
        self.track_parameters_info()
        self.track_init_cell()

    @staticmethod
    def track_dict(a_dict, a_selected_obj, para_type="i"):
        if para_type == "i":
            for item in a_selected_obj:
                text = item.get_text().strip()
                # 本意想看看能否自动转换类型, 目前看不成功, 不知道为啥
                if "type" in item.attrs:
                    type_str = item["type"]
                    if type_str == "int":
                        a_dict[item["name"]] = int(text)
                    elif type_str == "logical":
                        if text[0] == "T":
                            a_dict[item["name"]] = True
                        else:
                            a_dict[item["name"]] = False
                    else:
                        try:
                            a_dict[item["name"]] = float(text)
                        except:
                            a_dict[item["name"]] = text
                else:
                    try:
                        a_dict[item["name"]] = float(text)
                    except:
                        a_dict[item["name"]] = text
        if para_type == "v":
            for item in a_selected_obj:
                text = item.get_text().strip()
                a_dict[item["name"]] = text

    def track_incar_info(self):
        if self.incar_dict:
            return self.incar_dict
        self.track_dict(self.incar_dict, self.soup.select(self.incar_info_path + " > i"))
        self.track_dict(self.incar_dict, self.soup.select(self.incar_info_path + " > v"), "v")

        return self.incar_dict

    def track_parameters_info(self):
        if self.parameters_dict:
            return self.parameters_dict
        # parameters_info = self.soup.select(self.parameters_info_path)
        para_info_path = self.parameters_info_path
        while True:
            selected_obj_i = self.soup.select(para_info_path + " >i")
            selected_obj_v = self.soup.select(para_info_path + " >v")
            if not (len(selected_obj_i) + len(selected_obj_v)):
                break
            self.track_dict(self.parameters_dict, selected_obj_i)
            self.track_dict(self.parameters_dict, selected_obj_v, "v")
            para_info_path += " >separator"
            # print(para_info_path, self.parameters_dict, "\n" * 2)

        return self.parameters_dict

    def track_init_cell(self):
        if self.init_cell:
            return self.init_cell

        init_cell_str = self.soup.select(self.init_cell_path)
        df = pd.DataFrame(init_cell_str)
        df = df[0].str.split(expand=True)
        df = df.astype(float)
        self.init_cell = cp.array(df, dtype=float_accuracy).reshape(3, 3)
        return self.init_cell

    def track_poscar_info(self):
        if self.num_atom:
            return
        atomtype_info = self.soup.select(self.atomtype_info_path)
        a_list = []
        for item in atomtype_info:
            a_list.append(item.get_text().strip())
        ele_table = chemical_element_table.ChemicalElementTable()
        atom_dict = ele_table.read_element_tables()
        atomic_order_list = []
        num_one_type = 0
        for i in range(len(a_list)):
            if i % 5 == 0:
                num_one_type = int(a_list[i])
            elif i % 5 == 1:
                atomic_order = atom_dict[a_list[i]]
                self.atom_symbol_list.extend([a_list[i] for j in range(num_one_type)])
                atomic_order_list.extend([atomic_order for j in range(num_one_type)])
        self.array_dict["atomic_order"] = cp.array(atomic_order_list, dtype=np.int32)
        self.num_atom = len(self.atom_symbol_list)

    def track_cells(self):
        if "cells" in self.array_dict:
            return self.array_dict["cells"]
        cells_strs = self.soup.select(self.cells_info_path)
        df = pd.DataFrame(cells_strs)
        df = df[0].str.split(expand=True)
        df = df.astype(float)
        cells_array = cp.array(df, dtype=float_accuracy).reshape(-1, 3, 3)
        self.array_dict["cells"] = cells_array
        return cells_array

    def track_forces(self):
        if "forces" in self.array_dict:
            return self.array_dict["forces"]
        forces_strs = self.soup.select(self.forces_path)
        df = pd.DataFrame(forces_strs)
        df = df[0].str.split(expand=True)
        df = df.astype(float)
        forces_array = cp.array(df, dtype=float_accuracy).reshape(-1, self.num_atom, 3)
        self.array_dict["forces"] = forces_array
        return forces_array

    def track_positions(self):
        if "positions" in self.array_dict:
            return self.array_dict["positions"]
        pos_strs = self.soup.select(self.position_path)
        df = pd.DataFrame(pos_strs)
        df = df[0].str.split(expand=True)
        df = df.astype(float)
        pos_array = cp.array(df, dtype=float_accuracy).reshape(-1, self.num_atom, 3)
        self.array_dict["positions"] = pos_array
        self.num_image = len(pos_array)
        return pos_array

    def calc_dis_vect(self, index):
        pos_array = self.track_positions()[index]
        dis_frac_vect_array = pos_array[cp.newaxis, :, :] - pos_array[:, cp.newaxis, :]
        dis_frac_vect_array[dis_frac_vect_array > 0.5] -= 1.0
        dis_frac_vect_array[dis_frac_vect_array <= 0.5] += 1.0
        return dis_frac_vect_array

    def frac_to_cart(self, array, cell=None):
        if cell is None:
            cell = self.init_cell
        return array @ cell

    def run(self):
        self.track_incar_info()
        self.track_poscar_info()
        self.track_forces()
        self.track_positions()


if __name__ == "__main__":
    fp = "../file_lib/vasprun.xml"
    vp = VasprunData(fp)
    soup = vp.soup
    vp.track_poscar_info()
    df_f = vp.track_forces()
    pos = vp.track_positions()
    at = vp.atom_symbol_list
    pad = vp.parameters_dict

    print(pos)
    print(vp.num_atom)
    print(vp.num_image)
