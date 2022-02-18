#!usr/bin/env python
# -*- coding utf-8 -*-
"""
Project:    bs4
File:       vasprun.py.py
author:     Liu Duan-Yang
Email:      liuduanyangwj@qq.om
time:       2022/2/17  19:47
IDE:        PyCharm
"""
from bs4 import BeautifulSoup as Bs
import pandas as pd

# SSH提取原子类型信息的路径, 可以根据以后的版本修改
atomtype_info_path = "modeling > atominfo > array[name='atomtypes'] > set > rc > c"
forces_path = "modeling > calculation >  varray[name='forces'] > v"
incar_info_path = "incar > i "

class VaspRun:
    def __init__(self, vasprun_file_path):
        self.vasprun_file_path = vasprun_file_path
        self.soup = Bs(open(self.vasprun_file_path, encoding="utf-8"), "html.parser")
        self.incar_dict = {}
        # 临时存储原子种类的位置, 具体如何优化, 以后再优化 (未实现)
        self.atomtype_info_list = []
        self.track_poscar_info()

    def track_incar_dict(self):
        incar_info = self.soup.select(incar_info_path)
        for item in incar_info:
            self.incar_dict[item["name"]] = item.get_text().strip()

    def track_poscar_info(self):
        atomtype_info = self.soup.select(atomtype_info_path)
        a_list = []
        for item in atomtype_info:
            a_list.append(item.get_text().strip())
        self.atomtype_info_list = a_list

    def track_forces(self):
        forces_strs = self.soup.select(forces_path)
        df = pd.DataFrame(forces_strs)
        df = df[0].str.split(expand=True)
        df = df.astype(float)
        return df
        # return self.soup.find_all("varray", attrs={"name": "forces"})


if __name__ == "__main__":
    fp = "file_lib/vasprun.xml"
    vp = VaspRun(fp)
    soup = vp.soup
    # vp.track_poscar_info()
    # df_f = vp.track_forces()
    # at = vp.atomtype_info_list
    vp.track_incar_dict()
    print(vp.incar_dict)


