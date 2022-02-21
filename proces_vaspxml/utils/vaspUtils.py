from bs4 import BeautifulSoup as Bs
import pandas as pd


class VaspData:
    def __init__(self, vasprun_file_path):
        # SSH提取原子类型信息的路径, 可以根据以后的版本修改
        self.atomtype_info_path = "modeling > atominfo > array[name='atomtypes'] > set > rc > c"
        self.forces_path = "modeling > calculation >  varray[name='forces'] > v"
        self.incar_info_path = "incar > i "
        self.position_path = "calculation > structure > varray[name='positions']"
        self.soup = Bs(open(vasprun_file_path, encoding="utf-8"), "html.parser")
        self.incar_dict = {}
        # 临时存储原子种类的位置, 具体如何优化, 以后再优化 (未实现)
        self.atomtype_info_list = []
        self.track_poscar_info_list = pd.DataFrame()
        # 存储position信息 格式为一三维数组
        self.position_list = []

    def track_incar_dict(self):
        incar_info = self.soup.select(self.incar_info_path)
        for item in incar_info:
            self.incar_dict[item["name"]] = item.get_text().strip()

    def track_poscar_info(self):
        atomtype_info = self.soup.select(self.atomtype_info_path)
        a_list = []
        for item in atomtype_info:
            a_list.append(item.get_text().strip())
        self.atomtype_info_list = a_list

    def track_forces_info(self):
        forces_strs = self.soup.select(self.forces_path)
        df = pd.DataFrame(forces_strs)
        df = df[0].str.split(expand=True)
        df = df.astype(float)
        self.track_poscar_info_list = df

    def track_position(self):
        position = self.soup.select(self.position_path)
        ret = []
        for elem in position:
            tmp = []
            position_info = elem.select('v')
            for elem in position_info:
                tmp.append(elem.get_text().strip().split())
            ret.append(tmp)
        self.position_list = ret


    def run(self):
        self.track_incar_dict()
        self.track_poscar_info()
        self.track_forces_info()
        self.track_position()