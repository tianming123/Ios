import os

from bs4 import BeautifulSoup as Bs
import pandas as pd
import requests
import xlwt
import bs4

class VaspData:
    def __init__(self, vasprun_file_path):
        # SSH提取原子类型信息的路径, 可以根据以后的版本修改
        self.atomtype_info_path = "modeling > atominfo > array[name='atomtypes'] > set > rc > c"
        self.forces_path = "modeling > calculation >  varray[name='forces']"
        self.incar_info_path = "incar > i "
        self.position_path = "calculation > structure > varray[name='positions']"

        self.soup = Bs(open(vasprun_file_path, encoding="utf-8"), "html.parser")

        self.incar_dict = {}
        # 临时存储原子种类的位置, 具体如何优化, 以后再优化 (未实现)
        self.atomtype_info_list = []
        # 存储forces信息 格式为一三维数组 50*64*3
        self.track_poscar_info_list = []
        # 存储position信息 格式为一三维数组 50*64*3
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
        au = AtomUtils()
        atom_dict = au.read_element_tables()
        self.atomtype_info_list = [a_list[1],atom_dict[a_list[1]]]

    def track_forces_info(self):
        forces_strs = self.soup.select(self.forces_path)
        ret = []
        for elem in forces_strs:
            tmp = []
            forces_info = elem.select('v')
            for e in forces_info:
                tmp.append(e.get_text().strip().split())
            ret.append(tmp)
        self.track_poscar_info_list = ret

    def track_position(self):
        position_str = self.soup.select(self.position_path)
        ret = []
        for elem in position_str:
            tmp = []
            position_info = elem.select('v')
            for e in position_info:
                tmp.append(e.get_text().strip().split())
            ret.append(tmp)
        self.position_list = ret


    def run(self):
        self.track_incar_dict()
        self.track_poscar_info()
        self.track_forces_info()
        self.track_position()

class AtomUtils:

    def __init__(self):
        self.file_path = '../file_lib/element_table.xls'

    def get_element_tables(self):
        # 爬取元素周期表
        homepage = "https://www.webelements.com"
        r = requests.get("https://www.webelements.com")
        demo = r.text
        soup = Bs(demo, "html.parser")
        elemlis = soup.find_all('div', 'sym')
        i = 1
        result = []
        title = ["Name","Symbol","Atomic number","Relative atomic mass","Standard state","Appearance","Classification","Group in periodic table","Group name",
            "Period in periodic table","Block in periodic table","Shell structure","CAS Registry",
            "Density of solid","Molar volume","Thermal conductivity","Melting point","Boiling point","Enthalpy of fusion",
            "Atomic radius (empirical)","Molecular single bond covalent radius","van der Waals radius","Pauling electronegativity","Allred Rochow electronegativity","Mulliken-Jaffe electronegativity",
            "First ionisation energy","Second ionisation energy","Third ionisation energy","Universe","Crustal rocks","Human","Human abundance by weight"]
            # 第一行
        result.append(title)

        for tag in elemlis: # 遍历元素
            element = tag.a.attrs['href']   #元素的名字 也就是每个元素网页上url中不同的部分
            url = homepage + '/' + element
            r1 = requests.get(url)   #提取每个元素的信息 写入二维数组
            demo1 = r1.text;
            soup1 = Bs(demo1, "html.parser")
            list = []
            i = 0  # 记录当前是第几个信息
            for ul in soup1.find_all('ul', 'ul_facts_table'): #遍历元素的第一组表格
                for lis in ul.contents:
                    if type(lis) == bs4.element.Tag:
                        i = i + 1
                        info = lis.contents[1][1:]
                        if i == 4: # 处理特殊情况
                            if info[-11:] == '[see notes ':
                                info = info[0:-11]
                            elif info[-10:] == '[see note ':
                                info = info[0:-10]
                        list.append(info)
            for ul in soup1.find_all('ul', 'spark_table_list'): #遍历元素的第二组表格
                for lis in ul.contents:
                    if type(lis) == bs4.element.Tag:
                        info = ""
                        i = i + 1
                        for i in range(1,len(lis.contents)):
                            info += lis.contents[i].string
                        list.append(info[1:])
            for i in range(len(list)):
                list[i] = list[i].strip()
            result.append(list)
            print(list)


        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
        # 将数据写入第 i 行，第 j 列
        i = 0
        for data in result:
            for j in range(len(data)):
                sheet1.write(i, j, data[j])
            i = i + 1
        f.save(self.file_path)  # 保存文件

    #  返回原子符号与序数对照表
    def read_element_tables(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(here, self.file_path)
        df = pd.read_excel(self.file_path, usecols=[0,1,2])
        ret = dict(zip(df["Symbol"], df["Atomic number"]))
        return ret
