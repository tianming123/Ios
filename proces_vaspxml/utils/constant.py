class vaspConstant():
    def __init__(self):
        # SSH提取原子类型信息的路径, 可以根据以后的版本修改
        self.atomtype_info_path = "modeling > atominfo > array[name='atomtypes'] > set > rc > c"
        self.forces_path = "modeling > calculation >  varray[name='forces'] > v"
        self.incar_info_path = "incar > i "