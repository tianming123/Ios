from utils.vaspUtils import VaspData

if __name__ == '__main__':
    vd = VaspData('file_lib/vasprun.xml')
    vd.run()
    print(vd.incar_dict)
    print(vd.atomtype_info_list)
    print(vd.track_poscar_info_list)