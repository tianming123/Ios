from utils.vaspUtils import VaspData

if __name__ == '__main__':
    vd = VaspData('file_lib/vasprun.xml')
    vd.run()
