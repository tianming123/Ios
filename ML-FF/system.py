#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
此module可能是暂时的,有可能最后放到parameters.py或者preprocess_data.py这样的文件中

此module的作用主要是解决从指定的文件夹中搜集所有的MOVEMENT文件? vasprun文件?
然后每一个vasprun文件?按照从属的文件夹名给出一个system名字
更重要的是,要建立system和iamge的结构
'''
import parameters as pm
import os
import numpy as cp
# import cupy as cp
import pandas as pd
from utils.vasprun_data import VasprunData


class System():

    def __init__(self, sourcePath):
        '''
        初始化,记录源文件路径,目录

        检查是否有已经处理的文件,若有则可以节省相应的步骤

        若没有读取分拆源数据文件,读取到内存,并分别存储到不同的文件

        检查各个Image的dE波动是否符合要求,得到合格的Image数据的索引列表

        Determine Attributes Directly
        ---------------------
        sourcePath:                       str对象,存储该体系源文件,对于PWmat, 这个文件是MOVEMENT,对于VASP, 这个文件是vasprun.xml

        '''

        self.sourcePath = sourcePath
        self.dir = os.path.dirname(sourcePath)
        _, self.name = os.path.split(self.sourcePath)

    def loadData(self):
        '''
        此方法用于从源文件self.sourcePath或者已经分割的数据文件self.splittedDataFilePath中读取该system的主要数据

        Parameters
        ---------------------
        isSaveToFiles:                    bool对象,决定方法是否要将读取后的信息存储到各个文件中


        Returns
        ---------------------
        None


        Determine Attributes InDirectly
        ---------------------
        numOfImages:                      int对象,存储整个体系中的Image的个数
        numOfAtoms:                       int对象,存储整个体系中原子的总个数
        atomTypeList:                     list对象, int,numOfAtoms长度,存储整个体系中每个原子所属的原子种类
        allCell:                          cp.array对象,float,numOfImages*3*3,存储整个体系中所有Image的cell的信息
        allPos:                           cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的position的信息
        allForce:                         cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的force的信息
        allVelocity:                      cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的velocity的信息
        allEnergy:                        cp.array对象,float,numOfImages*numOfAtoms, 存储整个体系中所有Image的所有原子的energy的信息
        allEp:                            cp.array对象,float,numOfImages,存储整个体系中每个Image的Ep的信息
        allDE:                            cp.array对象,float,numOfImages,存储整个体系中每个Image的dE的信息
        atomTypeSet:                      tuple对象,int, 存储了体系中所包含的所有原子种类的信息
        atomCountAsType:                  cp.array对象,int,长度和atomTypeSet一致,存储了对应的每种原子的数目的信息
        '''
        if self.name == "vasprun.xml":
            self.load_data_from_vasp_file()

    def load_data_from_vasp_file(self):
        vasp_data = VasprunData(self.sourcePath)
        print(vasp_data.num_image)


    def readSourceFile(self, isSaveToFile=True):
        '''
        读取源文件 MOVEMENT? vasprun?,得到体系的各种必要数据
        默认情况下,读取完之后会将得到的数据分别存储到各个文件中
        最后各种信息具体存储到哪个文件,是由self.saveSplittedData方法实现的


        Parameters
        ---------------------
        isSaveToFiles:                    bool对象,决定方法是否要将读取后的信息存储到各个文件中


        Returns
        ---------------------
        None


        Determine Attributes Directly
        ---------------------
        numOfImages:                      int对象,存储整个体系中的Image的个数
        numOfAtoms:                       int对象,存储整个体系中原子的总个数
        atomTypeList:                     list对象, int,numOfAtoms长度,存储整个体系中每个原子所属的原子种类
        allCell:                          cp.array对象,float,numOfImages*3*3,存储整个体系中所有Image的cell的信息
        allPos:                           cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的position的信息
        allForce:                         cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的force的信息
        allVelocity:                      cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的velocity的信息
        allEnergy:                        cp.array对象,float,numOfImages*numOfAtoms, 存储整个体系中所有Image的所有原子的energy的信息
        allEp:                            cp.array对象,float,numOfImages,存储整个体系中每个Image的Ep的信息
        allDE:                            cp.array对象,float,numOfImages,存储整个体系中每个Image的dE的信息
        atomTypeSet:                      tuple对象,int, 存储了体系中所包含的所有原子种类的信息
        atomCountAsType:                  cp.array对象,int,长度和atomTypeSet一致,存储了对应的每种原子的数目的信息
        '''
        pass

    def checkCell(self):
        '''
        检查体系计算时,超胞是否不变,
        若不变,检查超胞是否是正交格子,
        若又是正交格子,检查是否三个方向基矢长度是否都大于2倍rCut
        若以上三者都符合,则可以不必考虑近邻有一个原子出现两次的情况


        Parameters
        ---------------------
        None


        Returns
        ---------------------
        None


        Determine Attributes Directly
        ---------------------
        isConstantCell:                   bool对象,存储整个体系中cell是否不变的信息
        isOrthogonalCell:                 bool对象,存储了整个体系中cell是否是’不变且是三个方向基矢两两互相垂直’的信息
        isCheckSupercell:                 bool对象,存储了是否需要在后续计算近邻结构信息时考虑更大的超胞里有两个相同位置的原子在同一个原子近邻的情况的信息
        '''
        pass

    def checkEtot(self):
        '''
        检查各个image的Ep和各个原子的能量和之差dE是否起伏很大
        若起伏小于固定值,则认为这个image是合格的,可以纳入计算


        Parameters
        ---------------------
        dEErrorLimit:                     float对象,若体系某个图像的原子平均dE偏离体系均值超过此数,则不被后续计算考虑,默认值pm.dEErrorLimit


        Returns
        ---------------------
        goodImageNum                      int对象，self.goodImageIndices的长度，即good images的数量


        Determine Attributes Directly
        ---------------------
        meanEtotError:                    1 cp.array对象,存储了所有Image的dE的均值
        goodImageIndices:                 cp.array对象,int类型,一维,存储了所有dE/self.numOfAtoms不超过pm.dEErrorLimit的Image的编号,是后续可以用于计算的Image的编号
        self.goodImageNum                 int对象，self.goodImageIndices的长度，即good images的数量
        '''
        pass

    def getGoodImageNum(self):
        '''
        返回self.goodImageNum,也就是system中好的image的个数


        Parameters
        ---------------------
        dEErrorLimit:                     float对象,若体系某个图像的原子平均dE偏离体系均值超过此数,则不被后续计算考虑,默认值pm.dEErrorLimit


        Returns
        ---------------------
        goodImageNum                      int对象，self.goodImageIndices的长度，即good images的数量


        Determine Attributes Indirectly
        ---------------------
        meanEtotError:                    1 cp.array对象,存储了所有Image的dE的均值
        goodImageIndices:                 cp.array对象,int类型,一维,存储了所有dE/self.numOfAtoms不超过pm.dEErrorLimit的Image的编号,是后续可以用于计算的Image的编号
        self.goodImageNum                 int对象，self.goodImageIndices的长度，即good images的数量
        '''
        pass

    def splitTrainAndTest(self, testRate, validation_rate):
        '''
        返回self.goodImageNum,也就是system中好的image的个数


        Parameters
        ---------------------
        dEErrorLimit:                     float对象,若体系某个图像的原子平均dE偏离体系均值超过此数,则不被后续计算考虑,默认值pm.dEErrorLimit


        Returns
        ---------------------
        None


        Determine Attributes Directly
        ---------------------
        self.testImageIndices:            cp.array对象,int成员,预备作为test的images的编号
        self.trainImageIndices:           cp.array对象,int成员,预备作为train的images的编号


        Determine Attributes Indirectly
        ---------------------
        meanEtotError:                    1 cp.array对象,存储了所有Image的dE的均值
        goodImageIndices:                 cp.array对象,int类型,一维,存储了所有dE/self.numOfAtoms不超过pm.dEErrorLimit的Image的编号,是后续可以用于计算的Image的编号
        self.goodImageNum                 int对象，self.goodImageIndices的长度，即good images的数量
        '''
        pass

    def calMaxNeighborNum(self, isSaveStructData=False):
        '''
        calMaxNeighborNum([isSaveStructData(bool)]) => int

        计算体系中最大近邻原子数,计算方式为对合格的每一个Image计算最大近邻原子数,然后取最大值,然后返回
        因为在计算过程中,会需要初步计算每个Image的近邻结构信息,所以可以用isSaveStructData决定是否将这些初步信息保存下来


        Parameters
        ---------------------
        isSaveStructData:                  bool对象,决定方法是否要将各个图像计算得到的初步的结构信息存储到文件中,默认值为False


        Returns
        ---------------------
        maxNeighborNum:                    int对象(并非体系的attribute),体系的所有图像所有原子中,最大的近邻原子数


        Determine Attributes Directly
        ---------------------

        maxNeighborNum:                    int对象,体系的最大近邻原子数
        featMaxNeighborNum:                dict对象,str对象->int对象,该system计算feat/dfeat时所使用的pm.maxNeighborNum(key值分别为'T'—txt feat,'C'-c bin feat,'F'-f bin feat,'DC'-c bin dfeat,'DF'-f bin dfeat)
        '''
        pass

    def getMaxNeighborNum(self, isSaveStructData=False):
        '''
        手动写的获取maxNeighborNum这个attr的get方法

        Parameters
        ---------------------
        isSaveStructData:                  bool对象,决定方法是否要将各个图像计算得到的初步的结构信息存储到文件中,默认值为False


        Returns
        ---------------------
        maxNeighborNum:                    int对象(并非体系的attribute),体系的所有图像所有原子中,最大的近邻原子数


        Determine Attributes Indirectly
        ---------------------

        maxNeighborNum:                    int对象,体系的最大近邻原子数
        featMaxNeighborNum:                dict对象,str对象->int对象,该system计算feat/dfeat时所使用的pm.maxNeighborNum(key值分别为'T'—txt feat,'C'-c bin feat,'F'-f bin feat,'DC'-c bin dfeat,'DF'-f bin dfeat)
        '''
        pass

    def calAtomCategoryDict(self):
        '''
        此方法用于在已经得到pm.atomTypeSet之后,计算其中的每一原子类型在System中的list(用cp.array存储)
        对每一种都有一个list,且长短不一,因此得到的结果self.atomCategoryDict是一个真正的dict对象,且其中每一个是cp.array


        Parameters
        ---------------------
        None


        Returns
        ---------------------
        None


        Determine Attributes Directly
        ---------------------
        atomCategoryDict:                    dict对象,int(原子种类) : cp.array(在体系所有原子中,属于该种原子(key)的所有原子的序号列表)
        '''
        pass

    def saveFeats(self, isAllNew=False):
        '''
        此方法计算该system的所有feat，并以featSaveForm指定的方式存储

        Parameters
        ---------------------
        featSaveForm:                       str对象,应为单个字母，决定以何种方式存储feat数据


        Returns
        ---------------------
        None:


        Determine Attributes
        ---------------------

        None
        '''
        pass

    def saveFeatAndDfeats(self):
        '''
        值得注意，以上存储时会将读取数据时需要的很多参数以txt的形式存在system的文件夹下的feat_info.txt文件中


        Parameters
        ---------------------
        featSaveForm:                       str对象,应为单个字母，决定以何种方式存储feat数据
        dfeatSaveForm:                      str对象,应为单个字母，决定以何种方式存储feat数据


        Returns
        ---------------------
        None:


        Determine Attributes
        ---------------------

        None
        '''
        pass

    def __getitem__(self, index):
        '''
        __getitem__(index(int))  =>  Image对象
        运算符重载函数,使得System的实例可以像list一样用[index]的形式索引到包含的Image
        此函数尚未完善,目前只支持单个整数的index的索引


        Parameters
        ---------------------
        index:                              int对象,希望返回的图像在体系中的序号


        Returns
        ---------------------
        Image(self,index):                  Image对象,即体系中第[index]个图像



        Determine Attributes
        ---------------------

        None
        '''
        pass

s = System("file_lib/vasprun.xml")
s.loadData()