#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
此module可能是暂时的,有可能最后放到parameters.py或者preprocess_data.py这样的文件中

此module的作用主要是解决从指定的文件夹中搜集所有的MOVMENT文件
然后每一个MOVEMENT文件按照从属的文件夹名给出一个system名字
更重要的是,要建立system和iamge的结构
'''
import parameters as pm
import os
import numpy as np
import cupy as cp
import pandas as pd
from image import Image


class System():
    '''
    定义一个体系,也就是一个原子体系,具体的说是一次DFT跑出来的MOVEMENT的class
    
    Parameters
    ---------------------
    sourcePath:                          str对象,建立System对象的源文件,即MOVEMENT文件的路径
    basicFuncVect:                       list对象,长度应为2,分别为体系要使用的basicFunc和basicDfunc,默认值为None,体系会使用cosBasicFunc和cosBasicDfunc



    Variable Attributes
    ---------------------    
    sourcePath:                          str对象,存储该体系源文件,即MOVEMENT文件的路径
    dir:                                 str对象,存储该体系MOVEMENT文件所在的文件夹的路径,可以认为是该体系的工作文件夹
    name:                                str对象,存储了该体系的名字,目前其值与self.dir相同
    structDataDir:                       str对象,存储了该体系存储初步结构信息的文件夹的路径,就是在self.dir下的名为'struct_data'的文件夹
    featDir:                             (废弃)str对象,存储了该体系存储feat信息的文件夹的路径,就是在self.dir下的名为'feat'的文件夹
    dfeatDir:                            (废弃)str对象,存储了该体系存储dfeat信息的文件夹的路径,就是在self.dir下的名为'dfeat'的文件夹
    featFilePathDict:                    dict对象,str对象->str对象,'C','F','DC','DF'的key对应的value分别指向feat的C形式,F形式和dfeat的C形式,F形式的存储文件,'T'对应值指向pm.featTrainTxt
    splittedDataFilePath:                str对象,指向的文件以二进制形式(.npy)存储了system的cell,pos,force,velocity,energy,atomTypeList,dE,Ep等所有后续计算可能用到的信息
    featCFilePath:                       (废弃)str对象,指向的文件中,按self.goodImageIndices的顺序存储每个Image的feat数据，单个Image的feat array的存储按C的方式存储
    featFortranFilePath:                 (废弃)str对象,指向的文件中,按self.goodImageIndices的顺序存储每个Image的feat数据，单个Image的feat array的存储按fortran的方式存储
    dfeatCFilePath:                      (废弃)str对象,指向的文件中,按self.goodImageIndices的顺序存储每个Image的dfeat数据，单个Image的dfeat array的存储按C的方式存储
    dfeatFortranFilePath:                (废弃)str对象,指向的文件中,按self.goodImageIndices的顺序存储每个Image的dfeat数据，单个Image的dfeat array的存储按fortran的方式存储
    featInfoTxtPath:                     str对象,指向的文件中,以txt格式存储了读取feat和dfeat必要的一些数据
    infoForReadPath:                     str对象,指向的文件中,以txt格式存储了一些关于该system的信息,可能是用户关心的，但只是用于输出,不会在后续被程序读取
    maxNeighborInfoPath:                 str对象,指向的文件中,只以txt格式存储两个整数，第一个是该system的maxNeighborNum,第二个是目前计算feat和dfeat所用到的pm.maxNeighborNum
    basicFunc:                           function对象,存储了该体系使用的basicFunc,默认是用cosBasciFunc
    basicDfunc:                          function对象,存储了该体系使用的basicDfunc,默认是用cosBasicDfunc
    cellFilePath:                        (废弃)str对象,存储整个体系中所有Image的cell信息的文件路径,为在self.dir下的一个'cell.npy'文件,存储的是一个cp.array对象
    posFilePath:                         (废弃)str对象,存储整个体系中所有Image的所有原子position信息的文件路径,为在self.dir下的一个'pos.npy'文件,存储的是一个cp.array对象
    forceFilePath:                       (废弃)str对象,存储整个体系中所有Image的所有原子force信息的文件路径,为在self.dir下的一个'force.npy'文件,存储的是一个cp.array对象
    velocityFilePath:                    (废弃)str对象,存储整个体系中所有Image的所有原子velocity信息的文件路径,为在self.dir下的一个'velocity.npy'文件,存储的是一个cp.array对象
    energyFilePath:                      (废弃)str对象,存储整个体系中所有Image的所有原子energy信息的文件路径,为在self.dir下的一个'energy.npy'文件,存储的是一个cp.array对象
    atomTypeListFilePath:                (废弃)str对象,存储整个体系中每个原子所属原子种类信息的列表的文件路径,为在self.dir下的一个'atom_type_list.npy'文件,存储的是一个list对象
    dEAndEpFilePath:                     (废弃)str对象,存储整个体系中所有Image的dE和Ep的信息的文件路径,为在self.dir下的一个'dE_Ep.npy'文件,存储的是一个cp.array对象
    otherInfoFilePath:                   (废弃)str对象,存储整个体系中其它一些信息的文件路径,为在self.dir下的一个'other_info.txt'文件,以txt格式存储了Image数,原子数,原子种类表,各类原子个数
    statLogPath:                         (废弃)str对象,存储整个体系一些计算进度的文件路径,为在self.dir下的一个'stat_log.txt'文件,以txt格式存储是否已分割源数据及是否已计算feat和dfeat
    isDataSplitted:                      bool对象,存储了是否已将MOVEMENT文件信息分别存储到各个文件的信息
    isFeatGenned:                        bool对象,存储了是否已经成功计算feat和dfeat的信息
    isConstantCell:                      bool对象,存储整个体系中cell是否不变的信息
    isOrthogonalCell:                    bool对象,存储了整个体系中cell是否是’不变且是三个方向基矢两两互相垂直’的信息
    isCheckSupercell:                    bool对象,存储了是否需要在后续计算近邻结构信息时考虑更大的超胞里有两个相同位置的原子在同一个原子近邻的情况的信息
    meanEtotError:                       1 cp.array对象,存储了所有Image的dE的均值
    goodImageIndices:                    cp.array对象,int类型,一维,存储了所有dE/self.numOfAtoms不超过pm.dEErrorLimit的Image的编号,是后续可以用于计算的Image的编号
    goodImageNum                         int对象，self.goodImageIndices的长度，即good images的数量
    self.testImageIndices:               cp.array对象,int成员,预备作为test的images的编号
    self.trainImageIndices:              cp.array对象,int成员,预备作为train的images的编号
    numOfImages:                         int对象,存储整个体系中的Image的个数
    numOfAtoms:                          int对象,存储整个体系中原子的总个数
    atomTypeList:                        list对象, int,numOfAtoms长度,存储整个体系中每个原子所属的原子种类
    allCell:                             cp.array对象,float,numOfImages*3*3,存储整个体系中所有Image的cell的信息
    allPos:                              cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的position的信息
    allForce:                            cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的force的信息
    allVelocity:                         cp.array对象,float,numOfImages*numOfAtoms*3, 存储整个体系中所有Image的所有原子的velocity的信息
    allEnergy:                           cp.array对象,float,numOfImages*numOfAtoms, 存储整个体系中所有Image的所有原子的energy的信息
    allEp:                               cp.array对象,float,numOfImages,存储整个体系中每个Image的Ep的信息
    allDE:                               cp.array对象,float,numOfImages,存储整个体系中每个Image的dE的信息
    atomTypeSet:                         tuple对象,int, 存储了体系中所包含的所有原子种类的信息
    atomCountAsType:                     cp.array对象,int,长度和atomTypeSet一致,存储了对应的每种原子的数目的信息
    atomCategoryDict:                    dict对象,int(原子种类) : cp.array(在体系所有原子中,属于该种原子(key)的所有原子的序号列表)
    maxNeighborNum:                      int对象,该system的最大近邻原子数
    featMaxNeighborNum:                  dict对象,str对象->int对象,该system计算feat/dfeat时所使用的pm.maxNeighborNum(key值分别为'T'—txt feat,'C'-c bin feat,'F'-f bin feat,'DC'-c bin dfeat,'DF'-f bin dfeat)
    



    Method Attributes
    ---------------------


    Internal Method Attributes
    ---------------------

      
    '''
    def __init__(self,sourcePath,basicFuncVect=pm.basicFuncVect):
        '''
        初始化,记录源文件路径,目录

        检查是否有已经处理的步骤,若有则可以节省相应的步骤

        若没有读取分拆源数据文件,读取到内存,并分别存储到不同的文件

        检查各个Image的dE波动是否符合要求,得到合格的Image数据的索引列表



        Determine Attributes Directly
        ---------------------
        sourcePath:                       str对象,存储该体系源文件,即MOVEMENT文件的路径
        dir:                              str对象,存储该体系MOVEMENT文件所在的文件夹的路径,可以认为是该体系的工作文件夹
        name:                             str对象,存储了该体系的名字,目前其值与self.dir相同
        structDataDir:                    str对象,存储了该体系存储初步结构信息的文件夹的路径,就是在self.dir下的名为'struct_data'的文件夹
        featDir:                          (废弃)str对象,存储了该体系存储feat信息的文件夹的路径,就是在self.dir下的名为'feat'的文件夹
        dfeatDir:                         (废弃)str对象,存储了该体系存储dfeat信息的文件夹的路径,就是在self.dir下的名为'dfeat'的文件夹
        featFilePathDict:                 dict对象,str对象->str对象,'C','F','DC','DF'的key对应的value分别指向feat的C形式,F形式和dfeat的C形式,F形式的存储文件,'T'对应值指向pm.featTrainTxt
        featCFilePath:                    (废弃)str对象,指向的文件中,按self.goodImageIndices的顺序存储每个Image的feat数据，单个Image的feat array的存储按C的方式存储
        featFortranFilePath:              (废弃)str对象,指向的文件中,按self.goodImageIndices的顺序存储每个Image的feat数据，单个Image的feat array的存储按fortran的方式存储
        dfeatCFilePath:                   (废弃)str对象,指向的文件中,按self.goodImageIndices的顺序存储每个Image的dfeat数据，单个Image的dfeat array的存储按C的方式存储
        dfeatFortranFilePath:             (废弃)str对象,指向的文件中,按self.goodImageIndices的顺序存储每个Image的dfeat数据，单个Image的dfeat array的存储按fortran的方式存储
        splittedDataFilePath:             str对象,指向的文件以二进制形式(.npy)存储了system的cell,pos,force,velocity,energy,atomTypeList,dE,Ep等所有后续计算可能用到的信息
        self.featInfoTxtPath:             str对象,指向的文件中,以txt格式存储了读取feat和dfeat必要的一些数据
        self.infoForReadPath:             str对象,指向的文件中,以txt格式存储了一些关于该system的信息,可能是用户关心的，但只是用于输出,不会在后续被程序读取
        self.maxNeighborInfoPath:         str对象,指向的文件中,只以txt格式存储两个整数，第一个是该system的maxNeighborNum,第二个是目前计算feat和dfeat所用到的pm.maxNeighborNum
        basicFunc:                        function对象,存储了该体系使用的basicFunc,默认是用cosBasciFunc
        basicDfunc:                       function对象,存储了该体系使用的basicDfunc,默认是用cosBasicDfunc
        


        Determine Attributes Indirectly
        ---------------------

        By self.checkStat
        
        cellFilePath:                     str对象,存储整个体系中所有Image的cell信息的文件路径,为在self.dir下的一个'cell.npy'文件,存储的是一个cp.array对象
        posFilePath:                      str对象,存储整个体系中所有Image的所有原子position信息的文件路径,为在self.dir下的一个'pos.npy'文件,存储的是一个cp.array对象
        forceFilePath:                    str对象,存储整个体系中所有Image的所有原子force信息的文件路径,为在self.dir下的一个'force.npy'文件,存储的是一个cp.array对象
        velocityFilePath:                 str对象,存储整个体系中所有Image的所有原子velocity信息的文件路径,为在self.dir下的一个'velocity.npy'文件,存储的是一个cp.array对象
        energyFilePath:                   str对象,存储整个体系中所有Image的所有原子energy信息的文件路径,为在self.dir下的一个'energy.npy'文件,存储的是一个cp.array对象
        atomTypeListFilePath:             str对象,存储整个体系中每个原子所属原子种类信息的列表的文件路径,为在self.dir下的一个'atom_type_list.npy'文件,存储的是一个list对象
        dEAndEpFilePath:                  str对象,存储整个体系中所有Image的dE和Ep的信息的文件路径,为在self.dir下的一个'dE_Ep.npy'文件,存储的是一个cp.array对象
        otherInfoFilePath:                str对象,存储整个体系中其它一些信息的文件路径,为在self.dir下的一个'other_info.txt'文件,以txt格式存储了Image数,原子数,原子种类表,各类原子个数
        statLogPath:                      str对象,存储整个体系一些计算进度的文件路径,为在self.dir下的一个'stat_log.txt'文件,以txt格式存储是否已分割源数据及是否已计算feat和dfeat
        isDataSplitted:                   bool对象,存储了是否已将MOVEMENT文件信息分别存储到各个文件的信息
        isFeatGenned:                     bool对象,存储了是否已经成功计算feat和dfeat的信息

        
        By self.readSourceFile/self.loadFromSplittedData
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


        By self.checkCell
        ---------------------
        
        isConstantCell:                   bool对象,存储整个体系中cell是否不变的信息
        isOrthogonalCell:                 bool对象,存储了整个体系中cell是否是’不变且是三个方向基矢两两互相垂直’的信息
        isCheckSupercell:                 bool对象,存储了是否需要在后续计算近邻结构信息时考虑更大的超胞里有两个相同位置的原子在同一个原子近邻的情况的信息


        By self.checkEtot
        ---------------------

        meanEtotError:                    1 cp.array对象,存储了所有Image的dE的均值
        goodImageIndices:                 cp.array对象,int类型,一维,存储了所有dE/self.numOfAtoms不超过pm.dEErrorLimit的Image的编号,是后续可以用于计算的Image的编号
        self.goodImageNum                 int对象，self.goodImageIndices的长度，即good images的数量


        By self.calMaxNeighborNum
        ---------------------

        maxNeighborNum:                   int对象,该system的最大近邻原子数
        featMaxNeighborNum:               dict对象,str对象->int对象,该system计算feat/dfeat时所使用的pm.maxNeighborNum(key值分别为'T'—txt feat,'C'-c bin feat,'F'-f bin feat,'DC'-c bin dfeat,'DF'-f bin dfeat)


        By self.splitTrainAndTest
        ---------------------
        self.testImageIndices:            cp.array对象,int成员,预备作为test的images的编号
        self.trainImageIndices:           cp.array对象,int成员,预备作为train的images的编号

        '''

        self.sourcePath=sourcePath
        self.dir=os.path.dirname(sourcePath)
        self.name=self.dir
        self.structDataDir=os.path.join(self.dir,'struct_data')
        #self.featDir=os.path.join(self.dir,'feat')
        #self.dfeatDir=os.path.join(self.dir,'dfeat')        
        self.splittedDataFilePath=os.path.join(self.dir,'splitted_data.npy')
        self.featFilePathDict={}
        self.featFilePathDict['T']=pm.featTrainTxt
        self.featFilePathDict['C']=os.path.join(self.dir,'feat.cbin')
        self.featFilePathDict['F']=os.path.join(self.dir,'feat.fbin')
        self.featFilePathDict['DC']=os.path.join(self.dir,'dfeat.cbin')
        self.featFilePathDict['DF']=os.path.join(self.dir,'dfeat.fbin')
        '''
        self.featCFilePath=os.path.join(self.dir,'feat.cbin')
        self.featFortranFilePath=os.path.join(self.dir,'feat.fbin')
        self.dfeatCFilePath=os.path.join(self.dir,'dfeat.cbin')
        self.dfeatFortranFilePath=os.path.join(self.dir,'dfeat.fbin')
        '''
        self.featInfoTxtPath=os.path.join(self.dir,'feat_info.txt')
        self.infoForReadPath=os.path.join(self.dir,'read_info.txt')
        self.maxNeighborInfoPath=os.path.join(self.dir,'max_neighbor_info.txt')
        self.basicFuncVect=basicFuncVect


                    
        if not os.path.isdir(self.structDataDir):
            os.mkdir(self.structDataDir)

        '''
        if not os.path.isdir(self.featDir):
            os.mkdir(self.featDir)

        if not os.path.isdir(self.dfeatDir):
            os.mkdir(self.dfeatDir)
        

        
        self.checkStat()

        '''

        '''
        self.loadData()

        self.checkCell()

        self.checkEtot()
        '''






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
        if hasattr(self,'allPos'):
            return
        
        if os.path.isfile(self.splittedDataFilePath) and os.stat(self.splittedDataFilePath).st_mtime>os.stat(self.sourcePath).st_mtime:
            self.loadFromSplittedData()
        else:
            self.readSourceFile()
        
        


        
        
        



    def checkStat(self):
        '''
        (废弃)
        检查是否有处理过的步骤

        一个体系,若有数据处理的步骤信息,则会存储在目录下额stat_log.txt中
        目前主要有包含是否读取拆分源数据文件以及是否已经产生特征


        Parameters
        ---------------------        
        None
        

        Returns
        ---------------------        
        None
        

        Determine Attributes Directly
        ---------------------        
        cellFilePath:                     (废弃)str对象,存储整个体系中所有Image的cell信息的文件路径,为在self.dir下的一个'cell.npy'文件,存储的是一个cp.array对象
        posFilePath:                      (废弃)str对象,存储整个体系中所有Image的所有原子position信息的文件路径,为在self.dir下的一个'pos.npy'文件,存储的是一个cp.array对象
        forceFilePath:                    (废弃)str对象,存储整个体系中所有Image的所有原子force信息的文件路径,为在self.dir下的一个'force.npy'文件,存储的是一个cp.array对象
        velocityFilePath:                 (废弃)str对象,存储整个体系中所有Image的所有原子velocity信息的文件路径,为在self.dir下的一个'velocity.npy'文件,存储的是一个cp.array对象
        energyFilePath:                   (废弃)str对象,存储整个体系中所有Image的所有原子energy信息的文件路径,为在self.dir下的一个'energy.npy'文件,存储的是一个cp.array对象
        atomTypeListFilePath:             (废弃)str对象,存储整个体系中每个原子所属原子种类信息的列表的文件路径,为在self.dir下的一个'atom_type_list.npy'文件,存储的是一个list对象
        dEAndEpFilePath:                  (废弃)str对象,存储整个体系中所有Image的dE和Ep的信息的文件路径,为在self.dir下的一个'dE_Ep.npy'文件,存储的是一个cp.array对象
        otherInfoFilePath:                (废弃)str对象,存储整个体系中其它一些信息的文件路径,为在self.dir下的一个'other_info.txt'文件,以txt格式存储了Image数,原子数,原子种类表,各类原子个数
        statLogPath:                      (废弃)str对象,存储整个体系一些计算进度的文件路径,为在self.dir下的一个'stat_log.txt'文件,以txt格式存储是否已分割源数据及是否已计算feat和dfeat
        isDataSplitted:                   (废弃)bool对象,存储了是否已将MOVEMENT文件信息分别存储到各个文件的信息
        isFeatGenned:                     (废弃)bool对象,存储了是否已经成功计算feat和dfeat的信息
        '''
        
        '''
        self.cellFilePath=os.path.join(self.dir,'cell.npy')
        self.posFilePath=os.path.join(self.dir,'pos.npy')
        self.forceFilePath=os.path.join(self.dir,'force.npy')
        self.velocityFilePath=os.path.join(self.dir,'velocity.npy')
        self.energyFilePath=os.path.join(self.dir,'energy.npy')        
        self.atomTypeListFilePath=os.path.join(self.dir,'atom_type_list.npy')
        self.dEAndEpFilePath=os.path.join(self.dir,'dE_Ep.npy')
        self.otherInfoFilePath=os.path.join(self.dir,'other_info.txt')
        self.statLogPath=os.path.join(self.dir,'stat_log.txt')
        

        self.isDataSplitted=False
        self.isFeatGenned=False
        if os.path.isfile(self.statLogPath):
            with open(self.statLogPath) as statLog:
                line=statLog.readline()
                if line and line.split()[-1]=='True':
                    self.isDataSplitted=True
                line=statLog.readline()
                if line and line.split()[-1]=='True':
                    self.isFeatGenned=True
        '''
        if os.path.isfile(self.splittedDataFilePath):        
            pass
            


    def saveStat(self):
        '''
        (废弃)
        将数据处理的步骤信息存到此体系目录下的stat_log.txt中
        第一行存储的是此体系的源文件数据是否已经分割到各个文件中,前面是文字提示信息,最后是self.isDataSplitted
        第二行存储的是次体系的feat和dfeat是否已经产生,前面是文字提示信息,最后是self.isFeatGenned


        Parameters
        ---------------------        
        None
        

        Returns
        ---------------------        
        None



        Determine Attributes
        ---------------------

        None
        
        '''
        '''
        with open(self.statLogPath,'w') as statLog:
            statLog.write('Datas have been splitted to several files?    '+str(self.isDataSplitted)+'\n')
            statLog.write('Feats and dFeats have been generated?         '+str(self.isFeatGenned)+'\n')
        '''
        pass
        

    def readSourceFile(self,isSaveToFile=True):
        '''
        读取MOVEMENT源文件,得到体系的各种必要数据
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

        with open(self.sourcePath) as sourceFile:
            allData=sourceFile.read()
            self.numOfImages=allData.count('Iteration')
            self.numOfAtoms=int(allData.split()[0])
            


        
        cell=cp.zeros((3,3))
        pos=cp.zeros((self.numOfAtoms,3))
        force=cp.zeros((self.numOfAtoms,3))
        velocity=cp.zeros((self.numOfAtoms,3))
        energy=cp.zeros((self.numOfAtoms,))
        
        self.atomTypeList=[]
        self.allCell=cp.zeros((self.numOfImages,3,3))
        self.allPos=cp.zeros((self.numOfImages,self.numOfAtoms,3))
        self.allForce=cp.zeros((self.numOfImages,self.numOfAtoms,3))
        self.allVelocity=cp.zeros((self.numOfImages,self.numOfAtoms,3))
        self.allEnergy=cp.zeros((self.numOfImages,self.numOfAtoms))
        self.allEp=cp.zeros((self.numOfImages,))
        self.allDE=cp.zeros((self.numOfImages,))
        

        with open(self.sourcePath) as sourceFile:
            
            line=sourceFile.readline()
            
            for indexOfImage in range(self.numOfImages):

                self.allEp[indexOfImage]=float(line.split()[-2])
                while not "Lattice vector" in line:
                    line=sourceFile.readline()
                tempList=[]
                for i in range(3):
                    tempList.append(sourceFile.readline().split()[:3])
                cell=cp.array(np.array(tempList).astype('float64'))
                self.allCell[indexOfImage]=cp.array(np.array(tempList).astype('float64'))

                while not "Position" in line:
                    line=sourceFile.readline()
                tempList=[]                
                if indexOfImage==0:
                    for i in range(self.numOfAtoms):
                        L=sourceFile.readline().split()
                        self.atomTypeList.append(int(L[0]))
                        tempList.append(L[1:4])
                else:
                    for i in range(self.numOfAtoms):
                        L=sourceFile.readline().split()
                        tempList.append(L[1:4])
                pos=cp.array(np.array(tempList).astype('float64'))
                self.allPos[indexOfImage]=cp.array(np.array(tempList).astype('float64'))
                sourceFile.readline()
                tempList=[]
                for i in range(self.numOfAtoms):
                    tempList.append(sourceFile.readline().split()[1:4])
                force=cp.array(np.array(tempList).astype('float64'))
                self.allForce[indexOfImage]=cp.array(np.array(tempList).astype('float64'))


                sourceFile.readline()
                tempList=[]
                for i in range(self.numOfAtoms):
                    tempList.append(sourceFile.readline().split()[1:4])
                velocity=cp.array(np.array(tempList).astype('float64'))
                self.allVelocity[indexOfImage]=cp.array(np.array(tempList).astype('float64'))
                

                line=sourceFile.readline()                
                self.allDE[indexOfImage]=float(line.split()[-1])
                tempList=[]
                for i in range(self.numOfAtoms):
                    tempList.append(sourceFile.readline().split()[1:2])
                energy=cp.array(np.squeeze(np.array(tempList).astype('float64')))
                self.allEnergy[indexOfImage]=cp.array(np.squeeze(np.array(tempList).astype('float64')))

                while not "Iteration" in line and line:
                    line=sourceFile.readline()


        self.atomTypeSet=tuple(set(self.atomTypeList))
        '''
        tempList=[]
        for item in self.atomTypeList:
            if item not in tempList:
                tempList.append(item)
        self.atomTypeSet=tuple(tempList)
        '''
        self.atomCountAsType=cp.array([self.atomTypeList.count(i) for i in self.atomTypeSet])
            
        if isSaveToFile:
            self.saveSplittedData()
            #self.isDataSplitted=True
            #self.saveStat()

                    
                
    def saveSplittedData(self):
        ''' 
        (此存储法已废弃)
        将一个体系的各种信息分别存储到各个文件中

        allCell               ===>   cellFilePath
        allPos                ===>   posFilePath
        allforce              ===>   forceFilePath
        allCell               ===>   cellFilePath
        allPos                ===>   posFilePath
        allForce              ===>   forceFilePath
        allVelocity           ===>   velocityFilePath
        allEnergy             ===>   enerygFilePath
        allDE,allEp           ===>   dEAndEpFilePath

        numOfImages           ===>   otherInfoFilePath (txt,第一行最后)
        numOfAtoms            ===>   otherInfoFilePath (txt,第二行最后)
        atomTypeSet           ===>   otherInfoFilePath (txt,第四行)
        atomCountAsType       ===>   otherInfoFilePath (txt,第五行)

        目前的存储方式：

        上述所有数据，除去numOfImages,numOfAtoms外,全部存储到self.splittedDataFilePath中


        Parameters
        ---------------------        
        None
        

        Returns
        ---------------------        
        None


        Determine Attributes
        ---------------------

        None
        
        '''
        '''
        np.save(self.cellFilePath,self.allCell)
        np.save(self.posFilePath,self.allPos)
        np.save(self.forceFilePath,self.allForce)
        np.save(self.velocityFilePath,self.allVelocity)
        np.save(self.energyFilePath,self.allEnergy)
        np.save(self.atomTypeListFilePath,self.atomTypeList)
        np.save(self.dEAndEpFilePath,cp.concatenate((self.allDE[:,cp.newaxis],self.allEp[:,cp.newaxis]),axis=1))        
        with open(self.otherInfoFilePath,'w') as otherInfo:
            otherInfo.write('Number of images:  '+str(self.numOfImages)+'\n')
            otherInfo.write('Number of atoms:   '+str(self.numOfAtoms)+'\n')
            otherInfo.write('All atom types included and corresponding number of atoms in this system(first line:type, second line: number):\n')
            otherInfo.write(str(self.atomTypeSet)[1:-1]+'\n')
            otherInfo.write(str(self.atomCountAsType)[1:-1]+'\n')
        '''
        allData=(self.allCell,self.allPos,self.allForce,self.allVelocity,self.allEnergy,self.atomTypeList,self.allDE,self.allEp,self.atomTypeSet,self.atomCountAsType)        
        np.save(self.splittedDataFilePath,allData)
        

    

    def loadFromSplittedData(self):
        '''
        (此读取方式已废弃)
        从各个文件中读取体系的各种信息

        allCell               <===   cellFilePath
        allPos                <===   posFilePath
        allforce              <===   forceFilePath
        allCell               <===   cellFilePath
        allPos                <===   posFilePath
        allForce              <===   forceFilePath
        allVelocity           <===   velocityFilePath
        allEnergy             <===   enerygFilePath
        allDE,allEp           <===   dEAndEpFilePath

        numOfImages           <===   otherInfoFilePath (txt,第一行最后)
        numOfAtoms            <===   otherInfoFilePath (txt,第二行最后)
        atomTypeSet           <===   otherInfoFilePath (txt,第四行)
        atomCountAsType       <===   otherInfoFilePath (txt,第五行)


        目前的读取方式：
        
        除numOfImages,numOfAtoms外，全部从self.splittedDataFilePath读取
        numOfImages           <===   self.allPos.shape[0]
        numOfAtoms            <===   self.allPos.shape[1]
        


        Parameters
        ---------------------        
        None
        

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

        '''
        self.allCell=np.load(self.cellFilePath)
        self.allPos=np.load(self.posFilePath)
        self.allForce=np.load(self.forceFilePath)
        self.allVelocity=np.load(self.velocityFilePath)
        self.allEnergy=np.load(self.energyFilePath)
        self.atomTypeList=np.load(self.atomTypeListFilePath)
        dEAndEp=np.load(self.dEAndEpFilePath)
        self.allDE=dEAndEp[:,0]
        self.allEp=dEAndEp[:,1]
        with open(self.otherInfoFilePath) as otherInfo:
            line=otherInfo.readline().rstrip()
            self.numOfImages=int(line.split()[-1])
            line=otherInfo.readline().rstrip()
            self.numOfAtoms=int(line.split()[-1])
            line=otherInfo.readline()
            line=otherInfo.readline().rstrip()
            self.atomTypeSet=tuple([int(i) for i in line.split(',')])
            line=otherInfo.readline().rstrip()
            self.atomCountAsType=cp.array([int(i) for i in line.split()])
        '''
        self.allCell,self.allPos,self.allForce,self.allVelocity,self.allEnergy,self.atomTypeList,self.allDE,self.allEp,self.atomTypeSet,self.atomCountAsType=np.load(self.splittedDataFilePath,None,True)
        self.numOfImages=self.allPos.shape[0]
        self.numOfAtoms=self.allPos.shape[1]
            
        
        

        
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
        if hasattr(self,'isCheckSupercell'):
            return
        self.loadData()
        self.isConstantCell=(self.allCell.max(axis=0)==self.allCell.min(axis=0)).all()
        self.isOrthogonalCell=False
        self.isCheckSupercell=True
        cosAng=cp.zeros((3,))
        #sinAng=cp.zeros((3,))
        #areaVect=cp.zeros((3,))
        #self.heightVect=cp.zeros((3))
        if self.isConstantCell:
            latticeConstants=cp.linalg.norm(self.allCell[0],axis=1)
            volume=cp.abs(cp.linalg.det(self.allCell[0]))
            idx0=cp.arange(3)
            idxm=(idx0-1)%3
            idxp=(idx0+1)%3
            cosAng[idx0]=cp.sum(self.allCell[0,idxm]*self.allCell[0,idxp],axis=1)/(latticeConstants[idxm]*latticeConstants[idxp])
            '''
            cosAng[0]=cp.sum(self.allCell[0][1]*self.allCell[0][2])/latticeConstants[1]/latticeConstants[2]
            cosAng[1]=cp.sum(self.allCell[0][0]*self.allCell[0][2])/latticeConstants[0]/latticeConstants[2]
            cosAng[2]=cp.sum(self.allCell[0][1]*self.allCell[0][0])/latticeConstants[1]/latticeConstants[0]
            '''
            sinAng=cp.sqrt(1.0-cosAng*cosAng)
            areaVect=latticeConstants[idxm]*latticeConstants[idxp]*sinAng[idx0]
            '''
            areaVect[0]=latticeConstants[1]*latticeConstants[2]*sinAng[0]
            areaVect[1]=latticeConstants[0]*latticeConstants[2]*sinAng[1]
            areaVect[2]=latticeConstants[1]*latticeConstants[0]*sinAng[2]
            '''
            self.heightVect=volume/areaVect
        if abs(cosAng).max()<0.0001:
            self.isOrthogonalCell=True
        if self.heightVect.min()>pm.rCut*2.0:
            self.isCheckSupercell=False
        
        
        

    def checkEtot(self,dEErrorLimit=pm.dEErrorLimit):
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
        if hasattr(self,'goodImageIndices'):
            return
        self.loadData()
        self.meanEtotError=cp.mean(self.allDE)
        self.goodImageIndices=cp.where(cp.abs(self.allDE-self.meanEtotError)/self.numOfAtoms<dEErrorLimit)[0]
        self.goodImageNum=len(self.goodImageIndices)
        return self.goodImageNum


    def getGoodImageNum(self,dEErrorLimit=pm.dEErrorLimit):
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
        if hasattr(self,'goodImageNum'):
            return self.goodImageNum
        else:
            return self.checkEtot(dEErrorLimit)


    def splitTrainAndTest(self,testRate=pm.testRate):
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
        testImageNum=self.getGoodImageNum()*testRate
        meshGoodImageIndices=self.goodImageIndices.copy()
        cp.random.shuffle(meshGoodImageIndices)
        self.testImageIndices=meshGoodImageIndices[:testImageNum]
        self.trainImageIndices=meshGoodImageIndices[testImageNum:]
        


    def calMaxNeighborNum(self,isSaveStructData=False):
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
        self.featMaxNeighborNum={'T':0,'C':0,'F':0,'DC':0,'DF':0}
        if os.path.isfile(self.maxNeighborInfoPath) and os.stat(self.maxNeighborInfoPath).st_mtime > os.stat(self.sourcePath).st_mtime:
            with open(self.maxNeighborInfoPath,'r') as infoFile:
                line=infoFile.readline().rstrip().split()
                if float(line[0])!=pm.rCut:
                    os.remove(self.maxNeighborInfoPath)
                    return self.calMaxNeighborNum(isSaveStructData)
                self.maxNeighborNum=int(line[-1])
                line=infoFile.readline().rstrip()
                if line:
                    line=line.split()
                    self.featMaxNeighborNum['T']=int(line[0])
                    self.featMaxNeighborNum['C']=int(line[1])
                    self.featMaxNeighborNum['F']=int(line[2])
                    self.featMaxNeighborNum['DC']=int(line[3])
                    self.featMaxNeighborNum['DF']=int(line[4])
        else:
            
            self.checkEtot()
            maxNeighborNum=0
            for index in self.goodImageIndices:
                maxNeighborNum=max(maxNeighborNum,self[int(index)].getMaxNeighborNum(isSaveStructData))  #cp.array对象自动迭代时取出的对象也是cp.array对象,维度降低一维
                                                                                                         #即便用索引取出的对象也是cp.array对象,所以特别需要注意,np.array类似,
                                                                                                         #不过似乎不影响cp或者np内部的绝大部分应用场合,如判断,指标,但和list等混用不行
            self.maxNeighborNum=maxNeighborNum
            with open(self.maxNeighborInfoPath,'w') as infoFile:
                infoFile.write(str(pm.rCut)+' '+str(self.maxNeighborNum)+'\n')
        
        
        
            
        return self.maxNeighborNum


    def getMaxNeighborNum(self,isSaveStructData=False):
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
        if hasattr(self,'maxNeighborNum'):
            return self.maxNeighborNum
        else:
            return self.calMaxNeighborNum()
    

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
        self.loadData()
        self.atomCategoryDict={}
        for atomType in pm.atomTypeSet:
            #self.atomCategoryDict[atomType]=cp.array([i for i in range(self.numOfAtoms) if self.atomTypeList[i]==atomType])
            self.atomCategoryDict[atomType]=cp.where(cp.array(self.atomTypeList)==atomType)[0]


    def saveFeats(self,featSaveForm='T',isAllNew=False):
        '''
        此方法计算该system的所有feat，并以featSaveForm指定的方式存储
        featSaveForm取值的意义：
        'T'/'t':以csv的文本形式存储，每一行分别是:原子在system中的序号，原子的种类序号，原子的能量，原子的所有features
                值得注意的是，会将20%的image的数据存在m.workDir下的testData.txt，其余的存在trainData.txt
        'C'/'c':以二进制矩阵的形式先后存储每一个image的energy向量和feat矩阵，矩阵按C语言的顺序存储
        'F'/'f':以二进制矩阵的形式先后存储每一个image的energy向量和feat矩阵，矩阵按Fortran的顺序存储
                值得注意，若是二进制存储(C或F)，会将读取数据时需要的很多参数以txt的形式存在system的文件夹下的feat_info.txt文件中


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
        featSaveForm=featSaveForm.upper()
        #self.calMaxNeighborNum()
        if self.featMaxNeighborNum[featSaveForm]==pm.maxNeighborNum and (not isAllNew) and os.path.isfile(self.featFilePathDict[featSaveForm]) \
            and os.stat(self.featFilePathDict[featSaveForm]).st_mtime>os.stat(self.sourcePath).st_mtime:            
            return
        #self.checkEtot()
        self.checkCell()
        self.splitTrainAndTest()        
        if featSaveForm=='T':
            #testImageIndices=cp.random.choice(self.goodImageIndices,int(len(self.goodImageIndices)*0.2))
            for index in self.goodImageIndices:
                aImage=self[int(index)]
                feat=aImage.calFeat()
                out=np.concatenate((np.expand_dims(cp.asnumpy(self.atomTypeList),1),np.expand_dims(cp.asnumpy(aImage.energy),1),feat),axis=1)
                df_out=pd.DataFrame(out)
                if index in self.trainImageIndices:
                    df_out.to_csv(pm.featTrainTxt, mode='a', header=False)
                else:
                    df_out.to_csv(pm.featTestTxt, mode='a', header=False)
        else:
            with open(self.featInfoTxtPath,'w') as featInfo:
                featInfo.write('Number of  good images:   '+str(len(self.goodImageIndices))+'\n')
                featInfo.write('Number of atoms:    '+str(self.numOfAtoms)+'\n')
                featInfo.write('Number of real Feats:   '+str(pm.realFeatNum)+'\n')
                featInfo.write('Maxium of neighbor atoms:   '+str(pm.maxNeighborNum)+'\n')
                featInfo.write('Number of bytes of a image in file:   '+str(8*(self.numOfAtoms+self.numOfAtoms*pm.realFeatNum))+'\n')
                featInfo.write('AtomTypeList:\n')
                featInfo.write(str(self.atomTypeList)[1:-1])
                featInfo.write('All indices of good images:\n')
                featInfo.write(str(self.goodImageIndices)[1:-1])
            if featSaveForm=='C':
                with open(self.featFilePathDict['C'],'wb') as binFile:
                    for index in self.goodImageIndices:
                        aImage=self[int(index)]
                        '''
                        energy=cp.ascontiguousarray(aImage.energy)
                        energy=cp.asnumpy(energy)
                        binFile.write(energy)
                        '''
                        feat=aImage.calFeat()
                        binFile.write(feat.tobytes())
            elif featSaveForm=='F':
                with open(self.featFilePathDict['F'],'wb') as binFile:
                    infoArray=cp.array((self.goodImageNum,self.numOfAtoms,pm.realFeatNum),cp.int32)
                    binFile.write(infoArray.tobytes())
                    binFile.write(cp.array(self.atomTypeList,cp.int32).tobytes())
                    for index in self.goodImageIndices:
                        aImage=self[int(index)]
                        '''
                        energy=cp.asfortranarray(aImage.energy)
                        energy=cp.asnumpy(energy)
                        binFile.write(energy)
                        '''
                        binFile.write(aImage.energy.tobytes())
                        feat=aImage.calFeat()
                        binFile.write(feat.tobytes())

        self.featMaxNeighborNum[featSaveForm]=pm.maxNeighborNum
        with open(self.maxNeighborInfoPath,'w') as infoFile:
            infoFile.write(str(pm.rCut)+' '+str(self.maxNeighborNum)+'\n')
            line=str(self.featMaxNeighborNum['T'])+'  '+str(self.featMaxNeighborNum['C'])+'  '+str(self.featMaxNeighborNum['F'])+'  '+str(self.featMaxNeighborNum['DC'])+'  '+str(self.featMaxNeighborNum['DF'])+'\n'
            infoFile.write(line)

    def saveFeatAndDfeats(self,featSaveForm='C'):
        '''
        此方法计算该system的所有feat和dfeat，并以featSaveForm和dfeatSaveForm指定的方式存储
        
        featSaveForm取值的意义：
        'C'/'c':以二进制矩阵的形式先后存储每一个good image的energy向量和feat矩阵，矩阵按C语言的顺序存储
        'F'/'f':以二进制矩阵的形式先后存储每一个good image的energy向量和feat矩阵，矩阵按Fortran的顺序存储
        
        dfeatSaveForm取值的意义：
        'C'/'c':以二进制矩阵的形式先后存储每一个good image的force矩阵和dfeat四维数组，矩阵和数组按C语言的顺序存储
        'F'/'f':以二进制矩阵的形式先后存储每一个good image的force矩阵和dfeat四维数组，矩阵和数组按Fortran的顺序存储

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
        featSaveForm=featSaveForm.upper()
        dfeatSaveForm='D'+featSaveForm
        #self.calMaxNeighborNum()
        isFeatNew=not(self.featMaxNeighborNum[featSaveForm]==pm.maxNeighborNum and os.path.isfile(self.featFilePathDict[featSaveForm]) and os.stat(self.featFilePathDict[featSaveForm]).st_mtime>os.stat(self.sourcePath).st_mtime)
        isDfeatNew=not(self.featMaxNeighborNum[dfeatSaveForm]==pm.maxNeighborNum and os.path.isfile(self.featFilePathDict[dfeatSaveForm]) and os.stat(self.featFilePathDict[dfeatSaveForm]).st_mtime>os.stat(self.sourcePath).st_mtime)
        if (not isFeatNew) and (not isDfeatNew):
            return
        
        self.checkEtot()        
        self.checkCell()
        featSaveForm=featSaveForm.upper()
        dfeatSaveFomr=dfeatSaveForm.upper()
        with open(self.featInfoTxtPath,'w') as featInfo:
            featInfo.write('Number of  good images:   '+str(len(self.goodImageIndices))+'\n')
            featInfo.write('Number of atoms:    '+str(self.numOfAtoms)+'\n')
            featInfo.write('Number of real Feats:   '+str(pm.realFeatNum)+'\n')
            featInfo.write('Maxium of neighbor atoms:   '+str(pm.maxNeighborNum)+'\n')
            featInfo.write('Number of bytes of a image in file:   '+str(8*(self.numOfAtoms*3+self.numOfAtoms*(1+pm.maxNeighborNum)*pm.realFeatNum))+'\n')
            featInfo.write('AtomTypeList:\n')
            featInfo.write(str(self.atomTypeList)[1:-1])
            featInfo.write('All indices of good images:\n')
            featInfo.write(str(self.goodImageIndices)[1:-1])
        featFileOpenForm='rb'
        dfeatFileOpenForm='rb'
        if isFeatNew:
            featFileOpenForm='wb'
        if isDfeatNew:
            dfeatFileOpenForm='wb'
        if featSaveForm=='C':
            with open(self.featFilePathDict['C'],featFileOpenForm) as featBinFile,open(self.featFilePathDict['DC'],dfeatFileOpenForm) as dfeatBinFile:
                for index in self.goodImageIndices:
                    aImage=self[int(index)]
                    '''
                    energy=cp.ascontiguousarray(aImage.energy)
                    energy=cp.asnumpy(energy)
                    binFile.write(energy)
                    '''
                    if isFeatNew:
                        feat=aImage.calFeat()
                        featBinFile.write(feat.tobytes())
                    if isDfeatNew:
                        dfeat=aImage.calDfeat()
                        dfeatBinFile.write(dfeat.tobytes())                    
        elif featSaveForm=='F':
            with open(self.featFilePathDict['F'],featFileOpenForm) as featBinFile,open(self.featFilePathDict['DF'],dfeatFileOpenForm) as dfeatBinFile:
                if isFeatNew:
                    infoArray=cp.array((self.goodImageNum,self.numOfAtoms,pm.realFeatNum),cp.int32)
                    featBinFile.write(infoArray.tobytes())
                    featBinFile.write(cp.array(self.atomTypeList,cp.int32).tobytes())
                if isDfeatNew:
                    infoArray=cp.array((self.goodImageNum,self.numOfAtoms,pm.realFeatNum,pm.maxNeighborNum+1),cp.int32)
                    dfeatBinFile.write(infoArray.tobytes())
                    dfeatBinFile.write(cp.array(self.atomTypeList,cp.int32).tobytes())
                for index in self.goodImageIndices:
                    aImage=self[int(index)]
                    '''
                    energy=cp.ascontiguousarray(aImage.energy)
                    energy=cp.asnumpy(energy)
                    binFile.write(energy)
                    '''
                    if isFeatNew:
                        featBinFile.write(aImage.energy.tobytes())
                        feat=aImage.calFeat()
                        featBinFile.write(feat.tobytes())
                    if isDfeatNew:
                        dfeatBinFile.write(aImage.energy.tobytes())
                        force=aImage.force
                        dfeatBinFile.write(force.tobytes())
                        feat=aImage.calFeat()
                        dfeatBinFile.write(feat.tobytes())
                        dfeatBinFile.write((aImage.neighborNumOfAllAtoms+1).astype(cp.int32).tobytes())
                        neighborList=(aImage.neighborListOfAllAtoms+1).astype(cp.int32)
                        dfeatBinFile.write(neighborList.tobytes())
                        
                        dfeat=aImage.calDfeat().transpose(3,1,0,2)
                        dfeatBinFile.write(dfeat.tobytes()) 

        self.featMaxNeighborNum[featSaveForm]=pm.maxNeighborNum
        self.featMaxNeighborNum[dfeatSaveForm]=pm.maxNeighborNum
        with open(self.maxNeighborInfoPath,'w') as infoFile:
            infoFile.write(str(pm.rCut)+' '+str(self.maxNeighborNum)+'\n')
            line=str(self.featMaxNeighborNum['T'])+'  '+str(self.featMaxNeighborNum['C'])+'  '+str(self.featMaxNeighborNum['F'])+'  '+str(self.featMaxNeighborNum['DC'])+'  '+str(self.featMaxNeighborNum['DF'])+'\n'
            infoFile.write(line)
        
                    
                    
            

            
            
            
    
            

    def __getitem__(self,index):
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
        self.loadData()                  #此句只适宜在调试阶段使用,正式阶段不应该使用之，不过不影响结果
        self.checkCell()                 #此句只适宜在调试阶段使用,正式阶段不应该使用之，不过不影响结果
        if isinstance(index,int):
            return Image(self.atomTypeList,self.allCell[index],self.allPos[index],self.isOrthogonalCell,self.isCheckSupercell,\
                   self.atomTypeSet,self.atomCountAsType,self.atomCategoryDict,self.allForce[index],self.allEnergy[index],\
                   self.allVelocity[index],self.allEp[index],self.allDE[index],self.basicFuncVect,self,index)
        else:
            raise IndexError('Now System obj just support int index!')
        


    
if __name__=='__main__':
    input('Press Enter to quit test:')