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
import time
import numpy as np
import cupy as cp
import pandas as pd
#from ase.cell import Cell


class Image():
    '''
    建立一个image的class,只支持从一个System的实例用一个整数index建立一个image

    
    Parameters
    ---------------------
    fromSystem:                          System对象,源体系
    IndexInSystem:                       int对象,该图像在源体系中的序号



    Variable Attributes
    ---------------------    
    fromSystem:                       System对象,存储源体系的信息
    numOfAtoms:                       int对象,存储源体系,也是该图像的原子个数
    indexInSystem:                    int对象,存储了该图像在源体系中的序号
    atomTypeList:                     list对象,int,numOfAtoms长度,存储源体系,也是该图像中每个原子所属的原子种类
    atomTypeSet:                      tuple对象,int, 存储了源体系,也是该图像中所包含的所有原子种类的信息
    atomCountAsType:                  cp.array对象,int,长度和atomTypeSet一致,存储了源体系,也是该图像对应的每种原子的数目的信息
    isOrthogonalCell:                 bool对象,存储了源体系,也是该图像中cell是否是’不变且是三个方向基矢两两互相垂直’的信息
    isCheckSupercell:                 bool对象,存储了源体系,也是该图像中是否需要在后续计算近邻结构信息时考虑更大的超胞里有两个相同位置的原子在同一个原子近邻的情况的信息
    cupyCell:                         cp.array对象,float,3*3,存储整个体系中所有Image的cell的信息
    pos:                              cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的position的信息
    force:                            cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的force的信息
    velocity:                         cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的velocity的信息
    energy:                           cp.array对象,float,numOfAtoms, 存储该图像中所有原子的energy的信息
    ep:                               cp.array对象,float,1,存储该图像中Ep的信息
    dE:                               cp.array对象,float,1,存储该图像中dE的信息
    basicFunc:                        function对象,存储了源体系,也是该图像使用的basicFunc,默认是用cosBasciFunc
    basicDfunc:                       function对象,存储了源体系,也是该图像使用的basicDfunc,默认是用cosBasicDfunc
    structDataFilePath:               str对象,存储该图像中部分近邻结构信息的文件路径,为在self.fromSystem.structDataDir下的一个'.npy'文件,文件名为图像在体系中的序号
    featFilePath:                     str对象,存储该图像中feat信息的文件路径,为在self.fromSystem.featDir下的一个'.npy'文件,文件名为图像在体系中的序号
    dfeatFilePath:                    str对象,存储该图像中dfeat信息的文件路径,为在self.fromSystem.dfeatDir下的一个'.npy'文件,文件名为图像在体系中的序号



    Method Attributes
    ---------------------


    Internal Method Attributes
    ---------------------

      
    '''
    def __init__(self,
        atomTypeList,
        cell,
        pos,
        isOrthogonalCell=None,
        isCheckSupercell=None,
        atomTypeSet=None,
        atomCountAsType=None,        
        atomCategoryDict=None,
        force=None,
        energy=None,
        velocity=None,
        ep=None,
        dE=None,
        basicFuncVect=None,
        fromSystem=None,
        indexInSystem=None):
        '''
        初始化,从源体系中继承相应的一部分数据        

        确定存储部分近邻结构数据、feat数据、dfeat数据的三个文件的路径


        Determine Attributes Directly
        ---------------------
        fromSystem:                       System对象,存储源体系的信息
        numOfAtoms:                       int对象,存储源体系,也是该图像的原子个数
        indexInSystem:                    int对象,存储了该图像在源体系中的序号
        atomTypeList:                     list对象,int,numOfAtoms长度,存储源体系,也是该图像中每个原子所属的原子种类
        atomTypeSet:                      tuple对象,int, 存储了源体系,也是该图像中所包含的所有原子种类的信息
        atomCountAsType:                  cp.array对象,int,长度和atomTypeSet一致,存储了源体系,也是该图像对应的每种原子的数目的信息
        isOrthogonalCell:                 bool对象,存储了源体系,也是该图像中cell是否是’不变且是三个方向基矢两两互相垂直’的信息
        isCheckSupercell:                 bool对象,存储了源体系,也是该图像中是否需要在后续计算近邻结构信息时考虑更大的超胞里有两个相同位置的原子在同一个原子近邻的情况的信息
        cupyCell:                         cp.array对象,float,3*3,存储整个体系中所有Image的cell的信息
        pos:                              cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的position的信息
        force:                            cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的force的信息
        velocity:                         cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的velocity的信息
        energy:                           cp.array对象,float,numOfAtoms, 存储该图像中所有原子的energy的信息
        ep:                               cp.array对象,float,1,存储该图像中Ep的信息
        dE:                               cp.array对象,float,1,存储该图像中dE的信息
        basicFunc:                        function对象,存储了源体系,也是该图像使用的basicFunc,默认是用cosBasciFunc
        basicDfunc:                       function对象,存储了源体系,也是该图像使用的basicDfunc,默认是用cosBasicDfunc
        structDataFilePath:               str对象,存储该图像中部分近邻结构信息的文件路径,为在self.fromSystem.structDataDir下的一个'.npy'文件,文件名为图像在体系中的序号
        featFilePath:                     (废弃)str对象,存储该图像中feat信息的文件路径,为在self.fromSystem.featDir下的一个'.npy'文件,文件名为图像在体系中的序号
        dfeatFilePath:                    (废弃)str对象,存储该图像中dfeat信息的文件路径,为在self.fromSystem.dfeatDir下的一个'.npy'文件,文件名为图像在体系中的序号

        '''
        self.atomTypeList=atomTypeList
        self.numOfAtoms=len(self.atomTypeList)
        #self.atomCountAsType=atomCountAsType
        self.atomCategoryDict=atomCategoryDict
        self.cupyCell=cell
        self.pos=pos
        self.isOrthogonalCell=isOrthogonalCell
        self.isCheckSupercell=isCheckSupercell
        self.force=force
        self.energy=energy
        self.velocity=velocity
        self.ep=ep
        self.dE=dE
        
        self.basicFuncVect=basicFuncVect
        self.fromSystem=fromSystem
        self.indexInSystem=indexInSystem
        
        if (self.isCheckSupercell is None) or (self.isOrthogonalCell is None):
            self.checkCell()
        
        if atomTypeSet is None:
            self.atomTypeSet=tuple(set(self.atomTypeList))
        else:
            self.atomTypeSet=atomTypeSet
        
        if atomCountAsType is None:
            self.atomCountAsType=cp.array([self.atomTypeList.count(i) for i in self.atomTypeSet])
        else:
            self.atomCountAsType=atomCountAsType
            
        if atomCategoryDict is None:
            self.atomCategoryDict={}
            for atomType in pm.atomTypeSet:
                self.atomCategoryDict[atomType]=cp.where(cp.array(self.atomTypeList)==atomType)[0]
        
        if self.basicFuncVect is not None:
            self.basicFunc=basicFuncVect[0]
            self.basicDfunc=basicFuncVect[1]
        else:
            self.basicFunc=self.cosBasicFunc
            self.basicDfunc=self.cosBasicDfunc
        
        if self.fromSystem is not None:
            self.systemDir=self.fromSystem.dir
            if self.indexInSystem is not None:
                self.structDataFilePath=os.path.join(self.fromSystem.structDataDir,str(self.indexInSystem)+'.npy')
            else:
                self.structDataFilePath=os.path.join(self.fromSystem.structDataDir,str(time.asctime())+'.npy')
        else:
            self.systemDir=os.path.join(os.path.abspath(
                '../../../Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/44645da87a31bfb7db286f106ec8531b/Message/MessageTemp/15d56fb7ffbddcabcf66864c27481df4/File'), 'temp')
            self.structDataFilePath=os.path.join(self.systemDir,str(time.asctime())+'.npy')
        
        
        '''
        self.fromSystem=fromSystem
        self.numOfAtoms=fromSystem.numOfAtoms
        self.indexInSystem=indexInSystem
        self.atomTypeList=fromSystem.atomTypeList
        self.atomTypeSet=fromSystem.atomTypeSet
        self.atomCountAsType=fromSystem.atomCountAsType
        self.isOrthogonalCell=fromSystem.isOrthogonalCell
        self.isCheckSupercell=fromSystem.isCheckSupercell
        self.cupyCell=fromSystem.allCell[indexInSystem]
        self.pos=fromSystem.allPos[indexInSystem]
        self.force=fromSystem.allForce[indexInSystem]
        self.energy=fromSystem.allEnergy[indexInSystem]
        self.velocity=fromSystem.allVelocity[indexInSystem]
        self.ep=fromSystem.allEp[indexInSystem]
        self.dE=fromSystem.allDE[indexInSystem]

        self.basicFunc=fromSystem.basicFunc
        self.basicDfunc=fromSystem.basicDfunc
        
        self.structDataFilePath=os.path.join(fromSystem.structDataDir,str(indexInSystem)+'.npy')
        '''
        #self.featFilePath=os.path.join(fromSystem.featDir,str(indexInSystem)+'.npy')
        #self.dfeatFilePath=os.path.join(fromSystem.dfeatDir,str(indexInSystem)+'.npy')
       
    def checkCell(self):
        '''
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
        if self.isCheckSupercell is not None:
            return
        self.isOrthogonalCell=False
        self.isCheckSupercell=True
        cosAng=cp.zeros((3,))
        #sinAng=cp.zeros((3,))
        #areaVect=cp.zeros((3,))
        #self.heightVect=cp.zeros((3))
        if True:
            latticeConstants=cp.linalg.norm(self.cupyCell,axis=1)
            volume=cp.abs(cp.linalg.det(self.cupyCell))
            idx0=cp.arange(3)
            idxm=(idx0-1)%3
            idxp=(idx0+1)%3
            cosAng[idx0]=cp.sum(self.cupyCell[idxm]*self.cupyCell[idxp],axis=1)/(latticeConstants[idxm]*latticeConstants[idxp])
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
        

    def calDistanceVectArray(self,isShiftZeroPoint=False,shiftZeroPointVect=cp.array([0.0,0.0,0.0])):
        '''
        得到任意两个原子的位矢差的矩阵,维数为numOfAtoms*numOfAtoms*3


        Parameters
        ---------------------        
        isShiftZeroPoint:                 bool对象,决定是否有零点偏移矢量,默认为False,事实上目前不起作用
        shiftZeroPointVect:               cp.array对象,float, 1*3, 零点偏移矢量,事实上目前不起作用
        

        Returns
        ---------------------        
        None
        

        Determine Attributes Directly
        ---------------------
        distanceVectArray:                cp.array对象,float, numOfAtoms*numOfAtoms*3, 存储了直角坐标系,以A为单位的所有原子两两距离矢量,(i,j)=>Rj-Ri,i为中心原子序号,j为近邻原子序号
        '''
        pos=self.pos.copy()
        if isShiftZeroPoint:
            pos=cp.matmul(pos,self.cupyCell)
            pos+=shiftPointVect
            invCell=cp.linalg.inv(self.cupyCell)
            pos=cp.matmul(pos,invCell)

            
        pos%=1.0
        pos%=1.0
            
        self.distanceVectArray=pos[cp.newaxis,:,:]-pos[:,cp.newaxis,:]
        self.distanceVectArray[self.distanceVectArray>0.5]=self.distanceVectArray[self.distanceVectArray>0.5]-1.0
        self.distanceVectArray[self.distanceVectArray<-0.5]=self.distanceVectArray[self.distanceVectArray<-0.5]+1.0
        self.distanceVectArray=cp.matmul(self.distanceVectArray,self.cupyCell)



    def calCellNeighborStruct(self,shiftVect=cp.zeros((3,)),shiftOrder=0,rMin=pm.rMin,rCut=pm.rCut):
        '''
        此方法用于计算Image中self.distanceVectArray移动一个固定的矢量后的所有原子的近邻结构
        此函数会自动调用之前计算的原子近邻的结构数据并更新之


        Parameters
        ---------------------        
        shiftVect:                        cp.array对象,计算近邻结构时在self.distanceVectArray上附加的矢量,默认值为零矢量,只于self.isCheckSupercell为真时起作用
        shiftOrder:                       int对象,若shiftVect不为零矢量时,此矢量在self.shiftVects中的序号
        

        Returns
        ---------------------        
        None
        

        Operate Attributes Directly(Not Determine)
        ---------------------
        isNeighborMask:                    cp.array对象,bool, numOfAtoms*numOfAtoms*3, 存储了原子两两之间是否相邻的信息,此信息是屡次附加不同shiftVect之后累计的结果
        neighborNumOfAllAtoms:             cp.array对象,int, numOfAtoms, 存储了图像中各个原子近邻原子的数目的信息,屡次附加不同shiftVect之后累计的结果
        neighborIndexOfCenterAtoms:        cp.array对象,int, 一维,和self.neighborIndexOfNeighborAtoms一起存储了图像中所有近邻原子对的序数对,屡次附加不同shiftVect之后累计的结果
        neighborIndexOfNeighborAtoms:      cp.array对象,int, 一维,和self.neighborIndexOfCenterAtoms一起存储了图像中所有近邻原子对的序数对,屡次附加不同shiftVect之后累计的结果
        
        '''
        rMinSquare=rMin*rMin
        rCutSquare=rCut*rCut
        if not shiftOrder:
            distanceVectArray=self.distanceVectArray            
        else:
            distanceVectArray=self.distanceVectArray+shiftVect      #此处很奇怪，如果先赋值self.distanceVectArray，再用if判断是否用+=shiftVect则会出现奇怪的错误，值得研究      
        distanceSquareArray=cp.sum(distanceVectArray**2,axis=2)
        isNeighborMask=(distanceSquareArray>rMinSquare) * (distanceSquareArray<rCutSquare)
        distanceVectArray[~isNeighborMask]=0.0
        distanceSquareArray[~isNeighborMask]=0.0
        neighborNumOfAllAtoms=isNeighborMask.sum(axis=1)
        #print(self.indexInSystem,shiftOrder,int(neighborNumOfAllAtoms.max()),shiftVect)


        
        self.isNeighborMask=self.isNeighborMask+isNeighborMask
        
        self.neighborNumOfAllAtoms=self.neighborNumOfAllAtoms+neighborNumOfAllAtoms

        neighborIndexOfCenterAtoms,neighborIndexOfNeighborAtoms=cp.where(isNeighborMask)
        if shiftOrder:
            neighborIndexOfNeighborAtoms+=shiftOrder*self.numOfAtoms
        
        self.neighborIndexOfCenterAtoms=cp.concatenate((self.neighborIndexOfCenterAtoms,neighborIndexOfCenterAtoms))
        self.neighborIndexOfNeighborAtoms=cp.concatenate((self.neighborIndexOfNeighborAtoms,neighborIndexOfNeighborAtoms))

        #print(self.neighborIndexOfCenterAtoms.shape,self.neighborIndexOfNeighborAtoms.shape)



    def calAllNeighborStruct(self,isSave=False,isCheckFile=True,rMin=pm.rMin,rCut=pm.rCut):
        '''
        此方法用于计算整个Image的近邻结构数据,但不包括最后的近邻列表和近邻原子位矢差矩阵的数据
        之所以有此方法是为了更快地搜集可以确定最大近邻原子数的数据
        而后两者在搜集上述数据时可以不涉及,尤其近邻列表的矩阵尺寸就需要最大近邻原子数这个参数

        
        Parameters
        ---------------------        
        isSave:                           bool对象,决定计算部分近邻信息后是否存储到文件, 默认值为False
        isCheckFile:                      boll对象,决定是否检查存储部分近邻信息的文件是否已经存在并读取已有文件,默认值为True
        rMin:                             float对象, 允许的近邻最小距离,默认值为pm.rMin,
        rMax:                             float对象, 允许的近邻最大距离,默认值为pm.rMax
        

        Returns
        ---------------------        
        None
        

        Determine Attributes Directly
        ---------------------
        shiftVects:                        cp.array对象,float, 7*3, 0矢量,以及self.cupyCell的六个方向单位矢量,只有在self.isCheckSupercell为真时会计算
        isNeighborMask:                    cp.array对象,bool, numOfAtoms*numOfAtoms*3, 存储了原子两两之间是否相邻的信息
        neighborNumOfAllAtoms:             cp.array对象,int, numOfAtoms, 存储了图像中各个原子近邻原子的数目的信息
        neighborIndexOfCenterAtoms:        cp.array对象,int, 一维,和self.neighborIndexOfNeighborAtoms一起存储了图像中所有近邻原子对的序数对
        neighborIndexOfNeighborAtoms:      cp.array对象,int, 一维,和self.neighborIndexOfCenterAtoms一起存储了图像中所有近邻原子对的序数对


        By self.calDistanceVectArray
        ---------------------        
        distanceVectArray:                 cp.array对象,float, numOfAtoms*numOfAtoms*3, 存储了直角坐标系,以A为单位的所有原子两两距离矢量,(i,j)=>Rj-Ri,i为中心原子序号,j为近邻原子序号
        '''


        if isCheckFile and os.path.isfile(self.structDataFilePath):
            self.shiftVects,self.distanceVectArray,self.isNeighborMask,self.neighborNumOfAllAtoms,self.neighborIndexOfCenterAtoms,self.neighborIndexOfNeighborAtoms=np.load(self.structDataFilePath,None,True)
            return

        
        self.isNeighborMask=False
        self.neighborNumOfAllAtoms=0
        self.neighborIndexOfCenterAtoms=cp.array([],cp.int32)        #此二处必须加上dtype,否则默认dtype是cp.float64
        self.neighborIndexOfNeighborAtoms=cp.array([],cp.int32)      #
        
        self.calDistanceVectArray()
        self.shiftVects=None
        self.calCellNeighborStruct()

        
        
            

        if self.isCheckSupercell:
            self.shiftVects=cp.zeros((7,3))
            self.shiftVects[1]=-self.cupyCell[0]
            self.shiftVects[2]=self.cupyCell[0]
            self.shiftVects[3]=-self.cupyCell[1]
            self.shiftVects[4]=self.cupyCell[1]
            self.shiftVects[5]=-self.cupyCell[2]
            self.shiftVects[6]=self.cupyCell[2]
            if self.isOrthogonalCell:
                for i in range(3):
                    if cp.sum(self.cupyCell[i]**2)<4.0*rCut*rCut:
                        self.calCellNeighborStruct(-self.cupyCell[i],i*2+1)
                        self.calCellNeighborStruct(self.cupyCell[i],i*2+2)
            else:
                for i in range(1,7):
                    self.calCellNeighborStruct(self.shiftVects[i],i)

       
            

        if isSave:
            structData=np.array((self.shiftVects,self.distanceVectArray,self.isNeighborMask,self.neighborNumOfAllAtoms,self.neighborIndexOfCenterAtoms,self.neighborIndexOfNeighborAtoms))
            np.save(self.structDataFilePath,structData,True)


    


        '''

            

        
        '''

    def getMaxNeighborNum(self,isSaveStructData=False):
        '''
        计算一个Image的最大近邻原子数,不需要储存,返回之即可


        Parameters
        ---------------------        
        isSaveStructData:                 bool对象,决定计算部分近邻信息后是否存储到文件, 默认值为False
        

        Returns
        ---------------------        
        int(self.neighborNumOfAllAtoms.max()):  int对象,该图像中的最大近邻原子数
        
        

        By self.calAllNeighborStruct
        ---------------------
        shiftVects:                        cp.array对象,float, 7*3, 0矢量,以及self.cupyCell的六个方向单位矢量,只有在self.isCheckSupercell为真时会计算
        isNeighborMask:                    cp.array对象,bool, numOfAtoms*numOfAtoms*3, 存储了原子两两之间是否相邻的信息
        neighborNumOfAllAtoms:             cp.array对象,int, numOfAtoms, 存储了图像中各个原子近邻原子的数目的信息
        neighborIndexOfCenterAtoms:        cp.array对象,int, 一维,和self.neighborIndexOfNeighborAtoms一起存储了图像中所有近邻原子对的序数对
        neighborIndexOfNeighborAtoms:      cp.array对象,int, 一维,和self.neighborIndexOfCenterAtoms一起存储了图像中所有近邻原子对的序数对


        By self.calDistanceVectArray Indirectly
        ---------------------        
        distanceVectArray:                 cp.array对象,float, numOfAtoms*numOfAtoms*3, 存储直角坐标,以A为单位的所有原子两两距离矢量,(i,j)=>Rj-Ri,i为中心原子序号,j为近邻原子序号
        '''

        self.calAllNeighborStruct(isSaveStructData)
        return int(self.neighborNumOfAllAtoms.max())
        
        


    def calAllNeighborInfo(self,isCheckFile=True):
        '''
        此方法用于计算整个Image的所有数据,包括最后的近邻列表和近邻原子位矢差矩阵等数据
        因为要用到最大近邻原子数和工作的所有原子类型列表等数据
        所以此函数的调用必须在初步处理了工作的全局变量才能开始
        此函数应只作为为计算feat和dfeat用


        Parameters
        ---------------------        
        isCheckFile:                            bool对象,决定是否检查存储部分近邻信息的文件是否已经存在并读取已有文件,默认值为True
        

        Returns
        ---------------------        
        None


        Determine Attributes Directly
        ---------------------
        neighborListOfAllAtoms:                 cp.array对象,int, numOfAtoms*pm.maxNeighborNum, (i,j)=> i是中心原子序号, 若(i,j)元素不为0,则是近邻原子序号+1
        neighborDistanceVectArrayOfAllAtoms:    cp.array对象,float, numOfAtoms*pm.maxNeighborNum*3, 所有近邻原子对(i,j)=> Rj-Ri,i为中心原子序号,j为近邻原子序号
        maskDictOfAllAtomTypesInNeighborList:   dict对象,int=>numOfAtoms*pm.maxNeighborNum的cp.array,从pm.atomTypeSet中的每种原子种类映射到self.neighborListOfAllAtoms中近邻是否此种原子
        neighborDistanceArrayOfAllAtoms:        cp.array对象,float, numOfAtoms*pm.maxNeighborNum,所有近邻原子对(i,j)=> |Rj-Ri|,i为中心原子序号,j为近邻原子序号
        neighborUnitVectArrayOfAllAtoms:        cp.array对象,float, numOfAtoms*pm.maxNeighborNum*3,所有近邻原子对(i,j)=> (Rj-Ri)/|Rj-Ri|,i为中心原子序号,j为近邻原子序号
        abDistanceArray:                        cp.array对象,float, numOfAtoms*pm.maxNeighborNum*pm.maxNeighborNum, (i,j1,jb2)=>所有原子的近邻原子两两间的距离矢量       
        abUnitVectArray:                        cp.array对象,float, numOfAtoms*pm.maxNeighborNum*pm.maxNeighborNum*3, (i,j1,jb2)=>所有原子的近邻原子两两间的距离矢量的单位矩阵
        basic2bFeatArrayOfAllNeighborPairs:     cp.array对象,float, numOfAtoms*pm.maxNeighborNum*pm.mulNumOf2bFeat,self.basicFunc(self.neighborDistanceArrayOfAllAtoms,pm.mFV2b(缩写))
        basic3bFeatArrayOfAllNeighborPairs:     cp.array对象,float, numOfAtoms*pm.maxNeighborNum*pm.mulNumOf3bFeat,self.basicFunc(self.neighborDistanceArrayOfAllAtoms,pm.mFV3b(缩写))
        basic3bFeatArrayOfAllABpairs:           cp.array对象,float,numOfAtoms*pm.maxNeighborNum*pm.maxNeighborNum*pm.mulNumOf3bFeat,self.basicFunc(self.abDistanceArrayOfAllAtoms,pm.mFV3b)
        

        By self.calAllNeighborStruct
        ---------------------
        shiftVects:                             cp.array对象,float, 7*3, 0矢量,以及self.cupyCell的六个方向单位矢量,只有在self.isCheckSupercell为真时会计算
        isNeighborMask:                         cp.array对象,bool, numOfAtoms*numOfAtoms*3, 存储了原子两两之间是否相邻的信息
        neighborNumOfAllAtoms:                  cp.array对象,int, numOfAtoms, 存储了图像中各个原子近邻原子的数目的信息
        neighborIndexOfCenterAtoms:             cp.array对象,int, 一维,和self.neighborIndexOfNeighborAtoms一起存储了图像中所有近邻原子对的序数对
        neighborIndexOfNeighborAtoms:           cp.array对象,int, 一维,和self.neighborIndexOfCenterAtoms一起存储了图像中所有近邻原子对的序数对


        By self.calDistanceVectArray Indirectly
        ---------------------        
        distanceVectArray:                      cp.array对象,float, numOfAtoms*numOfAtoms*3, 存储直角坐标,以A为单位的所有原子两两距离矢量,(i,j)=>Rj-Ri,i为中心原子序号,j为近邻原子序号
        '''       


        if hasattr(self,'neighborDistanceVectArrayOfAllAtoms'):
            return

        
        self.calAllNeighborStruct(isSave=False,isCheckFile=isCheckFile)
        self.neighborAtomOrderInNeighborListOfCenterAtoms=cp.concatenate([cp.arange(int(self.neighborNumOfAllAtoms[index])) for index in range(self.numOfAtoms)])

        argsort=cp.argsort(self.neighborIndexOfCenterAtoms)
        self.neighborIndexOfCenterAtoms=self.neighborIndexOfCenterAtoms[argsort]
        self.neighborIndexOfNeighborAtoms=self.neighborIndexOfNeighborAtoms[argsort]


        
        self.neighborListOfAllAtoms=-cp.ones((self.numOfAtoms,1+pm.maxNeighborNum),dtype=cp.int)  #此处必得有dtype,否则默认是cp.float64类型. 而zeros_like或者ones_like函数的dtype默认和模板array一致
        self.neighborDistanceVectArrayOfAllAtoms=cp.zeros((self.numOfAtoms,pm.maxNeighborNum,3))
        self.maskDictOfAllAtomTypesInNeighborList={}


        #计算近邻原子列表self.neighborListOfAllAtoms以及近邻矢量矩阵self.neighborDistanceVectArrayOfAllAtoms
        if self.isCheckSupercell:     #此为在有考虑大超胞的情况下的计算
            self.neighborDistanceVectArrayOfAllAtoms[self.neighborIndexOfCenterAtoms,self.neighborAtomOrderInNeighborListOfCenterAtoms]=\
                    self.distanceVectArray[self.neighborIndexOfCenterAtoms,self.neighborIndexOfNeighborAtoms%self.numOfAtoms]+self.shiftVects[self.neighborIndexOfNeighborAtoms//self.numOfAtoms]
            for indexOfAtom in range(self.numOfAtoms):
                neighborList=self.neighborIndexOfNeighborAtoms[self.neighborIndexOfCenterAtoms==indexOfAtom]%self.numOfAtoms     #%self.numOfAtoms)  #此处是否要用到%self.numOfAtoms需要再议…………
                self.neighborListOfAllAtoms[indexOfAtom,1:1+neighborList.shape[0]]=neighborList                  #两种情况下是否要在第一列加上中心原子序数也需要再议

        else:                       #此为在没有考虑大超胞的情况下的计算
            self.neighborDistanceVectArrayOfAllAtoms[self.neighborIndexOfCenterAtoms,self.neighborAtomOrderInNeighborListOfCenterAtoms]=self.distanceVectArray[self.neighborIndexOfCenterAtoms,self.neighborIndexOfNeighborAtoms]
            for indexOfAtom in range(self.numOfAtoms):
                neighborList=self.neighborIndexOfNeighborAtoms[self.neighborIndexOfCenterAtoms==indexOfAtom]
                self.neighborListOfAllAtoms[indexOfAtom,1:1+neighborList.shape[0]]=neighborList        
        self.neighborListOfAllAtoms[:,0]=cp.arange(self.numOfAtoms)
            


        '''
        #此为初始版本，其效率令人难以忍受，需要30+秒的时间，需要尝试提高效率
        #此段为计算每个中心原子在其近邻原子的近邻列表中的序号self.centerAtomOrderInNeighborListOfNeighborAtoms:
        self.centerAtomOrderInNeighborListOfNeighborAtoms=-cp.ones_like(self.neighborListOfAllAtoms)
        for i in range(self.numOfAtoms):
            for j in range(pm.maxNeighborNum):
                if self.neighborListOfAllAtoms[i,j]>-1:
                    k=self.neighborListOfAllAtoms[i,j]
                    k_line=k%self.numOfAtoms
                    k_shiftOrder=k//self.numOfAtoms
                    k_invShiftOrder=k_shiftOrder-min(k_shiftOrder,1)*(-1)**(k_shiftOrder%2)
                    k_inv=k_invShiftOrder*self.numOfAtoms+i
                    invOrder=cp.where(self.neighborListOfAllAtoms[k_line]==k_inv)[0][0]
                    self.centerAtomOrderInNeighborListOfNeighborAtoms[i,j]=invOrder
                    pass

                    
        #上一步计算出来的结果实际上是个矩阵，并不适合使用，下面实际将之变成可以供后续使用的一维数组，和self.neighborIndexOfCenterAtoms，self.neighborIndexOfNeighborAtoms一样长度
        #实际上，为了便于使用,在这一步也需要将self.neighborIndexOfNeighborAtoms对self.numOfAtoms取余
        #self.neighborIndexOfNeighborAtoms=self.neighborIndexOfNeighborAtoms%self.numOfAtoms      #后续的方法仍然要用到这个
        self.centerAtomOrderInNeighborListOfNeighborAtoms=self.centerAtomOrderInNeighborListOfNeighborAtoms[self.centerAtomOrderInNeighborListOfNeighborAtoms>-1]
        '''


        '''
        #第二版的方法仍然让人难以忍受其速度，所以寻找第三版完全不需要用到循环，全程矩阵操作的办法！
        #完全采用另外一种方法计算self.centerAtomOrderInNeighborListOfNeighborAtoms
        #这一次直接将之设置成一维的，直接利用cp.where进行计算
        #self.centerAtomOrderInNeighborListOfNeighborAtomsSlow=self.centerAtomOrderInNeighborListOfNeighborAtoms.copy()
        self.centerAtomOrderInNeighborListOfNeighborAtoms=-cp.ones_like(self.neighborIndexOfNeighborAtoms)
        k_shiftOrder=self.neighborIndexOfNeighborAtoms//self.numOfAtoms
        k_invShiftOrder=k_shiftOrder-cp.where(k_shiftOrder<1,k_shiftOrder,1)*(-1)**(k_shiftOrder%2)
        k_inv=k_invShiftOrder*self.numOfAtoms+self.neighborIndexOfCenterAtoms
        neighborIndexOfNeighborAtomsMod=self.neighborIndexOfNeighborAtoms%self.numOfAtoms
        for i in range(len(self.centerAtomOrderInNeighborListOfNeighborAtoms)):
            self.centerAtomOrderInNeighborListOfNeighborAtoms[i]=cp.where(self.neighborListOfAllAtoms[neighborIndexOfNeighborAtomsMod[i]]==k_inv[i])[0][0]
        #cp.where((self.neighborIndexOfCenterAtoms==self.neighborIndexOfNeighborAtoms[i]%self.numOfAtoms)*(self.neighborIndexOfNeighborAtoms==k_inv[i]))[0][0]
        #int(np.argwhere((cp.asnumpy(self.neighborIndexOfCenterAtoms)==int(self.neighborIndexOfNeighborAtoms[i]%self.numOfAtoms))*(cp.asnumpy(self.neighborIndexOfNeighborAtoms)==int(k_inv[i]))))            

        #self.centerAtomOrderInNeighborListOfNeighborAtoms=self.neighborAtomOrderInNeighborListOfCenterAtoms[self.centerAtomOrderInNeighborListOfNeighborAtoms]
        self.neighborIndexOfNeighborAtoms=neighborIndexOfNeighborAtomsMod
        '''

        neighborIndexOfNeighborAtomsMod=self.neighborIndexOfNeighborAtoms%self.numOfAtoms
        neighborIndexArgsort=((self.neighborIndexOfCenterAtoms<<16)+self.neighborIndexOfNeighborAtoms).argsort()        
        invNeighborIndexOfNeighborAtoms=self.neighborIndexOfNeighborAtoms//self.numOfAtoms
        invNeighborIndexOfNeighborAtoms==invNeighborIndexOfNeighborAtoms-cp.where(invNeighborIndexOfNeighborAtoms<1,invNeighborIndexOfNeighborAtoms,1)*(-1)**(invNeighborIndexOfNeighborAtoms%2)
        invNeighborIndexOfNeighborAtoms=invNeighborIndexOfNeighborAtoms*self.numOfAtoms+self.neighborIndexOfCenterAtoms        
        invNeighborIndexInvArgsort=((neighborIndexOfNeighborAtomsMod<<16)+invNeighborIndexOfNeighborAtoms).argsort().argsort()
        self.centerAtomOrderInNeighborListOfNeighborAtoms=self.neighborAtomOrderInNeighborListOfCenterAtoms[neighborIndexArgsort][invNeighborIndexInvArgsort]
        self.neighborIndexOfNeighborAtoms=neighborIndexOfNeighborAtomsMod
            

        '''
        for i in range(self.numOfAtoms):
            mask=self.neighborListOfAllAtoms[i]>-1
            k=self.neighborListOfAllAtoms[i][mask]
            k_line=k%self.numOfAtoms
            k_shiftOrder=k//self.numOfAtoms
            k_invShiftOrder=k_shiftOrder-cp.where(k_shiftOrder<1,k_shiftOrder,1)*(-1)**k_shiftOrder
            k_inv=k_invShiftOrder*self.numOfAtoms+i
            invOrder=cp.where(self.neighborListOfAllAtoms[k_line]==k_inv)[0][0]
            self.centerAtomOrderInNeighborListOfNeighborAtoms[i][mask]=invOrder
        '''

                

        #计算每种原子的mask,以一个len(pm.atomTypeSet) or pm.atomTypeNum*self.numOfAtoms*(1+pm.maxNeighborNum)的mask矩阵来将存储所有的结果,可命名为self.allTypeNeighborMaskArray
        #然后原来的self.maskDictOfAllAtomTypesInNeighborList可以是指向这个Array部分结果的矩阵
        self.allTypeNeighborMaskArray=cp.zeros((pm.atomTypeNum,self.numOfAtoms,1+pm.maxNeighborNum),dtype=cp.bool)
        self.allTypeNeighborMaskArray[:,:,0]=True
        atomTypeListArray=cp.array(self.atomTypeList)
        for i in range(pm.atomTypeNum):
            neighborAtomType=pm.atomTypeSet[i]
            self.allTypeNeighborMaskArray[i][self.neighborIndexOfCenterAtoms,1+self.neighborAtomOrderInNeighborListOfCenterAtoms]=(atomTypeListArray[self.neighborIndexOfNeighborAtoms]==neighborAtomType)
            self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType]=self.allTypeNeighborMaskArray[i,:,1:]


        '''
        #此为原来计算self.maskDictOfAllAtomTypesInNeighobrList的方法，由于在后续计算中不适用,被废除
        for neighborAtomType in pm.atomTypeSet:
            if hasattr(cp,'isin'):
                self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType]=(cp.asarray((self.neighborListOfAllAtoms>-1)*(cp.isin(self.neighborListOfAllAtoms%self.numOfAtoms,self.fromSystem.atomCategoryDict[neighborAtomType]))))
            else:
                self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType]=(cp.asarray((self.neighborListOfAllAtoms>-1)*\
                            cp.asarray(np.isin(cp.asnumpy(self.neighborListOfAllAtoms%self.numOfAtoms),cp.asnumpy(self.fromSystem.atomCategoryDict[neighborAtomType])))))
        '''
        
        
        

            
            


        self.neighborDistanceArrayOfAllAtoms=cp.sqrt(cp.sum(self.neighborDistanceVectArrayOfAllAtoms**2,axis=2))
        self.neighborUnitVectArrayOfAllAtoms=cp.zeros_like(self.neighborDistanceVectArrayOfAllAtoms)
        mask=(self.neighborDistanceArrayOfAllAtoms>0)
        self.neighborUnitVectArrayOfAllAtoms[mask]=self.neighborDistanceVectArrayOfAllAtoms[mask]/cp.expand_dims(self.neighborDistanceArrayOfAllAtoms[mask],-1)   #不同维数矩阵的自动broad_cast计算似乎要求维数少的那一方的所有维数作为最后几维

        

        abDistanceVectArray=self.neighborDistanceVectArrayOfAllAtoms[:,:,cp.newaxis,:]-self.neighborDistanceVectArrayOfAllAtoms[:,cp.newaxis,:,:] #此处究竟哪个在前哪个在后合适上需要研究

        mask=self.neighborDistanceArrayOfAllAtoms==0
        abDistanceVectArray[mask]=0
        abDistanceVectArray.transpose(0,2,1,3)[mask]=0
        self.abDistanceArray=cp.sqrt(cp.sum(abDistanceVectArray**2,axis=3))
        
        #mask=(self.abDistanceArray<pm.rMin)+(self.abDistanceArray>pm.rCut)
        #abDistanceVectArray[mask]=0
        #self.abDistanceArray[mask]=0

        self.abUnitVectArray=cp.zeros((self.numOfAtoms,pm.maxNeighborNum,pm.maxNeighborNum,3))
        mask=self.abDistanceArray>0
        self.abUnitVectArray[mask]=abDistanceVectArray[mask]/cp.expand_dims(self.abDistanceArray[mask],-1)


        self.basic2bFeatArrayOfAllNeighborPairs=self.basicFunc(self.neighborDistanceArrayOfAllAtoms,pm.mulFactorVectOf2bFeat)#此处对整个近邻距离矩阵用basicFunc计算一个结果储存,是2bFeat相关
        self.basic3bFeatArrayOfAllNeighborPairs=self.basicFunc(self.neighborDistanceArrayOfAllAtoms,pm.mulFactorVectOf3bFeat)#此处对整个近邻距离矩阵用basicFunc计算一个结果储存
        self.basic3bFeatArrayOfAllABpairs=self.basicFunc(self.abDistanceArray,pm.mulFactorVectOf3bFeat,pm.rCut2)  #此处对整个的三体的ab距离矩阵用basicFunc计算一个结果储存,是3bFeat相关,2b和3b不一样
        

        basic2bDfeatArrayOfAllNeighborPairs=self.basicDfunc(self.neighborDistanceArrayOfAllAtoms,pm.mulFactorVectOf2bFeat)
        self.allNeighborPairs2bDfeatArray=basic2bDfeatArrayOfAllNeighborPairs[:,:,:,cp.newaxis]*self.neighborUnitVectArrayOfAllAtoms[:,:,cp.newaxis,:]
        #self.allNeighborPairs2bDfeatArray=cp.einsum('ija,ijr->ijar',basic2bDfeatArrayOfAllNeighborPairs,self.neighborUnitVectArrayOfAllAtoms)
        basic3bDfeatArrayOfAllNeighborPairs=self.basicDfunc(self.neighborDistanceArrayOfAllAtoms,pm.mulFactorVectOf3bFeat)
        self.allNeighborPairs3bDfeatArray=basic3bDfeatArrayOfAllNeighborPairs[:,:,:,cp.newaxis]*self.neighborUnitVectArrayOfAllAtoms[:,:,cp.newaxis,:]
        #self.allNeighborPairs3bDfeatArray=cp.einsum('ija,ijr->ijar',basic3bDfeatArrayOfAllNeighborPairs,self.neighborUnitVectArrayOfAllAtoms)
        basic3bDfeatArrayOfAllABpairs=self.basicDfunc(self.abDistanceArray,pm.mulFactorVectOf3bFeat,pm.rCut2)
        self.allABpairs3bDfeatArray=basic3bDfeatArrayOfAllABpairs[:,:,:,:,cp.newaxis]*self.abUnitVectArray[:,:,:,cp.newaxis,:]
        #self.allABpairs3bDfeatArray=cp.einsum('ijka,ijkr->ijkar',basic3bDfeatArrayOfAllABpairs,self.abUnitVectArray)
        
       


    @staticmethod
    def cosBasicFunc(distanceArray,mulFactorVect,rCut=pm.rCut,rMin=pm.rMin):
        '''
        此方法为以cos为基础的basic函数
        先令：
        x=wR*x+bR        (wR=2/(pm.rCut-pm.rMin),bR=-(pm.rCut+pm.rMin)/(pm.rCut-pm.rMin))
        h=1-mFV.max()
        然后有
        psai(x)=     0.5*cos((x-mFV[alpha])/h*pi)+0.5     for abs(x-mFV[alpha])<h
                                0                         for else
        对不同alpha有不同值,从而在输入distanceArray扩展出一个len(mFV)的维度然后返回
        
        mFV:mulFactorVect
        w:2.0/(rCut-rMin)
        b:-(rCut+rMin)/(rCut-rMin)
        bR:pm.biasOfDistanceSaclar


        Parameters
        ---------------------        
        distanceArray:                   cp.array对象,需要计算其basicFunc值的距离矩阵
        

        Returns
        ---------------------        
        expandedArray:                   cp.array对象,shape为(shape(distanceArray),len(mulFactorVect))
        
        '''
        w=2.0/(rCut-rMin)
        b=-(rCut+rMin)/(rCut-rMin)
        inDistanceArray=w*distanceArray+b        
        mask=(inDistanceArray>b)
        h=1.0-float(mulFactorVect.max())
        inDistanceArray=cp.expand_dims(inDistanceArray,-1)-mulFactorVect
        expandedArray=cp.zeros_like(inDistanceArray)
        mask=(cp.abs(inDistanceArray)<h)*cp.expand_dims(mask,-1)
        expandedArray[mask]=0.5*cp.cos(inDistanceArray[mask]/h*cp.pi)+0.5
        return expandedArray




    @staticmethod
    def cosBasicDfunc(distanceArray,mulFactorVect,rCut=pm.rCut,rMin=pm.rMin):
        '''
        此方法为以cos为基础的basic函数的导数
        先令：
        x=wR*x+bR        (wR=2/(pm.rCut-pm.rMin),bR=-(pm.rCut+pm.rMin)/(pm.rCut-pm.rMin))
        h=1-mFV.max()
        然后有
        dpsai(x)=   -0.5*pi/h*sin((x-mFV[alpha])/h*pi)         for abs(x-mFV[alpha])<h
                                  0                            for else
        对不同alpha有不同值,从而在输入distanceArray扩展出一个len(mFV)的维度然后返回
        mFV:mulFactorVect
        wR:pm.weightOfDistanceScaler
        bR:pm.biasOfDistanceSaclar


        Parameters
        ---------------------        
        distanceArray:                   cp.array对象,需要计算其basicDfunc值的距离矩阵
        

        Returns
        ---------------------        
        expandedArray:                   cp.array对象,shape为(shape(distanceArray),len(mulFactorVect))
        '''
        w=2.0/(rCut-rMin)
        b=-(rCut+rMin)/(rCut-rMin)
        inDistanceArray=w*distanceArray+b
        mask=(inDistanceArray>b)
        h=1.0-float(mulFactorVect.max())
        inDistanceArray=cp.expand_dims(inDistanceArray,-1)-mulFactorVect
        expandedArray=cp.zeros_like(inDistanceArray)
        mask=(cp.abs(inDistanceArray)<h)*cp.expand_dims(mask,-1)
        expandedArray[mask]=-0.5*cp.pi*w/h*cp.sin(inDistanceArray[mask]/h*cp.pi)
        return expandedArray     
        



    def getFeat(self,isSaveIfNone=True):
        '''
        检查是否存在feat文件,否则计算feat,决定是否存储并返回之


        Parameters
        ---------------------        
        isSaveIfNone:                           bool对象,决定是否在计算完feat后存储到文件中，默认值为True
        

        Returns
        ---------------------        
        feat:                                   cp.array对象, float, numOfAtoms*pm.realFeatNum，所有原子的feat信息


        Determine Attributes InDirectly
        ---------------------

        By self.calAllNeighborInfo：

        All same as in self.calAllNeighborInfo
        '''
        if os.path.isfile(self.featFilePath):
            pass
        else:
            return self.calFeat(self,isSaveIfNone)




    def calFeat(self,isSave=True):
        '''
        计算所有原子的feat

        Parameters
        ---------------------        
        isSave:                                 bool对象,决定是否在计算完feat后存储到文件中，默认值为True
        

        Returns
        ---------------------        
        feat:                                   cp.array对象, float, numOfAtoms*pm.realFeatNum，所有原子的feat信息


        Determine Attributes Indirectly
        ---------------------

        By self.calAllNeighborInfo：

        All same as in self.calAllNeighborInfo
        '''
        self.calAllNeighborInfo()
        feat=cp.zeros((self.numOfAtoms,pm.allFeatNum))
        for featLabel in pm.featLabelList:
            if len(featLabel)==1:
                self.calSub2bFeat(feat,featLabel)                    
            elif len(featLabel)==2:
                self.calSub3bFeat(feat,featLabel)


        feat=np.delete(cp.asnumpy(feat),cp.asnumpy(pm.dupFeatIndexVect),axis=1)            
        return feat
            


        


    def calSub2bFeat(self,feat,featLabel):
        '''
        计算所有原子与特定种类近邻原子相关的feat,原子种类信息存储在featLabel中(featLabel[0])


        Parameters
        ---------------------        
        feat:                                   cp.array对象, float, numOfAtoms*pm.realFeatNum,所有原子的feat信息,正在计算的过程中
        featLabel:                              tuple对象, int, 长度为1, 为近邻原子的种类
        

        Returns
        ---------------------        
        None


        Operate Attributes Directly(Nont determine)
        ---------------------

        feat:                                   cp.array对象, float, numOfAtoms*pm.realFeatNum，所有原子的feat信息
        '''        
        #indexVectOfAllNeighborAtoms=self.fromSystem.atomCategoryDict[neighborAtomType]
        neighborAtomType=featLabel[0]
        if not len(self.atomCategoryDict[neighborAtomType]):
            return
        #print(self.basic2bFeatArrayOfAllNeighborPairs.shape)
        #orderOfNeighborAtomType=pm.atomTypeSet.index(neighborAtomType)
        if pm.isSumFor2bFeat:
            #'''#*+sum算法
            feat[:,pm.featSegmentEndpointSliceDict[(neighborAtomType,)]]=\
                cp.sum(self.basic2bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType][:,:,cp.newaxis],axis=1)
            #'''#*+sum算法
        else:
            #'''#*matmul算法
            feat[:,pm.featSegmentEndpointSliceDict[(neighborAtomType,)]]=\
                           cp.matmul(self.basic2bFeatArrayOfAllNeighborPairs.transpose(0,2,1),cp.expand_dims(self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType],-1)).squeeze(-1)
            #'''#*matmul算法
        #feat[:,pm.featSegmentEndpointSliceDict[(neighborAtomType,)]]=\
                          #cp.einsum('ijk,ij->ik',self.basic2bFeatArrayOfAllNeighborPairs,self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType])
        #feat[:,pm.featSegmentEndpointSliceDict[(neighborAtomType,)]]=\
             #cp.sum(self.basic2bFeatArrayOfAllNeighborPairs*cp.expand_dims(self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType],-1),axis=1)

   



    def calSub3bFeat(self,feat,featLabel):
        '''
        计算对特定种类的A B原子类型的组合(featLabel)计算其3bfeat,原子种类A=featLabel[0],原子种类B=featLabel[1]
        

        Parameters
        ---------------------        
        feat:                                   cp.array对象, float, numOfAtoms*pm.realFeatNum,所有原子的feat信息,正在计算的过程中
        featLabel:                              tuple对象, int, 长度为2, 为两种近邻原子的种类
        

        Returns
        ---------------------        
        None


        Operate Attributes Directly(Nont determine)
        ---------------------

        feat:                                   cp.array对象, float, numOfAtoms*pm.realFeatNum，所有原子的feat信息
        '''
        aNeighborAtomType=featLabel[0]
        bNeighborAtomType=featLabel[1]
        if (not len(self.atomCategoryDict[aNeighborAtomType])) or (not len(self.atomCategoryDict[bNeighborAtomType])):
            return
        #orderOfANeighborAtomType=pm.atomTypeSet.index(aNeighborAtomType)
        #orderOfBNeighborAtomType=pm.atomTypeSet.index(bNeighborAtomType)
        #orderOfFeatLabel=pm.featLabelList.index(featLabel)

        
        #aSub3bFeat=self.basic3bFeatArrayOfAllNeighborPairs*cp.expand_dims(self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType],-1)
        #bSub3bFeat=self.basic3bFeatArrayOfAllNeighborPairs*cp.expand_dims(self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType],-1)

        aSub3bFeat=self.basic3bFeatArrayOfAllNeighborPairs*cp.expand_dims(self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType],-1)
        bSub3bFeat=self.basic3bFeatArrayOfAllNeighborPairs*cp.expand_dims(self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType],-1)
        

        #'''#预计的新算法        
        if pm.isSumFor3bFeat:
            #*+sum算法
            aSub3bFeat=cp.sum(aSub3bFeat[:,:,cp.newaxis,:,cp.newaxis]*self.basic3bFeatArrayOfAllABpairs[:,:,:,cp.newaxis,:],axis=1)
            feat[:,pm.featSegmentEndpointSliceDict[featLabel]]=\
                       cp.sum(aSub3bFeat[:,:,:,cp.newaxis,:]*bSub3bFeat[:,:,cp.newaxis,:,cp.newaxis],axis=1).reshape(self.numOfAtoms,-1)
            #*+sum算法
        else:
            #matmul算法
            #matmul算法有个BUG，当pm.mulNumOf3bFeat==1时，会出现结果错误，需要找到原因，@方法同样
            #此错误似乎与这个问题无关，而是cupy的ufunc cp.matmul的broadcast的问题，具体情况暂时不明，以后可以测试
            #经过检验，发现此属于cupy的Bug，我是无法解决的……            
            if pm.mulNumOf3bFeat!=1:
                aSub3bFeat=cp.matmul(aSub3bFeat.transpose(0,2,1)[:,cp.newaxis,:,:],self.basic3bFeatArrayOfAllABpairs.transpose(0,3,1,2))
                feat[:,pm.featSegmentEndpointSliceDict[featLabel]]=\
                       cp.matmul(aSub3bFeat,bSub3bFeat[:,cp.newaxis,:,:]).transpose(0,2,3,1).reshape(self.numOfAtoms,-1)
            else:
                aSub3bFeat=cp.matmul(aSub3bFeat.transpose(0,2,1),self.basic3bFeatArrayOfAllABpairs.squeeze(-1))
                feat[:,pm.featSegmentEndpointSliceDict[featLabel]]=\
                       cp.matmul(aSub3bFeat,bSub3bFeat).reshape(self.numOfAtoms,-1)
            #matmul算法
        #'''#预计的新算法
            
        
        

        
        '''#einsum算法'ijl,ijkm,ikn->ilnm'
        feat[:,pm.featSegmentEndpointSliceDict[featLabel]]=\
                    cp.einsum('ima,imnc,inb->iabc',aSub3bFeat,self.basic3bFeatArrayOfAllABpairs,bSub3bFeat).reshape(self.numOfAtoms,-1)
        '''#einsum算法
        
        
    


    
    def getDfeat(self,isSaveIfNone=True):
        '''
        检查是否存在dfeat文件,否则计算dfeat,决定是否存储并返回之
        '''
        pass




    def calDfeat(self,isSave=True):
        '''
        计算所有原子的dfeat
        '''
        
        self.calAllNeighborInfo()
        dfeat=cp.zeros((self.numOfAtoms,pm.maxNeighborNum+1,pm.allFeatNum,3))

        for featLabel in pm.featLabelList:
            if len(featLabel)==1:
                self.calSub2bDfeat(dfeat,featLabel)
            elif len(featLabel)==2:
                self.calSub3bDfeat(dfeat,featLabel)

        dfeat=np.delete(cp.asnumpy(dfeat),cp.asnumpy(pm.dupFeatIndexVect),axis=2)
        return -dfeat




    def calSub2bDfeat(self,dfeat,featLabel):
        '''
        计算所有原子与特定种类近邻原子相关的dfeat,原子种类信息存储在featLabel中(featLabel[0])
        '''
        neighborAtomType=featLabel[0]
        if not len(self.atomCategoryDict[neighborAtomType]):
            return
        atomTypeSlice=pm.featSegmentEndpointSliceDict[featLabel]
        categoryList=self.atomCategoryDict[neighborAtomType]
        
        if pm.isSumFor2bDfeat:
            dfeat[:,0,atomTypeSlice,:]=cp.sum(self.allNeighborPairs2bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType][:,:,cp.newaxis,cp.newaxis],axis=1)
        else:
            dfeat[:,0,atomTypeSlice,:]=cp.matmul(self.allNeighborPairs2bDfeatArray.transpose(0,2,3,1),self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType][:,cp.newaxis,:,cp.newaxis]).squeeze(-1)
        #dfeat[:,0,atomTypeSlice,:]=cp.einsum('ijkl,ij->ikl',self.allNeighborPairs2bDfeatArray,self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType])
        #dfeat[:,0,atomTypeSlice,:]=cp.sum(self.allNeighborPairs2bDfeatArray*cp.expand_dims(cp.expand_dims(self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType],-1),-1),1)

        dfeat[categoryList,1:,atomTypeSlice,:]+=self.allNeighborPairs2bDfeatArray[categoryList]#,:,:,:]


    def calSub3bDfeat(self,dfeat,featLabel):
        '''
        计算所有原子与特定种类近邻原子相关的dfeat,原子种类信息存储在featLabel中(featLabel[0])
        '''
        aNeighborAtomType=featLabel[0]
        bNeighborAtomType=featLabel[1]
        if (not len(self.atomCategoryDict[aNeighborAtomType])) or (not len(self.atomCategoryDict[bNeighborAtomType])):
            return
        categoryListA=self.atomCategoryDict[aNeighborAtomType]
        categoryListB=self.atomCategoryDict[bNeighborAtomType]
        atomTypeSliceAB=pm.featSegmentEndpointSliceDict[featLabel]



        #intermidArray1=self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis,cp.newaxis]
        #intermidArray2=self.self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]

        '''#三体feat较多时的算法，部分1 
        intermidArray=(self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]).transpose(0,2,1)[:,cp.newaxis,:,:]@self.basic3bFeatArrayOfAllABpairs
        dfeat[:,0,atomTypeSliceAB,:]+=((self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis,cp.newaxis]).transpose(0,2,3,1)[:,:,cp.newaxis,:,:]@\
                                       intermidArray.transpose(0,2,1,3)[:,cp.newaxis,:,:,:]).transpose(0,1,2,4,3).reshape(self.numOfAtoms,-1,3)
        
        intermidArray=(self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis,cp.newaxis]).transpose(0,2,3,1)[:,cp.newaxis,:,:,:]@\
                       self.basic3bFeatArrayOfAllABpairs[:,:,cp.newaxis,:,:]
        dfeat[:,0,atomTypeSliceAB,:]+=((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]).transpose(0,2,1)[:,cp.newaxis,cp.newaxis,:,:]@\
                                       intermidArray.transpose(0,2,4,1,3)).transpose(0,3,1,2,4).reshape(self.numOfAtoms,-1,3)
        '''#

        #在测试中发现，很多情况下乘法+sum确实会比matmul或者dot或者@更快，所以需要重新检查何时用何种策略
        if pm.isSumFor3bDfeat:

            
            #首先是计算中心原子feat对自己坐标的梯度，有两项
            #这是第一项
            intermidArray=self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]
            intermidArray=cp.sum(intermidArray[:,cp.newaxis,:,:,cp.newaxis]*self.basic3bFeatArrayOfAllABpairs[:,:,:,cp.newaxis,:],axis=2)
            intermidArray2=self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis,cp.newaxis]
            dfeat[:,0,atomTypeSliceAB,:]+=cp.sum(intermidArray2[:,:,:,cp.newaxis,cp.newaxis,:]*intermidArray[:,:,cp.newaxis,:,:,cp.newaxis],axis=1).reshape(self.numOfAtoms,-1,3)

            #这是中心原子feat对自己坐标的梯度的第二项
            intermidArray=self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis,cp.newaxis]
            intermidArray=cp.sum(intermidArray[:,cp.newaxis,:,:,cp.newaxis,:]*self.basic3bFeatArrayOfAllABpairs[:,:,:,cp.newaxis,:,cp.newaxis],axis=2)
            intermidArray2=self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]
            dfeat[:,0,atomTypeSliceAB,:]+=cp.sum(intermidArray2[:,:,:,cp.newaxis,cp.newaxis,cp.newaxis]*intermidArray[:,:,cp.newaxis,:,:,:],axis=1).reshape(self.numOfAtoms,-1,3)


            #以下是计算中心原子的feat，对周围近邻原子的坐标的梯度，此思路是先计算，再转换成近邻原子对中心原子坐标的梯度
            #首先是初始化
            subDfeat=cp.zeros((self.numOfAtoms,pm.maxNeighborNum,pm.mulNumOf3bFeat**3,3))

            #这里的第一部分是考虑，近邻原子属于featLabel (A,B)中的A类原子时的梯度，同样会有两项
            #这里是第一项
            intermidArray=self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]
            intermidArray=cp.sum(intermidArray[:,:,cp.newaxis,:,cp.newaxis]*self.basic3bFeatArrayOfAllABpairs[:,:,:,cp.newaxis,:],axis=1)
            intermidArray2=-self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis,cp.newaxis]
            subDfeat+=(intermidArray2[:,:,:,cp.newaxis,cp.newaxis,:]*intermidArray[:,:,cp.newaxis,:,:,cp.newaxis]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)

            #这里是近邻原子属于featLabel (A,B)中的A类原子时的梯度的第二项
            intermidArray=self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]
            intermidArray=cp.sum(intermidArray[:,:,cp.newaxis,:,cp.newaxis,cp.newaxis]*self.allABpairs3bDfeatArray[:,:,:,cp.newaxis,:,:],axis=1)
            intermidArray2=self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]
            subDfeat+=(intermidArray2[:,:,:,cp.newaxis,cp.newaxis,cp.newaxis]*intermidArray[:,:,cp.newaxis,:,:,:]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)

            #这里是近邻原子属于featLabel (A,B)中的B类原子时的梯度的第一项
            intermidArray=self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]
            intermidArray=cp.sum(intermidArray[:,cp.newaxis,:,:,cp.newaxis]*self.basic3bFeatArrayOfAllABpairs[:,:,:,cp.newaxis,:],axis=2)
            intermidArray2=-self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis,cp.newaxis]
            subDfeat+=(intermidArray[:,:,:,cp.newaxis,:,cp.newaxis]*intermidArray2[:,:,cp.newaxis,:,cp.newaxis,:]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)

            #这里是近邻原子属于featLabel (A,B)中的B类原子时的梯度的第二项
            intermidArray=self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]
            intermidArray=cp.sum(-intermidArray[:,cp.newaxis,:,:,cp.newaxis,cp.newaxis]*self.allABpairs3bDfeatArray[:,:,:,cp.newaxis,:,:],axis=2)
            intermidArray2=self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]
            subDfeat+=(intermidArray[:,:,:,cp.newaxis,:,:]*intermidArray2[:,:,cp.newaxis,:,cp.newaxis,cp.newaxis]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
            
        else:

            #接下来这一部分是考虑如果三体feat数目较多时，是否是matmul算法会更有效率
            
            #接下来同样首先是计算中心原子feat对自己坐标的梯度，这是第一项
            intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]\
                            ).transpose(0,2,1)[:,cp.newaxis,:,:],self.basic3bFeatArrayOfAllABpairs)
            dfeat[:,0,atomTypeSliceAB,:]+=cp.matmul((self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis,cp.newaxis]\
                            ).transpose(0,2,3,1)[:,:,cp.newaxis,:,:],intermidArray.transpose(0,2,1,3)[:,cp.newaxis,:,:,:]).transpose(0,1,2,4,3).reshape(self.numOfAtoms,-1,3)

            #这是中心原子feat对自己坐标的梯度的第二项
            intermidArray=cp.matmul((self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis,cp.newaxis]\
                           ).transpose(0,2,3,1)[:,cp.newaxis,:,:,:],self.basic3bFeatArrayOfAllABpairs[:,:,cp.newaxis,:,:])
            dfeat[:,0,atomTypeSliceAB,:]+=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]\
                           ).transpose(0,2,1)[:,cp.newaxis,cp.newaxis,:,:],intermidArray.transpose(0,2,4,1,3)).transpose(0,3,1,2,4).reshape(self.numOfAtoms,-1,3)

            #以下同样是计算中心原子的feat，对周围近邻原子的坐标的梯度，此思路是先计算，再转换成近邻原子对中心原子坐标的梯度            
            #首先是初始化
            subDfeat=cp.zeros((self.numOfAtoms,pm.maxNeighborNum,pm.mulNumOf3bFeat**3,3))

            
            #这里的第一部分是考虑，近邻原子属于featLabel (A,B)中的A类原子时的梯度，同样会有两项
            #这里是第一项
            intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]\
                          ).transpose(0,2,1)[:,cp.newaxis,:,:],self.basic3bFeatArrayOfAllABpairs.transpose(0,2,1,3))
            subDfeat+=((-self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis,cp.newaxis])[:,:,:,cp.newaxis,cp.newaxis,:]*\
                    intermidArray[:,:,cp.newaxis,:,:,cp.newaxis]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)

            #这里是近邻原子属于featLabel (A,B)中的A类原子时的梯度的第二项
            intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]\
                          ).transpose(0,2,1)[:,cp.newaxis,cp.newaxis,:,:],self.allABpairs3bDfeatArray.transpose(0,2,3,1,4))
            subDfeat+=((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis])[:,:,:,cp.newaxis,cp.newaxis,cp.newaxis]*\
                   intermidArray.transpose(0,1,3,2,4)[:,:,cp.newaxis,:,:,:]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)

            #这里是近邻原子属于featLabel (A,B)中的B类原子时的梯度的第一项
            intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]\
                          ).transpose(0,2,1)[:,cp.newaxis,:,:],self.basic3bFeatArrayOfAllABpairs)
            subDfeat+=(intermidArray[:,:,:,cp.newaxis,:,cp.newaxis]*(-self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis,cp.newaxis]\
                          )[:,:,cp.newaxis,:,cp.newaxis,:]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)

            #这里是近邻原子属于featLabel (A,B)中的B类原子时的梯度的第二项
            intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]\
                          ).transpose(0,2,1)[:,cp.newaxis,cp.newaxis,:,:],-self.allABpairs3bDfeatArray.transpose(0,1,3,2,4))
            subDfeat+=(intermidArray.transpose(0,1,3,2,4)[:,:,:,cp.newaxis,:,:]*(self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType]\
                          [:,:,cp.newaxis])[:,:,cp.newaxis,:,cp.newaxis,cp.newaxis]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
            
            
        


        


        
        '''
        dfeat[:,0,atomTypeSliceAB,:]+=cp.einsum('imar,im,inb,in,imnc->iabcr',self.allNeighborPairs3bDfeatArray,self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType],\
                                                self.basic3bFeatArrayOfAllNeighborPairs,self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType],self.basic3bFeatArrayOfAllABpairs\
                                                ).reshape(self.numOfAtoms,-1,3)
        dfeat[:,0,atomTypeSliceAB,:]+=cp.einsum('ima,im,inbr,in,imnc->iabcr',self.basic3bFeatArrayOfAllNeighborPairs,self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType],\
                                                self.allNeighborPairs3bDfeatArray,self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType],self.basic3bFeatArrayOfAllABpairs\
                                                ).reshape(self.numOfAtoms,-1,3)
        '''
                                                

        '''
        print(cp.einsum('ijar,ijnb,in,inc->ijabcr',self.allNeighborPairs3bDfeatArray[categoryListA],self.basic3bFeatArrayOfAllABpairs[categoryListA],\
                                                             self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][categoryListA],\
                                                             self.basic3bFeatArrayOfAllNeighborPairs[categoryListA]).shape)
        '''


        

        '''
        dfeat[categoryListA,1:,atomTypeSliceAB,:]+=cp.einsum('ijar,ijnb,in,inc->ijabcr',self.allNeighborPairs3bDfeatArray[categoryListA],self.basic3bFeatArrayOfAllABpairs[categoryListA],\
                                                             self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][categoryListA],\
                                                             self.basic3bFeatArrayOfAllNeighborPairs[categoryListA]).reshape(len(categoryListA),pm.maxNeighborNum,-1,3)
        dfeat[categoryListA,1:,atomTypeSliceAB,:]+=cp.einsum('ija,ijnb,in,incr->ijabcr',self.basic3bFeatArrayOfAllNeighborPairs[categoryListA],self.basic3bFeatArrayOfAllABpairs[categoryListA],\
                                                             self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][categoryListA],\
                                                             self.allNeighborPairs3bDfeatArray[categoryListA]).reshape(len(categoryListA),pm.maxNeighborNum,-1,3)
       
        dfeat[categoryListB,1:,atomTypeSliceAB,:]+=cp.einsum('ijma,im,ijbr,imc->ijabcr',self.basic3bFeatArrayOfAllABpairs[categoryListB],\
                                                   self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][categoryListB],self.allNeighborPairs3bDfeatArray[categoryListB],\
                                                   self.basic3bFeatArrayOfAllNeighborPairs[categoryListB]).reshape(len(categoryListB),pm.maxNeighborNum,-1,3)
        
        dfeat[categoryListB,1:,atomTypeSliceAB,:]+=cp.einsum('ijma,im,ijb,imcr->ijabcr',self.basic3bFeatArrayOfAllABpairs[categoryListB],\
                                                   self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][categoryListB],self.basic3bFeatArrayOfAllNeighborPairs[categoryListB],\
                                                   self.allNeighborPairs3bDfeatArray[categoryListB]).reshape(len(categoryListB),pm.maxNeighborNum,-1,3)
        '''



        
        #接下来的计算方式，需要先计算中心原子i的3d feat对每个近邻原子j的坐标的梯度，然后再转换
        #首先是初始化
        #subDfeat=cp.zeros((self.numOfAtoms,pm.maxNeighborNum,pm.mulNumOf3bFeat**3,3))


       
        '''#三体feat较多时的算法，部分2
        #第一部分，考虑j的种类是aNeighborAtomType的情形

        intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]).transpose(0,2,1)[:,cp.newaxis,:,:],\
                       self.basic3bFeatArrayOfAllABpairs.transpose(0,2,1,3))
        subDfeat+=((-self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis,cp.newaxis])[:,:,:,cp.newaxis,cp.newaxis,:]*\
                    intermidArray[:,:,cp.newaxis,:,:,cp.newaxis]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
        intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]).transpose(0,2,1)[:,cp.newaxis,cp.newaxis,:,:],\
                     self.allABpairs3bDfeatArray.transpose(0,2,3,1,4))
        subDfeat+=((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis])[:,:,:,cp.newaxis,cp.newaxis,cp.newaxis]*\
                   intermidArray.transpose(0,1,3,2,4)[:,:,cp.newaxis,:,:,:]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
        '''#
        




        
        #subDfeat+=cp.einsum('ijar,ij,inb,in,injc->ijabcr',-self.allNeighborPairs3bDfeatArray,self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType],self.basic3bFeatArrayOfAllNeighborPairs,\
                            #self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType],self.basic3bFeatArrayOfAllABpairs).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
        #subDfeat+=cp.einsum('ija,ij,inb,in,injcr->ijabcr',self.basic3bFeatArrayOfAllNeighborPairs,self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType],self.basic3bFeatArrayOfAllNeighborPairs,\
                            #self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType],self.allABpairs3bDfeatArray).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
      
        


        
            
        '''#
        #第二部分，考虑j的种类是bNeighborAtomType的情形

        intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]).transpose(0,2,1)[:,cp.newaxis,:,:],self.basic3bFeatArrayOfAllABpairs)
        subDfeat+=(intermidArray[:,:,:,cp.newaxis,:,cp.newaxis]*(-self.allNeighborPairs3bDfeatArray*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis,cp.newaxis])[:,:,cp.newaxis,:,cp.newaxis,:]\
                   ).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
        intermidArray=cp.matmul((self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType][:,:,cp.newaxis]).transpose(0,2,1)[:,cp.newaxis,cp.newaxis,:,:],\
                   -self.allABpairs3bDfeatArray.transpose(0,1,3,2,4))
        subDfeat+=(intermidArray.transpose(0,1,3,2,4)[:,:,:,cp.newaxis,:,:]*(self.basic3bFeatArrayOfAllNeighborPairs*self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType][:,:,cp.newaxis]
                                )[:,:,cp.newaxis,:,cp.newaxis,cp.newaxis]).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
        '''#


        #subDfeat+=cp.einsum('ima,im,ijbr,ij,ijmc->ijabcr',self.basic3bFeatArrayOfAllNeighborPairs,self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType],-self.allNeighborPairs3bDfeatArray,\
                            #self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType],self.basic3bFeatArrayOfAllABpairs).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)
        #subDfeat+=cp.einsum('ima,im,ijb,ij,ijmcr->ijabcr',self.basic3bFeatArrayOfAllNeighborPairs,self.maskDictOfAllAtomTypesInNeighborList[aNeighborAtomType],self.basic3bFeatArrayOfAllNeighborPairs,\
                            #self.maskDictOfAllAtomTypesInNeighborList[bNeighborAtomType],-self.allABpairs3bDfeatArray).reshape(self.numOfAtoms,pm.maxNeighborNum,-1,3)




        #接下来是将这个subDfeat倒换到真正的dfeat中
        '''
        #加了mask这一段是错误的，会导致A==B的时候，计算不对，至于为何以后再探讨，另外，似乎加了这个速度也反而变慢了
        mask=cp.isin(self.neighborIndexOfNeighborAtoms,categoryListA)+cp.isin(self.neighborIndexOfNeighborAtoms,categoryListB)
        dfeat[self.neighborIndexOfCenterAtoms[mask],self.neighborAtomOrderInNeighborListOfCenterAtoms[mask]+1,atomTypeSliceAB]+=subDfeat[self.neighborIndexOfNeighborAtoms[mask],self.centerAtomOrderInNeighborListOfNeighborAtoms[mask]]
        '''
        dfeat[self.neighborIndexOfCenterAtoms,self.neighborAtomOrderInNeighborListOfCenterAtoms+1,atomTypeSliceAB]+=subDfeat[self.neighborIndexOfNeighborAtoms,self.centerAtomOrderInNeighborListOfNeighborAtoms]
       


 
    
if __name__=='__main__':   
    input('Press Enter to quit test:')
