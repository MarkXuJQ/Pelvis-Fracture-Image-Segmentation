#!/usr/bin/env python

__author__ = 'Q610098308'


# TODO: podpiac suwaki do callbacka
# TODO: ustawiz dobre zakresy na suwakach okienkowania
# TODO: opakowac w zewnetrzny interfejs
# TODO: rozkminic metode na kolorowanie przedzialow
#  TODO: dobrze sie bawic

import vtk
from PyQt5 import QtWidgets, QtCore
from DicomVis_ui import Ui_Form   #导入ui设计文件
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

DefaultColor =[[200,0,0,255],
[100,200,0,0],
[100,255,255,255],
[100,0,0,0],
[100,255,155,0],
[100,0,80,80],
[100,255,60,0],
[100,80,0,255],
[125,125,125,205],
[100,0,255,255],
[100,73,60,255],
[100,255,55,255],
[100,0,100,0],
[50,85,0,0],
[50,85,0,127],
[50,255,255,255],
[50,255,255,255],
[50,170,170,0],
[50,255,96,178],
[50,190,190,190],
[0,255,255,0] ,
[0,0,0,0] ]


# 尝试导入自定义模块，若导入失败，则创建空的类
try:
    from lib import StudyData as StudyData
    from lib import VisuAnalysisWidget
except(ImportError):
    class VisuAnalysisWidget(QWidget):
        pass

# vtk回调函数类
class vtkMyCallback():
    """
    交互的Callback
    """
    def __init__(self):
        #super(self).__init__()
        self.IPW = []       #图像平面小部件
        self.RCW = []       #ResliceCursorWidget

    #def Execute(self, caller, event,callData):
    def __call__(self, caller, ev):
        #更新图像平面小部件的位置 
        rep = caller.GetRepresentation()
        rc = rep.GetResliceCursorActor().GetCursorAlgorithm().GetResliceCursor()

        for i in range(3):
            ps = self.IPW[i].GetPolyDataAlgorithm()
            #vtkPlaneSource * ps = static_cast < vtkPlaneSource * > (this->IPW[i]->GetPolyDataAlgorithm());
            ps.SetOrigin(self.RCW[i].GetResliceCursorRepresentation().GetPlaneSource().GetOrigin())
            ps.SetPoint1(
                self.RCW[i].GetResliceCursorRepresentation().GetPlaneSource().GetPoint1())
            ps.SetPoint2(
                self.RCW[i].GetResliceCursorRepresentation().GetPlaneSource().GetPoint2())
            #################################################################
            # ps.SetNormal(rc.GetPlane(i).GetNormal())
            # ps.SetCenter(rc.GetPlane(i).GetOrigin())
            self.IPW[i].UpdatePlacement()

        for i in range(3):
            self.RCW[i].Render()

        self.IPW[0].GetInteractor().GetRenderWindow().Render()

    #设置图像平面小部件和ResliceCursorWidget
    def setPlaneWidget(self, param):
        self.IPW.append(param)

    def setResliceCursor(self, param):
        self.RCW.append(param)

# DicomVis类继承自VisuAnalysisWidget类
class DicomVis(VisuAnalysisWidget):

    def __init__(self, parent = None):
        # 变量初始化
        self.dataExtent = []
        self.actorList = []
        self.dataDimensions = []
        self.planeWidget = []
        self.dataRange = ()

        self.ImageData = vtk.vtkImageData()
        self.LabelImageData = vtk.vtkImageData()

        # initialize GUI
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # define viewers定义视图
        self.ResliceImageView =[]
        [self.viewerXY, self.viewerYZ, self.viewerXZ] = [vtk.vtkResliceImageViewer() for x in range(3)]

        # self.viewerXY.GetImageActor().GetProperty().SetOpacity(0.3)
        # self.viewerYZ.GetImageActor().GetProperty().SetOpacity(0.3)
        # self.viewerXZ.GetImageActor().GetProperty().SetOpacity(0.3)
        self.ResliceImageView.append(self.viewerXY)
        self.ResliceImageView.append(self.viewerYZ)
        self.ResliceImageView.append(self.viewerXZ)

        # set render windows for viewers设置渲染窗口
        self.RenderWindow = []
        self.RenderWindow.append(self.ui.XYPlaneWidget.GetRenderWindow())
        self.RenderWindow.append(self.ui.YZPlaneWidget.GetRenderWindow())
        self.RenderWindow.append(self.ui.XZPlaneWidget.GetRenderWindow())

        self.ui.XYPlaneWidget.GetRenderWindow().SetMultiSamples(False)
        self.ui.YZPlaneWidget.GetRenderWindow().SetMultiSamples(False)
        self.ui.XZPlaneWidget.GetRenderWindow().SetMultiSamples(False)

        # 将渲染器设置给视图
        self.viewerXY.SetRenderWindow(self.ui.XYPlaneWidget.GetRenderWindow())
        self.viewerYZ.SetRenderWindow(self.ui.YZPlaneWidget.GetRenderWindow())
        self.viewerXZ.SetRenderWindow(self.ui.XZPlaneWidget.GetRenderWindow())

        # attach interactors to viewers设置交互器
        self.viewerXY.SetupInteractor(self.ui.XYPlaneWidget)
        self.viewerYZ.SetupInteractor(self.ui.YZPlaneWidget)
        self.viewerXZ.SetupInteractor(self.ui.XZPlaneWidget)

        # set slicing orientation for viewers设置切片定位
        self.viewerXY.SetSliceOrientationToXZ()
        self.viewerYZ.SetSliceOrientationToYZ()
        self.viewerXZ.SetSliceOrientationToXY()

        # setup volume rendering设置体素渲染
        self.volRender = vtk.vtkRenderer()
        self.volRenWin = self.ui.VolumeWidget.GetRenderWindow()
        self.volRenWin.SetMultiSamples(0)
        self.volRenWin.AddRenderer(self.volRender)

        # Set up the interaction设置交互
        self.interactorXY = self.viewerXY.GetRenderWindow().GetInteractor()
        self.interactorXZ = self.viewerXZ.GetRenderWindow().GetInteractor()
        self.interactorYZ = self.viewerYZ.GetRenderWindow().GetInteractor()
        interactorVolume = self.volRenWin.GetInteractor()

        self.RenderList=[]
        for i in range(3):
            #render = vtk.vtkRenderer()
            #self.RenderList.append(render)
            self.RenderList.append(self.ResliceImageView[i].GetRenderer())

        self.contoursA = vtk.vtkContourFilter()
        #self.contoursA.SetValue(0, 500)
        self.volRender = vtk.vtkRenderer()
        self.volRenWin = self.ui.VolumeWidget.GetRenderWindow()
        self.volRenWin.AddRenderer(self.volRender)

    # 更新数据
    def updateData(self, studydata):
        self.load_study_from_path(studydata.getPath())
    # 显示数据
    def show_Data(self):
        # 计算数据尺寸和范围
        self.dataExtent = self.ImageData.GetExtent()
        dataDimensionX = self.dataExtent[1]-self.dataExtent[0]
        dataDimensionY = self.dataExtent[3]-self.dataExtent[2]
        dataDimensionZ = self.dataExtent[5]-self.dataExtent[4]
        self.dataDimensions = [dataDimensionX, dataDimensionY, dataDimensionZ]

        # Calculate index of middle slice计算中间切片的索引
        midslice1 = int((self.dataExtent[1]-self.dataExtent[0])/2 + self.dataExtent[0])
        midslice2 = int((self.dataExtent[3]-self.dataExtent[2])/2 + self.dataExtent[2])
        midslice3 = int((self.dataExtent[5]-self.dataExtent[4])/2 + self.dataExtent[4])

        # Calculate enter
        center = [midslice1, midslice2, midslice3]

        # Get data range获取数据范围
        self.dataRange = self.ImageData.GetScalarRange()
        print(self.dataRange)

        ##################################
        # 获取交互器
        interactor = self.volRenWin.GetInteractor()  # vtk.vtkRenderWindowInteractor()

        # 清空平面小部件列表
        self.planeWidget.clear()
        #window.SetInteractor(interactor)
        # 创建CellPicker和Property
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        ipwProp = vtk.vtkProperty()

        # create plane source here planeWidget创建平面小部件
        for i in range(3):
            self.planeWidget.append(vtk.vtkImagePlaneWidget())
            self.planeWidget[i].SetInteractor(interactor)
            self.planeWidget[i].SetPicker(picker)
            self.planeWidget[i].RestrictPlaneToVolumeOn()

            color = [0, 0, 0]
            color[i] = 1
            self.planeWidget[i].GetPlaneProperty().SetColor(color)
            self.planeWidget[i].SetTexturePlaneProperty(ipwProp)
            self.planeWidget[i].TextureInterpolateOff()

            self.planeWidget[i].SetResliceInterpolateToLinear()
            self.planeWidget[i].SetInputData(self.ImageData)
            self.planeWidget[i].SetPlaneOrientation(i)
            self.planeWidget[i].SetSliceIndex(center[i] // 2)
            self.planeWidget[i].DisplayTextOn()
            self.planeWidget[i].SetDefaultRenderer(self.volRender)
            self.planeWidget[i].SetWindowLevel(1000, 100)
            self.planeWidget[i].On()
            self.planeWidget[i].InteractionOn()

        # 设置平面小部件的LookupTable
        self.planeWidget[1].SetLookupTable(self.planeWidget[0].GetLookupTable())
        self.planeWidget[2].SetLookupTable(self.planeWidget[0].GetLookupTable())

        # 创建ResliceCursor和Callback对象
        resliceCursorWidget = []
        resliceCursorRep = []

        resliceCursor =  vtk.vtkResliceCursor() #self.ResliceImageView[0].GetResliceCursor()#
        resliceCursor.SetCenter(self.ImageData.GetCenter())
        resliceCursor.SetThickMode(0)
        resliceCursor.SetThickness(10, 10, 10)

        callback = vtkMyCallback()

        # 我们在这里设置observer.设置观察者
        TestMyCallbackList = []
        for i in range(3):

            resliceCursorWidget.append(self.ResliceImageView[i].GetResliceCursorWidget())
            resliceCursorWidget[i].SetInteractor(self.ResliceImageView[i].GetRenderWindow().GetInteractor())
            #
            resliceCursorRep.append(vtk.vtkResliceCursorLineRepresentation())
            resliceCursorRep[i].GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(resliceCursor)
            resliceCursorRep[i].GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(i)
            resliceCursorWidget[i].SetRepresentation(resliceCursorRep[i])
            #resliceCursorRep.append(self.ResliceImageView[i].GetResliceCursorWidget().GetRepresentation())
            resliceCursorRep[i].GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(resliceCursor)
            resliceCursorRep[i].GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(i)
            # self.ResliceImageView[i].SetResliceCursor(resliceCursor)
            self.ResliceImageView[i].SetInputData(self.ImageData)
            self.ResliceImageView[i].SetSliceOrientation(i)
            self.ResliceImageView[i].SetResliceModeToAxisAligned()
            #self.ResliceImageView[i].SetSlice(center[i] // 2)


            (minValue, maxValue) = self.ImageData.GetScalarRange()
            reslice = resliceCursorRep[i].GetReslice()

            callback.setPlaneWidget(self.planeWidget[i])
            callback.setResliceCursor(self.ResliceImageView[i].GetResliceCursorWidget())

            reslice.SetBackgroundColor(minValue, minValue, minValue, minValue)
            self.ResliceImageView[i].GetResliceCursorWidget().AddObserver(vtk.vtkCommand.InteractionEvent, callback)

            # 禁用平面小部件并设置相机位置
            for i in range(3):
                self.planeWidget[i].SetEnabled(False)
            camPos = [0, 0, 0]
            camPos[i] = 1

        # 设置平面小部件的窗宽窗位以及颜色映射
        for i in range(3):
            self.planeWidget[i].SetWindowLevel(maxValue - minValue, (minValue + maxValue) / 2.0)
            self.planeWidget[i].GetColorMap().SetLookupTable(resliceCursorRep[0].GetLookupTable())

        # 将渲染器添加到渲染器列表中
        for i in range(3):
            self.RenderList.append(self.ResliceImageView[i].GetRenderer())

        # 更新平面小部件位置并渲染
        for i in range(3):
            #self.RenderList[i].ResetCamera()
            #self.RenderWindow[i].Render()
            self.planeWidget[i].UpdatePlacement()
            self.ResliceImageView[i].GetResliceCursorWidget().Render()
            self.planeWidget[i].GetInteractor().GetRenderWindow().Render()

        self.volRender.ResetCamera()
        self.volRenWin.Render()

        # 将MPR复选框设置为选中状态
        self.ui.MPRCheckBox.setChecked(True)

    # 从文件路径获取NIFTI图像3d
    def GetImageDataFromPath(self,path):
        print("Path:",path)
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(path)
        reader.Update()
        # self.volRender.RemoveActor()

        count =len(self.actorList)
        for aindex in range(count):
            self.volRender.RemoveActor(self.actorList[aindex])

        self.LabelImageData.DeepCopy(reader.GetOutput())
        (minValue, maxValue) = self.LabelImageData.GetScalarRange()

        index =0
        for scalar in range(int(minValue)+1, int(maxValue)):

            threshold = vtk.vtkImageThreshold()
            threshold.ThresholdBetween(scalar,scalar)
            threshold.SetInputData(self.LabelImageData)
            threshold.SetOutValue(-1024)
            threshold.SetInValue(1024)
            threshold.Update()

            if threshold.GetOutput().GetNumberOfPoints()<1:
                continue

            contoursA = vtk.vtkContourFilter()
            contoursA.SetInputData(threshold.GetOutput())
            contoursA.SetValue(0, 0.5)
            contoursA.Update()

            # create the mapper创建Mapper

            mapperA = vtk.vtkPolyDataMapper()
            mapperA.ScalarVisibilityOff()
            mapperA.SetInputData(contoursA.GetOutput())
            ##############################################

            # create the actor创建Actor
            actorA = vtk.vtkActor()
            propA = vtk.vtkProperty()
            propA.SetColor(DefaultColor[index][0] / 255.0, DefaultColor[index][1] / 255.0,
                           DefaultColor[index][2] / 255.0)
            propA.SetOpacity(1)
            actorA.SetProperty(propA)
            actorA.SetMapper(mapperA)
            self.actorList.append(actorA)
            self.volRender.AddActor(actorA)

            index=index+1

        self.volRender.ResetCamera()
        self.volRenWin.Render()

    # 从文件夹路径加载DICOM数据
    def load_dicom_from_path(self, studyPath):
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(studyPath)
        self.reader.Update()
        #self.GetDicmoDataFromPath(studyPath)

        self.ImageData.DeepCopy( self.reader.GetOutput())
        self.show_Data()
    # 从文件路径加载NII数据
    def load_Nii_from_path(self, studyPath):
        print("PATH:",studyPath)
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(studyPath)
        reader.Update()
        #self.GetDicmoDataFromPath(studyPath)

        self.ImageData.DeepCopy(reader.GetOutput())
        self.show_Data()
       # self.on_MPRCheckBox_stateChanged(2)
    # MPR复选框状态更改的槽函数
    @QtCore.pyqtSlot(int)
    def on_MPRCheckBox_stateChanged(self, state):
        mode = 0
        if 2== state:
            mode=1
        for pair in zip([self.viewerXY, self.viewerYZ, self.viewerXZ]):
            pair[0].SetResliceMode(mode)
            pair[0].GetRenderer().ResetCamera()
            pair[0].Render()
    # 关闭窗口
    def close(self):
        self.ui.XYPlaneWidget.Finalize()
        self.ui.YZPlaneWidget.Finalize()
        self.ui.XZPlaneWidget.Finalize()
        self.ui.VolumeWidget.Finalize()
    # 关闭事件
    def closeEvent(self, event):
        self.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = DicomVis()
    print(type(window))
    window.show()

    studyPath = "D:\\DICOM"
    window.load_dicom_from_path(studyPath)
    exitStatus = app.exec_()
    #del(window)
    sys.exit(exitStatus)
