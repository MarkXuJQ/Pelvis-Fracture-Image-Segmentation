import os
import sys
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QCheckBox
from PyQt5.QtCore import Qt

class VTK3DViewer(QWidget):
    """基于VTK的3D可视化组件"""
    
    def __init__(self, parent=None):
        super(VTK3DViewer, self).__init__(parent)
        
        # 创建布局
        self.main_layout = QVBoxLayout(self)
        
        # 创建VTK渲染窗口
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtk_widget)
        
        # 创建控制面板
        self.controls_layout = QHBoxLayout()
        self.main_layout.addLayout(self.controls_layout)
        
        # 创建渲染模式选择器
        self.mode_label = QLabel("渲染模式:")
        self.controls_layout.addWidget(self.mode_label)
        
        self.render_mode = QComboBox()
        self.render_mode.addItems(["体积渲染", "表面渲染"])
        self.render_mode.currentIndexChanged.connect(self.change_render_mode)
        self.controls_layout.addWidget(self.render_mode)
        
        # 创建不透明度滑块
        self.opacity_label = QLabel("不透明度:")
        self.controls_layout.addWidget(self.opacity_label)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.change_opacity)
        self.controls_layout.addWidget(self.opacity_slider)
        
        # 创建显示原始数据复选框
        self.show_original = QCheckBox("显示原始数据")
        self.show_original.setChecked(True)
        self.show_original.stateChanged.connect(self.toggle_original_data)
        self.controls_layout.addWidget(self.show_original)
        
        # 创建显示分割结果复选框
        self.show_segmentation = QCheckBox("显示分割结果")
        self.show_segmentation.setChecked(True)
        self.show_segmentation.stateChanged.connect(self.toggle_segmentation)
        self.controls_layout.addWidget(self.show_segmentation)
        
        # 初始化VTK渲染器
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # 设置交互样式
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # 初始化数据
        self.volume_actor = None
        self.surface_actor = None
        self.original_volume_actor = None
        self.original_surface_actor = None
        self.volume_data = None
        self.mask_data = None
        
        # 设置颜色映射
        self.setup_color_maps()
        
        # 启动交互器
        self.interactor.Initialize()
        
    def setup_color_maps(self):
        """设置颜色映射"""
        # 创建原始数据的颜色映射
        self.original_color = vtk.vtkColorTransferFunction()
        self.original_color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        self.original_color.AddRGBPoint(0.5, 0.5, 0.5, 0.5)
        self.original_color.AddRGBPoint(1.0, 1.0, 1.0, 1.0)
        
        # 创建原始数据的不透明度映射
        self.original_opacity = vtk.vtkPiecewiseFunction()
        self.original_opacity.AddPoint(0.0, 0.0)
        self.original_opacity.AddPoint(0.3, 0.0)
        self.original_opacity.AddPoint(0.7, 0.5)
        self.original_opacity.AddPoint(1.0, 0.9)
        
        # 创建分割结果的颜色映射
        self.segmentation_color = vtk.vtkColorTransferFunction()
        self.segmentation_color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        self.segmentation_color.AddRGBPoint(1.0, 1.0, 0.0, 0.0)  # 红色
        
        # 创建分割结果的不透明度映射
        self.segmentation_opacity = vtk.vtkPiecewiseFunction()
        self.segmentation_opacity.AddPoint(0.0, 0.0)
        self.segmentation_opacity.AddPoint(1.0, 0.7)
        
    def set_volume_data(self, volume, mask=None):
        """
        设置3D体积数据
        
        参数:
            volume: 原始3D体积数据 (numpy数组)
            mask: 分割掩码 (numpy数组，可选)
        """
        if volume is None:
            return
            
        # 存储数据副本
        self.volume_data = volume.copy()
        
        # 标准化到[0,1]范围
        if volume.max() > 0:
            volume = volume / volume.max()
            
        # 创建VTK图像数据
        self.vtk_volume = self._numpy_to_vtk_image(volume)
        
        # 如果有掩码，也转换
        if mask is not None:
            self.mask_data = mask.astype(np.float32)
            self.vtk_mask = self._numpy_to_vtk_image(self.mask_data)
        else:
            self.mask_data = None
            self.vtk_mask = None
            
        # 渲染数据
        self.render_data()
        
    def _numpy_to_vtk_image(self, np_array):
        """将numpy数组转换为VTK图像数据"""
        # 确保数据类型正确
        if np_array.dtype != np.float32:
            np_array = np_array.astype(np.float32)
            
        # 获取形状
        if len(np_array.shape) == 3:
            depth, height, width = np_array.shape
        else:
            raise ValueError("输入必须是3D数组")
            
        # 创建VTK数据数组
        vtk_data = vtk.vtkFloatArray()
        vtk_data.SetNumberOfComponents(1)
        vtk_data.SetNumberOfTuples(depth * height * width)
        
        # 填充数据 (需要正确的内存布局)
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    vtk_data.SetValue(z * width * height + y * width + x, np_array[z, y, x])
                    
        # 创建VTK图像数据
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, depth)
        vtk_image.GetPointData().SetScalars(vtk_data)
        
        return vtk_image
        
    def render_data(self):
        """渲染数据"""
        # 清除现有actors
        self.renderer.RemoveAllViewProps()
        
        # 根据当前渲染模式选择渲染方法
        render_mode = self.render_mode.currentText()
        
        if render_mode == "体积渲染":
            self._volume_rendering()
        else:
            self._surface_rendering()
            
        # 重置相机
        self.renderer.ResetCamera()
        
        # 更新渲染
        self.vtk_widget.GetRenderWindow().Render()
        
    def _volume_rendering(self):
        """体积渲染方法"""
        # 如果选择了显示原始数据
        if self.show_original.isChecked() and self.vtk_volume is not None:
            # 创建体积属性
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetColor(self.original_color)
            volume_property.SetScalarOpacity(self.original_opacity)
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()
            
            # 创建体积映射器
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(self.vtk_volume)
            
            # 创建体积
            self.original_volume_actor = vtk.vtkVolume()
            self.original_volume_actor.SetMapper(mapper)
            self.original_volume_actor.SetProperty(volume_property)
            
            # 添加到渲染器
            self.renderer.AddViewProp(self.original_volume_actor)
            
        # 如果选择了显示分割结果且有掩码
        if self.show_segmentation.isChecked() and self.vtk_mask is not None:
            # 创建体积属性
            seg_property = vtk.vtkVolumeProperty()
            seg_property.SetColor(self.segmentation_color)
            seg_property.SetScalarOpacity(self.segmentation_opacity)
            seg_property.ShadeOn()
            seg_property.SetInterpolationTypeToLinear()
            
            # 创建体积映射器
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(self.vtk_mask)
            
            # 创建体积
            self.volume_actor = vtk.vtkVolume()
            self.volume_actor.SetMapper(mapper)
            self.volume_actor.SetProperty(seg_property)
            
            # 添加到渲染器
            self.renderer.AddViewProp(self.volume_actor)
            
    def _surface_rendering(self):
        """表面渲染方法"""
        # 如果选择了显示原始数据
        if self.show_original.isChecked() and self.vtk_volume is not None:
            # 创建等值面
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(self.vtk_volume)
            contour.SetValue(0, 0.5)  # 等值面阈值
            
            # 创建映射器
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOff()
            
            # 创建actor
            self.original_surface_actor = vtk.vtkActor()
            self.original_surface_actor.SetMapper(mapper)
            self.original_surface_actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # 灰色
            self.original_surface_actor.GetProperty().SetOpacity(0.5)
            
            # 添加到渲染器
            self.renderer.AddActor(self.original_surface_actor)
            
        # 如果选择了显示分割结果且有掩码
        if self.show_segmentation.isChecked() and self.vtk_mask is not None:
            # 创建等值面
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(self.vtk_mask)
            contour.SetValue(0, 0.5)  # 等值面阈值
            
            # 创建映射器
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOff()
            
            # 创建actor
            self.surface_actor = vtk.vtkActor()
            self.surface_actor.SetMapper(mapper)
            self.surface_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色
            self.surface_actor.GetProperty().SetOpacity(0.7)
            
            # 添加到渲染器
            self.renderer.AddActor(self.surface_actor)
            
    def change_render_mode(self, index):
        """改变渲染模式"""
        self.render_data()
        
    def change_opacity(self, value):
        """改变不透明度"""
        opacity = value / 100.0
        
        # 对于体积渲染
        if self.volume_actor:
            self.volume_actor.GetProperty().SetScalarOpacity(self.segmentation_opacity)
            
        # 对于表面渲染
        if self.surface_actor:
            self.surface_actor.GetProperty().SetOpacity(opacity)
            
        self.vtk_widget.GetRenderWindow().Render()
        
    def toggle_original_data(self, state):
        """切换原始数据显示"""
        self.render_data()
        
    def toggle_segmentation(self, state):
        """切换分割结果显示"""
        self.render_data()
        
    def cleanup(self):
        """清理VTK对象"""
        if self.interactor:
            self.interactor.TerminateApp() 