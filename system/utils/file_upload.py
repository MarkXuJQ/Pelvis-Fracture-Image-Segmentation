import os
import shutil
from datetime import datetime
import paramiko
import time
from config.settings import STORAGE_CONFIG

class FileUploader:
    def __init__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.config = STORAGE_CONFIG

    def connect(self):
        """建立SSH连接"""
        try:
            self.ssh.connect(
                hostname=self.config['host'],
                port=self.config['port'],
                username=self.config['username'],
                password=self.config['password']
            )
            self.sftp = self.ssh.open_sftp()
            return True
        except Exception as e:
            print(f"连接失败: {str(e)}")
            raise

    def disconnect(self):
        """关闭连接"""
        if hasattr(self, 'sftp'):
            self.sftp.close()
        if hasattr(self, 'ssh'):
            self.ssh.close()

    def upload_medical_image(self, file_path, patient_id, image_type):
        """
        上传医学图像
        :param file_path: 本地文件路径
        :param patient_id: 病人ID
        :param image_type: 'ct' 或 'xray'
        :return: 存储路径
        """
        try:
            self.connect()
            
            # 创建目标路径
            now = datetime.now()
            relative_path = f"{image_type}/{now.year}/{now.month:02d}/{now.day:02d}"
            filename = f"{patient_id}_{now.strftime('%Y%m%d_%H%M%S')}.dcm"
            
            remote_dir = f"{self.config['base_path']}/{relative_path}"
            remote_path = f"{remote_dir}/{filename}"
            
            # 确保远程目录存在
            self.ssh.exec_command(f"mkdir -p {remote_dir}")
            
            # 上传文件
            self.sftp.put(file_path, remote_path)
            
            # 返回相对路径（用于存储在数据库中）
            return f"{relative_path}/{filename}"
            
        finally:
            self.disconnect()

    def get_medical_image(self, relative_path, local_path):
        """
        下载医学图像
        :param relative_path: 数据库中存储的相对路径
        :param local_path: 本地保存路径
        """
        try:
            self.connect()
            remote_path = f"{self.config['base_path']}/{relative_path}"
            self.sftp.get(remote_path, local_path)
        finally:
            self.disconnect() 