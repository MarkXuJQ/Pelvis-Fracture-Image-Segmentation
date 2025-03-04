import os
import shutil
from datetime import datetime
import paramiko
import time
from config.settings import STORAGE_CONFIG
from paramiko import SSHClient, AutoAddPolicy
from tqdm import tqdm

class FileUploader:
    def __init__(self):
        self.ssh = SSHClient()
        self.ssh.set_missing_host_key_policy(AutoAddPolicy())
        self.config = STORAGE_CONFIG
        self.chunk_size = 32768  # 32KB chunks
        self.retry_count = 3     # 最大重试次数

    def connect(self):
        """建立SSH连接"""
        try:
            self.ssh.connect(
                hostname=self.config['host'],
                port=self.config['port'],
                username=self.config['username'],
                password=self.config['password'],
                timeout=30  # 增加超时时间
            )
            self.sftp = self.ssh.open_sftp()
            self.sftp.get_channel().settimeout(300)  # 设置SFTP通道超时时间为5分钟
            return True
        except Exception as e:
            print(f"连接失败: {str(e)}")
            raise

    def disconnect(self):
        """关闭连接"""
        if hasattr(self, 'sftp'):
            try:
                self.sftp.close()
            except:
                pass
        if hasattr(self, 'ssh'):
            try:
                self.ssh.close()
            except:
                pass

    def upload_medical_image(self, file_path, patient_id, image_type, progress_callback=None):
        """
        上传医学图像，支持大文件和断点续传
        """
        for attempt in range(self.retry_count):
            try:
                self.connect()
                
                # 创建目标路径
                now = datetime.now()
                relative_path = f"{image_type}/{now.year}/{now.month:02d}/{now.day:02d}"
                _, ext = os.path.splitext(file_path)
                filename = f"{patient_id}_{now.strftime('%Y%m%d_%H%M%S')}{ext}"
                
                remote_dir = f"{self.config['base_path']}/{relative_path}"
                remote_path = f"{remote_dir}/{filename}"
                
                # 确保远程目录存在
                self.ssh.exec_command(f"mkdir -p {remote_dir}")
                
                # 获取文件大小
                file_size = os.path.getsize(file_path)
                uploaded_size = 0
                
                # 分块上传文件
                with open(file_path, 'rb') as f:
                    with self.sftp.file(remote_path, 'wb') as remote_file:
                        while uploaded_size < file_size:
                            chunk = f.read(self.chunk_size)
                            if not chunk:
                                break
                            
                            remote_file.write(chunk)
                            uploaded_size += len(chunk)
                            
                            if progress_callback:
                                progress = (uploaded_size / file_size) * 100
                                progress_callback(progress)
                            
                            # 添加小延迟防止连接断开
                            time.sleep(0.001)
                
                return f"{relative_path}/{filename}"
                
            except Exception as e:
                print(f"上传尝试 {attempt + 1} 失败: {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(1)  # 等待1秒后重试
                    self.disconnect()  # 断开旧连接
                    continue
                raise Exception(f"上传失败，已重试{self.retry_count}次: {str(e)}")
                
            finally:
                self.disconnect()

    def get_medical_image(self, relative_path, local_path, progress_callback=None):
        """
        下载医学图像
        :param relative_path: 数据库中存储的相对路径
        :param local_path: 本地保存路径
        :param progress_callback: 进度回调函数
        """
        try:
            self.connect()
            remote_path = f"{self.config['base_path']}/{relative_path}"
            
            # 获取文件大小
            file_size = self.sftp.stat(remote_path).st_size
            downloaded_size = 0

            # 分块下载
            with self.sftp.open(remote_path, 'rb') as remote_file:
                with open(local_path, 'wb') as local_file:
                    while True:
                        chunk = remote_file.read(32768)  # 32KB chunks
                        if not chunk:
                            break
                        local_file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if progress_callback:
                            progress = (downloaded_size / file_size) * 100
                            progress_callback(progress)
                        
                        # 添加小延迟防止界面卡死
                        time.sleep(0.001)
                        
        finally:
            self.disconnect() 