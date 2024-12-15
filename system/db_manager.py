import pyodbc
from sqlalchemy import create_engine, Column, Integer, String, CHAR, VARCHAR, DateTime, Date, ForeignKey, Enum, Text

#from sqlalchemy import Column, Integer, String, create_engine, Float, ForeignKey, NCHAR, VARCHAR
from sqlalchemy.dialects.mysql import ENUM
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# 数据库连接设置
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=LAPTOP-5NGQ4BFB;DATABASE=Pelvis;Trusted_Connection=yes"
conn = pyodbc.connect(connection_string)
engine = create_engine("mssql+pyodbc://", creator=lambda: conn)

# 测试 SQLAlchemy 连接
try:
    with engine.connect() as connection:
        print("SQLAlchemy 连接成功")
except Exception as e:
    print("SQLAlchemy 连接失败:", e)

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# 医生表
'''class Doctor(Base):
    __tablename__ = 'doctors'
    doctor_id = Column(String(20), primary_key=True)
    doctor_name = Column(String(50))
    doctor_password = Column(String(20))
    phone = Column(String(11))
    specialty = Column(String(50))'''

# 病人表
class Patient(Base):
    __tablename__ = 'Patient'
    patient_id = Column(VARCHAR(20), primary_key=True)
    patient_name = Column(VARCHAR(50),nullable=False)
    patient_age = Column(Integer, nullable=False)
    date_of_birth = Column(Date, nullable=False)
    id_number = Column(CHAR(18), nullable=False)
    patient_password = Column(VARCHAR(20),nullable=False)
    phone = Column(VARCHAR(11),nullable=False)
    patient_gender = Column(CHAR(2), nullable=False)
    contact_person = Column(VARCHAR(36), nullable=False)
    contact_phone = Column(CHAR(11), nullable=False)
    # 反向关系，表示一个病人有多个骨折历史
    FractureHistories = relationship("FractureHistories", back_populates="Patient", cascade="all, delete")


class FractureHistories(Base):
    __tablename__ = 'FractureHistories'

    history_id = Column(Integer, primary_key=True, autoincrement=True)  # 病史ID，自动增长
    patient_id = Column(VARCHAR(20), ForeignKey('Patient.patient_id'), nullable=False)  # 外键，关联 Patients 表
    fracture_date = Column(Date, nullable=False)  # 骨折日期
    fracture_location = Column(Enum('pelvis', 'femur', 'spine', 'other', name='fracture_location_enum'),
                               nullable=False)  # 骨折部位
    fracture_type = Column(Enum('AO_A1', 'AO_A2', 'AO_B1', 'other', name='fracture_type_enum'),
                           nullable=False)  # 骨折类型（AO分型）
    severity_level = Column(Enum('mild', 'moderate', 'severe', name='severity_level_enum'), nullable=False)  # 骨折严重程度
    diagnosis_details = Column(Text)  # 诊断描述
    diagnosis_hospital = Column(VARCHAR(20),nullable=False)

    # 定义与 Patient 表的关系
    Patient = relationship("Patient", back_populates="FractureHistories")

# 管理员表
'''class Admin(Base):
    __tablename__ = 'admins'
    admin_id = Column(String(20), primary_key=True)
    admin_name = Column(String(50))
    admin_password = Column(String(20))
    phone = Column(String(11))'''

'''def verify_user(user_id, password, user_type):
    if user_type == 'doctor':
        user = session.query(Doctor).filter_by(doctor_id=user_id).first()
    elif user_type == 'patient':
        user = session.query(Patient).filter_by(patient_id=user_id).first()
    elif user_type == 'admin':
        user = session.query(Admin).filter_by(admin_id=user_id).first()
    else:
        return False, "无效的用户类型"

    if not user:
        return False, "用户不存在"

    if user_type == 'doctor':
        is_correct_password = user.doctor_password == password
    elif user_type == 'patient':
        is_correct_password = user.patient_password == password
    else:  # admin
        is_correct_password = user.admin_password == password

    if is_correct_password:
        return True, "登录成功"
    else:
        return False, "密码错误"
'''
'''def register_user(user_id, name, password, phone, user_type, specialty=None):
    if user_type == 'doctor':
        new_user = Doctor(doctor_id=user_id, doctor_name=name, doctor_password=password, phone=phone, specialty=specialty)
    elif user_type == 'patient':
        new_user = Patient(patient_id=user_id, patient_name=name, patient_password=password, phone=phone)
    elif user_type == 'admin':
        new_user = Admin(admin_id=user_id, admin_name=name, admin_password=password, phone=phone)
    else:
        return False, "无效的用户类型"

    try:
        session.add(new_user)
        session.commit()
        return True, "注册成功"
    except Exception as e:
        session.rollback()
        return False, f"数据库错误: {e}"
'''