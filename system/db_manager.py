import pyodbc
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 手动建立连接并将其传递给 SQLAlchemy
'''connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=菠啵莓;DATABASE=Pelvis-Fracture-Image-Segmentation;Trusted_Connection=yes"
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
class Doctor(Base):
    __tablename__ = 'doctors'
    doctor_id = Column(String(20), primary_key=True)
    doctor_name = Column(String(50))
    doctor_password = Column(String(20))
    phone = Column(String(11))
    specialty = Column(String(50))

# 病人表
class Patient(Base):
    __tablename__ = 'patients'
    patient_id = Column(String(20), primary_key=True)
    patient_name = Column(String(50))
    patient_password = Column(String(20))
    phone = Column(String(11))

# 管理员表
class Admin(Base):
    __tablename__ = 'admins'
    admin_id = Column(String(20), primary_key=True)
    admin_name = Column(String(50))
    admin_password = Column(String(20))
    phone = Column(String(11))
def verify_user(user_id, password, user_type):
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