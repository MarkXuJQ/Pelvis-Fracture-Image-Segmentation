from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_config import db_config
from db_manager import patients

# 创建数据库连接
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
Session = sessionmaker(bind=engine)
session = Session()

def get_all_patient_info():
    session = Session()  # 获取数据库会话
    try:
        patient_info = session.query(patients).all()

        # 将病人信息转换为字典列表，以便后续在表格中显示
        patient_list = []
        for patient in patient_info:
            patient_list.append({
                "patient_id": patient.patient_id,
                "patient_name": patient.patient_name,
                "age": patient.age,
                "gender": patient.gender,
                "id_card": patient.id_card,
                "date_of_birth": patient.date_of_birth,
                "phone_number": patient.phone_number,
                "password_hash": patient.password_hash,
                "contact_person": patient.contact_person,
                "contact_phone": patient.contact_phone,

            })

        # 返回字典列表
        return patient_list

    except Exception as e:
        print(f"Error fetching housing info: {e}")
        return []

    finally:
        # 确保关闭会话
        session.close()