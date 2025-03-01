from datetime import datetime
import pyodbc
from sqlalchemy import create_engine, Column, Integer, String, CHAR, VARCHAR, DateTime, Date, ForeignKey, Enum, Text
import pymysql
#from sqlalchemy import Column, Integer, String, create_engine, Float, ForeignKey, NCHAR, VARCHAR
from sqlalchemy.dialects.mysql import ENUM
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# 数据库连接设置（MySQL）

#connection_string = "mysql+pymysql://root:hys12138@localhost:3306/pelvis"

# 创建 SQLAlchemy 引擎
#engine = create_engine(connection_string)

# 测试 SQLAlchemy 连接
# try:
#     with engine.connect() as connection:
#         print("SQLAlchemy 连接成功")
# except Exception as e:
#     print("SQLAlchemy 连接失败:", e)
#
# Session = sessionmaker(bind=engine)
# session = Session()

Base = declarative_base()

# 医生表
class doctors(Base):
    __tablename__ = 'doctors'
    doctor_id = Column(String(20), primary_key=True)
    doctor_name = Column(String(50),nullable=False)
    doctor_password = Column(String(20),nullable=False)
    phone = Column(String(11))
    specialty = Column(String(50))

    # 添加与聊天记录的关系
    sent_messages = relationship("chat_records", foreign_keys="[chat_records.sender_id]", back_populates="sender")
    received_messages = relationship("chat_records", foreign_keys="[chat_records.receiver_id]", back_populates="receiver")

# 病人表
class patients(Base):
    __tablename__ = 'patients'
    # 定义表的字段
    patient_id = Column(String(6), primary_key=True, nullable=False)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(Enum('male', 'female', 'other'), nullable=True)
    contact_person = Column(String(100), nullable=True)
    contact_phone = Column(String(20), nullable=True)
    phone_number = Column(String(20), nullable=True)
    age = Column(Integer, nullable=True)
    id_card = Column(String(18), nullable=True)
    patient_name = Column(String(100), nullable=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), nullable=True)

    # 反向关系，表示一个病人有多个骨折历史
    fracturehistories = relationship("fracturehistories", back_populates="patients", cascade="all, delete")


class fracturehistories(Base):
    __tablename__ = 'fracturehistories'
    # 定义表的字段
    history_id = Column(String(6), primary_key=True, nullable=False)
    patient_id = Column(String(6), ForeignKey('patients.patient_id', ondelete="CASCADE", onupdate="RESTRICT"),
                        nullable=False)
    fracture_date = Column(Date, nullable=False)
    fracture_location = Column(Enum('pelvis', 'femur', 'spine', 'other'), nullable=False)
    severity_level = Column(Enum('mild', 'moderate', 'severe'), nullable=False)
    diagnosis_details = Column(Text, nullable=True)

    # 定义外键关系（可以用于查询相关患者信息）
    patients = relationship("patients", back_populates="fracturehistories")


# 管理员表
class Admin(Base):
    __tablename__ = 'admins'
    admin_id = Column(String(20), primary_key=True)
    admin_name = Column(String(50))
    admin_password = Column(String(20))
    phone = Column(String(11))

class chat_records(Base):
    __tablename__ = 'chat_records'
    # 定义表的字段
    id = Column(Integer, primary_key=True, autoincrement=True)
    sender_id = Column(Integer, ForeignKey('doctors.doctor_id', ondelete="CASCADE", onupdate="RESTRICT"), nullable=False)
    receiver_id = Column(Integer, ForeignKey('doctors.doctor_id', ondelete="CASCADE", onupdate="RESTRICT"), nullable=False)
    message_content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 定义外键关系（可以用于查询发送者和接收者的用户信息）
    sender = relationship("doctors", foreign_keys=[sender_id], back_populates="sent_messages")
    receiver = relationship("doctors", foreign_keys=[receiver_id], back_populates="received_messages")


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
'''

import pymysql
from pymysql import Error
from db_config import db_config
import logging

# 设置日志记录
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_connection():
    try:
        connection = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            charset=db_config['charset'],
            port=db_config['port']
        )
        '''cursor = connection.cursor()
        try:
            # 删除表格的 SQL 语句
            cursor.execute("DROP TABLE IF EXISTS fracturehistories")
            cursor.execute("DROP TABLE IF EXISTS patients")
            connection.commit()  # 提交事务
            print("Table deleted successfully!")
        except pymysql.MySQLError as e:
            print(f"Error deleting table: {e}")
        finally:
            cursor.close()
            connection.close()'''
        logger.info("Successfully connected to MySQL database")
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None


def verify_user(user_id, password, user_type):
    try:
        connection = get_connection()
        if not connection:
            logger.error("Failed to establish database connection")
            return False, "数据库连接失败"

        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # 根据 user_type 选择正确的 ID 列名
        id_column_map = {
            "doctor": "doctor_id",
            "patient": "patient_id",
            "admin": "admin_id"
        }

        # 获取正确的 ID 列名
        id_column = id_column_map.get(user_type.lower())
        if not id_column:
            logger.error(f"Invalid user type: {user_type}")
            return False, "用户类型错误"

        # 生成 SQL 语句
        query = f"SELECT * FROM {user_type}s WHERE {id_column} = %s AND password = %s"

        logger.debug(f"Executing query: {query} with params: {user_id}, {password}")
        cursor.execute(query, (user_id, password))

        user = cursor.fetchone()

        cursor.close()
        connection.close()

        if user:
            logger.info(f"User {user_id} successfully logged in")
            return True, "登录成功！"
        logger.warning(f"Failed login attempt for user {user_id}")
        return False, "用户名或密码错误"

    except pymysql.Error as e:
        logger.error(f"Database error: {str(e)}")
        return False, "数据库错误"


def register_user(user_id, name, password, phone, user_type, specialty=None):
    try:
        connection = get_connection()
        if not connection:
            return False, "数据库连接失败"
            
        cursor = connection.cursor()
        
        if user_type == 'doctor':
            query = """INSERT INTO doctors 
                    (id, name, password, phone, specialty) 
                    VALUES (%s, %s, %s, %s, %s)"""
            values = (user_id, name, password, phone, specialty)
        else:
            query = f"""INSERT INTO {user_type}s 
                    (id, name, password, phone) 
                    VALUES (%s, %s, %s, %s)"""
            values = (user_id, name, password, phone)
            
        cursor.execute(query, values)
        connection.commit()
        
        cursor.close()
        connection.close()
        
        return True, "注册成功"
    except Exception as e:
        if connection:
            connection.rollback()
        return False, f"数据库错误: {e}"


# MySQL 数据库初始化函数
def init_database():
    try:

        # 首先创建数据库连接（不指定数据库名）
        conn_params = db_config.copy()
        conn_params.pop('database')  # 移除数据库名称
        
        connection = pymysql.connect(**conn_params)
        cursor = connection.cursor()
        
        # 创建数据库
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute(f"USE {db_config['database']}")


        # 创建医生表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                doctor_id VARCHAR(20) PRIMARY KEY,
                doctor_name VARCHAR(50) NOT NULL,
                doctor_password VARCHAR(100) NOT NULL,
                phone VARCHAR(20),
                specialty VARCHAR(50)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)

        # 创建病人表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(6) PRIMARY KEY,
                patient_name VARCHAR(100) NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                phone_number VARCHAR(20),
                date_of_birth DATE,
                gender ENUM('male', 'female', 'other'),
                contact_person VARCHAR(100),
                contact_phone VARCHAR(20),
                email VARCHAR(100) UNIQUE,
                age INT,
                id_card VARCHAR(18) UNIQUE
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)

        # 创建骨折病历表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fracturehistories (
                history_id VARCHAR(6) PRIMARY KEY,
                patient_id VARCHAR(6) NOT NULL,
                fracture_date DATE NOT NULL,
                fracture_location ENUM('pelvis', 'femur', 'spine', 'other') NOT NULL,
                severity_level ENUM('mild', 'moderate', 'severe') NOT NULL,
                diagnosis_details TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                    ON DELETE CASCADE
                    ON UPDATE RESTRICT
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)
        #聊天记录
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sender_id INT NOT NULL,
                receiver_id INT NOT NULL,
                message_content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sender_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY (receiver_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE ON UPDATE RESTRICT
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)

        # 创建管理员表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id VARCHAR(20) PRIMARY KEY,
                name VARCHAR(50) NOT NULL,
                password VARCHAR(100) NOT NULL,
                phone VARCHAR(20)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)
        # 创建协作任务表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id INT AUTO_INCREMENT PRIMARY KEY,
                task_title VARCHAR(200) NOT NULL,
                task_description TEXT,
                assigned_doctor_id VARCHAR(20),
                patient_id VARCHAR(6),
                due_date DATETIME,
                status ENUM('pending', 'completed', 'in_progress') DEFAULT 'pending',
                FOREIGN KEY (assigned_doctor_id) REFERENCES doctors(doctor_id)
                    ON DELETE SET NULL ON UPDATE CASCADE,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)
        # 创建文档表
        cursor.execute("""
                   CREATE TABLE IF NOT EXISTS documents (
                       doc_id INT AUTO_INCREMENT PRIMARY KEY,
                       patient_id VARCHAR(6) NOT NULL,
                       doctor_id VARCHAR(20) NOT NULL,
                       file_path VARCHAR(255),
                       description TEXT,
                       uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                       FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                           ON DELETE CASCADE ON UPDATE RESTRICT,
                       FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
                           ON DELETE CASCADE ON UPDATE RESTRICT
                   ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
               """)
        # 创建笔记表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                note_id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id VARCHAR(6) NOT NULL,
                doctor_id VARCHAR(20) NOT NULL,
                note_content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT,
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
                    ON DELETE CASCADE ON UPDATE RESTRICT
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info("Database initialized successfully")
        return True
        
    except Error as e:
        logger.error(f"Error initializing database: {e}")
        return False

# 插入病人信息的函数
def insert_patient(patient_id, patient_name, password_hash, phone_number=None, date_of_birth=None,
                   gender=None, contact_person=None, contact_phone=None, email=None, age=None, id_card=None):
    """插入病人信息"""
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # 验证性别合法性
        if gender and gender not in ['male', 'female', 'other']:
            raise ValueError(f"Invalid gender: {gender}")

        # 插入数据的 SQL
        insert_query = """
        INSERT INTO patients (patient_id, patient_name, password_hash, phone_number, date_of_birth, gender, 
                              contact_person, contact_phone, email, age, id_card)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (patient_id, patient_name, password_hash, phone_number, date_of_birth,
                                      gender, contact_person, contact_phone, email, age, id_card))
        connection.commit()
        logger.info(f"Successfully inserted patient {patient_name} with ID {patient_id}.")
    except pymysql.MySQLError as e:
        logger.error(f"Error inserting patient: {e}")
    finally:
        cursor.close()
        connection.close()

def insert_fracture_history(history_id, patient_id, fracture_date, fracture_location, severity_level, diagnosis_details):
    """插入骨折病历信息"""
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # 验证骨折位置和严重程度的合法性
        if fracture_location not in ['pelvis', 'femur', 'spine', 'other']:
            raise ValueError(f"Invalid fracture location: {fracture_location}")
        if severity_level not in ['mild', 'moderate', 'severe']:
            raise ValueError(f"Invalid severity level: {severity_level}")

        # 检查患者是否存在
        cursor.execute("SELECT patient_id FROM patients WHERE patient_id = %s", (patient_id,))
        if not cursor.fetchone():
            raise ValueError(f"Patient with ID {patient_id} does not exist.")

        # 插入数据的 SQL
        insert_query = """
        INSERT INTO fracturehistories (history_id, patient_id, fracture_date, fracture_location, severity_level, diagnosis_details)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (history_id, patient_id, fracture_date, fracture_location, severity_level, diagnosis_details))
        connection.commit()
        logger.info(f"Successfully inserted fracture history for patient ID {patient_id}.")
    except pymysql.MySQLError as e:
        logger.error(f"Error inserting fracture history: {e}")
    finally:
        cursor.close()
        connection.close()


def insert_doctor(doctor_id, doctor_name, doctor_password, phone=None, specialty=None):
    """插入医生信息"""
    try:
        connection = get_connection()  # 获取数据库连接
        cursor = connection.cursor()

        # 插入数据的 SQL
        insert_query = """
        INSERT INTO doctors (doctor_id, doctor_name, doctor_password, phone, specialty)
        VALUES (%s, %s, %s, %s, %s)
        """
        print(22)
        # 执行插入操作
        cursor.execute(insert_query, (doctor_id, doctor_name, doctor_password, phone, specialty))
        connection.commit()  # 提交事务

        logger.info(f"Successfully inserted doctor {doctor_name} with ID {doctor_id}.")
    except pymysql.MySQLError as e:
        logger.error(f"Error inserting doctor: {e}")
    finally:
        cursor.close()
        connection.close()


def insert_chat_record(sender_id, receiver_id, message_content):
    """插入聊天记录"""
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # 验证发送者和接收者 ID 是否有效
        if sender_id == receiver_id:
            raise ValueError("Sender and receiver cannot be the same.")

        # 插入数据的 SQL
        insert_query = """
        INSERT INTO chat_records (sender_id, receiver_id, message_content)
        VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (sender_id, receiver_id, message_content))
        connection.commit()
        logger.info(f"Successfully inserted chat record from {sender_id} to {receiver_id}.")
    except pymysql.MySQLError as e:
        logger.error(f"Error inserting chat record: {e}")
    except ValueError as ve:
        logger.error(ve)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def insert_task(task_id, task_title, task_description=None, assigned_doctor_id=None, patient_id=None,
                due_date=None, status='pending'):
    """插入任务信息"""
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # 验证任务状态合法性
        if status not in ['pending', 'completed', 'in_progress']:
            raise ValueError(f"Invalid status: {status}")

        # 插入数据的 SQL
        insert_query = """
        INSERT INTO tasks (task_id, task_title, task_description, assigned_doctor_id, patient_id, due_date, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (task_id, task_title, task_description, assigned_doctor_id, patient_id, due_date, status))
        connection.commit()
        logger.info(f"Successfully inserted task {task_title} with ID {task_id}.")
    except pymysql.MySQLError as e:
        logger.error(f"Error inserting task: {e}")
    except ValueError as ve:
        logger.error(f"Error in status validation: {ve}")
    finally:
        cursor.close()
        connection.close()

import pymysql
from pymysql import Error
from datetime import datetime
import logging

logger = logging.getLogger(__name__)




insert_doctor(1, "Dr. John Doe", "password123", "1234567890", "Orthopedics")
insert_doctor(2, "Dr. Doe", "password123", "1234567890", "Orthopedics")
insert_doctor(3, "Dr. Jane Smith", "password456", "2345678901", "Cardiology")
insert_doctor(4, "Dr. Emily White", "password789", "3456789012", "Pediatrics")
insert_doctor(5, "Dr. Michael Brown", "password101", "4567890123", "Dermatology")
insert_doctor(6, "Dr. Lisa Green", "password202", "5678901234", "Neurology")
insert_doctor(7, "Dr. Mark Lee", "password303", "6789012345", "Gastroenterology")

# 假设要插入一条聊天记录
#insert_chat_record(2, 1, "Damn it!")
#insert_chat_record(2, 1, "Shit!")
#insert_chat_record(3, 1, "Shit!")
#insert_chat_record(1, 3, "ok!")
# 示例：插入一条病人信息
insert_patient(
    patient_id="P00001",
    patient_name="张三",
    password_hash="hashed_password_123",
    phone_number="1234567890",
    date_of_birth="1985-06-15",
    gender="male",
    contact_person="李四",
    contact_phone="0987654321",
    email="zhangsan@example.com",
    age=38,
    id_card="123456789012345678"
)

# 示例：插入一条骨折病历信息
insert_fracture_history(
    history_id="F00001",
    patient_id="P00001",
    fracture_date="2024-06-15",
    fracture_location="pelvis",
    severity_level="moderate",
    diagnosis_details="Fracture at the pelvic region."
)
# 插入第一条病人信息
insert_patient(
    patient_id="P00002",
    patient_name="李四",
    password_hash="hashed_password_456",
    phone_number="2345678901",
    date_of_birth="1990-08-22",
    gender="female",
    contact_person="王五",
    contact_phone="9876543210",
    email="lisi@example.com",
    age=35,
    id_card="234567890123456789"
)

# 插入第一条骨折病历信息
insert_fracture_history(
    history_id="F00002",
    patient_id="P00002",
    fracture_date="2024-07-10",
    fracture_location="femur",
    severity_level="severe",
    diagnosis_details="Severe femur fracture due to accident."
)

# 插入第二条病人信息
insert_patient(
    patient_id="P00003",
    patient_name="王五",
    password_hash="hashed_password_789",
    phone_number="3456789012",
    date_of_birth="1987-02-18",
    gender="male",
    contact_person="赵六",
    contact_phone="8765432109",
    email="wangwu@example.com",
    age=38,
    id_card="345678901234567890"
)

# 插入第二条骨折病历信息
insert_fracture_history(
    history_id="F00003",
    patient_id="P00003",
    fracture_date="2024-07-20",
    fracture_location="spine",
    severity_level="mild",
    diagnosis_details="Mild spine fracture after a fall."
)
insert_task(
    task_id=1,
    task_title="Complete Patient Diagnosis",
    task_description="Diagnose the patient based on recent tests and provide recommendations.",
    assigned_doctor_id=1,
    patient_id="P00001",
    due_date="2025-03-01 15:00:00",
    status="pending"
)

# 在程序启动时初始化数据库
if __name__ == "__main__":
    init_database()
