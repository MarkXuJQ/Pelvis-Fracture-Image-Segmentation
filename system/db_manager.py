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
        query = f"SELECT * FROM {user_type}s WHERE id = %s AND password = %s"
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
        
    except Error as e:
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
        
    except Error as e:
        print(f"数据库错误: {str(e)}")
        return False, f"注册失败: {str(e)}"

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
                id VARCHAR(20) PRIMARY KEY,
                name VARCHAR(50) NOT NULL,
                password VARCHAR(100) NOT NULL,
                phone VARCHAR(20),
                specialty VARCHAR(50)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)
        
        # 创建病人表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id VARCHAR(20) PRIMARY KEY,
                name VARCHAR(50) NOT NULL,
                password VARCHAR(100) NOT NULL,
                phone VARCHAR(20)
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
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info("Database initialized successfully")
        return True
        
    except Error as e:
        logger.error(f"Error initializing database: {e}")
        return False

# 在程序启动时初始化数据库
if __name__ == "__main__":
    init_database()
