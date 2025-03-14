import sys
import os

# 添加父目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pymysql
from system.database.db_config import db_config

def inspect_database():
    try:
        # 连接数据库
        connection = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            charset=db_config['charset'],
            port=db_config['port']
        )
        
        cursor = connection.cursor()
        
        # 查看所有表
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print("\n现有表单:")
        for table in tables:
            print(f"- {table[0]}")
            
            # 查看每个表的结构
            cursor.execute(f"DESCRIBE {table[0]}")
            columns = cursor.fetchall()
            print("\n列结构:")
            for col in columns:
                print(f"  {col[0]}: {col[1]}")
            print("\n")
            
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    inspect_database() 