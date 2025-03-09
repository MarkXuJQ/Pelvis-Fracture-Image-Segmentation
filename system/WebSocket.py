from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pymysql
from datetime import datetime
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_manager import get_connection
from db_config import db_config
import os
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG)
os.environ['FLASK_ENV'] = 'development'  # 或者 'production'

# 创建数据库连接
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
Session = sessionmaker(bind=engine)
session = Session()

app = Flask(__name__)
socketio = SocketIO(app)


# 处理消息的函数
# 在 handle_message 中添加日志
@socketio.on('send_message')
def handle_message(data):
    sender_id = data['sender_id']
    receiver_id = data['receiver_id']
    message_content = data['message']

    # 保存消息到数据库
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO chat_records (sender_id, receiver_id, message_content)
            VALUES (%s, %s, %s)
        """, (sender_id, receiver_id, message_content))

        # 2. **存入未读消息，但不影响聊天记录**

        cursor.execute("""
            INSERT INTO messages (sender_id, receiver_id, message_type, message_content, is_read)
            VALUES (%s, %s, 'system', %s, %s)
        """, (sender_id, receiver_id,  message_content, 'false'))  # `False` 代表未读

        connection.commit()
        logging.info(f"Message from {sender_id} to {receiver_id}: {message_content}")
    except Exception as e:
        logging.error(f"Error inserting message: {e}")
    finally:
        cursor.close()
        connection.close()
    print('Received message:', data)
    # 返回消息给客户端
    emit('receive_message', data, broadcast=True)

@socketio.on('get_doctors_except_user')
def get_doctors_except_user(data):
    """查询数据库，返回除了当前 user_id 以外的所有医生"""
    doctor_id = data['user_id']
    try:
        connection = get_connection()
        cursor = connection.cursor()
        query = """
            SELECT doctor_id, doctor_name FROM doctors
            WHERE doctor_id != %s
        """
        cursor.execute(query, (doctor_id,))
        doctors = [{'doctor_id': row[0], 'doctor_name': row[1]} for row in cursor.fetchall()]

        # 发送医生列表给客户端
        emit('doctors_list', {'doctors': doctors}, broadcast=False)

    except pymysql.MySQLError as e:
        logging.error(f"获取医生列表失败: {e}")
    finally:
        cursor.close()
        connection.close()

# 获取聊天记录
# 假设有一个 WebSocket 服务器：
@socketio.on('get_chat_history')
def get_chat_history(data):
    sender_id = data['sender_id']
    receiver_id = data['receiver_id']
    # 从数据库中获取聊天记录
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            SELECT sender_id, receiver_id, message_content, timestamp 
            FROM chat_records
            WHERE (sender_id = %s AND receiver_id = %s) OR (sender_id = %s AND receiver_id = %s)
            ORDER BY timestamp ASC
        """, (sender_id, receiver_id, receiver_id, sender_id))

        chat_history = cursor.fetchall()

        # 返回历史记录给客户端
        chat_history_data = [{'sender_id': row[0], 'receiver_id': row[1], 'message_content': row[2],
                              'timestamp': row[3].strftime("%Y-%m-%d %H:%M:%S")} for row in chat_history]
        logging.info(f"Sending chat history: {chat_history_data}")
        emit('chat_history', {'history': chat_history_data}, broadcast=False)
    except Exception as e:
        logging.error(f"Error fetching chat history: {e}")
    finally:
        cursor.close()
        connection.close()
    print('Requesting chat history:', data)
    # 执行获取聊天记录的操作并返回
    emit('chat_history', {'history': []})

@socketio.on('get_all_patients')
def get_all_patients():
    """查询数据库，返回所有病人的 patient_id 和 patient_name"""
    try:
        connection = get_connection()
        cursor = connection.cursor()
        query = """
            SELECT patient_id, patient_name FROM patients
        """
        cursor.execute(query)
        patients = [{'patient_id': row[0], 'patient_name': row[1]} for row in cursor.fetchall()]

        # 发送病人列表给客户端
        emit('patients_list', {'patients': patients}, broadcast=False)

    except pymysql.MySQLError as e:
        logging.error(f"获取病人列表失败: {e}")
    finally:
        cursor.close()
        connection.close()

@socketio.on('get_task_list')
def get_task_list(data):
    print("Fetching task list...")
    assigned_doctor_id = data['assigned_doctor_id']
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("""
                    SELECT task_id, task_title 
                    FROM tasks
                    WHERE assigned_doctor_id = %s
                """, (assigned_doctor_id))
        tasks = cursor.fetchall()  # 获取所有任务数据

        # 返回任务列表给客户端
        task_list_data = [{'task_id': row[0], 'task_title': row[1]} for row in tasks]
        print(f"Sending task list: {task_list_data}")
        emit('task_list', {'tasks': task_list_data}, broadcast=False)

    except pymysql.MySQLError as e:
        logging.error(f"Error fetching task list: {e}")
    finally:
        cursor.close()
        connection.close()

@socketio.on('delete_task')
def delete_task(data):
    task_id = data['task_id']
    print(f"Deleting task with ID: {task_id}")

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # 执行 SQL 查询删除任务
        cursor.execute("""
            DELETE FROM tasks
            WHERE task_id = %s
        """, (task_id,))

        connection.commit()

        # 向客户端确认删除成功
        emit('task_deleted', {'task_id': task_id}, broadcast=False)
        print(f"Task with ID {task_id} has been deleted.")

    except pymysql.MySQLError as e:
        logging.error(f"Error deleting task: {e}")
        emit('task_deleted', {'task_id': None, 'error': str(e)}, broadcast=False)
    finally:
        cursor.close()
        connection.close()

@socketio.on('update_task_title')
def update_task_title(data):
    task_id = data['task_id']
    new_task_title = data['new_task_title']
    print(f"Updating task ID {task_id} with new title: {new_task_title}")

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # 执行 SQL 语句更新任务标题
        cursor.execute("""
            UPDATE tasks
            SET task_title = %s
            WHERE task_id = %s
        """, (new_task_title, task_id))

        connection.commit()

        # 向客户端确认更新成功
        emit('task_title_updated', {'task_id': task_id, 'new_task_title': new_task_title}, broadcast=False)
        print(f"Task ID {task_id} title updated to: {new_task_title}")

    except pymysql.MySQLError as e:
        logging.error(f"Error updating task title: {e}")
        emit('task_title_updated', {'task_id': None, 'error': str(e)}, broadcast=False)
    finally:
        cursor.close()
        connection.close()
@socketio.on('update_task_details')
def update_task_details(data):
    """更新任务详情"""
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # 获取任务信息
        task_id = data.get('task_id')
        task_description = data.get('task_description')
        due_date = data.get('due_date')
        status = data.get('status')
        assigned_doctor_id = data.get('assigned_doctor_id')

        if not task_id:
            emit('task_update_failed', {'error': '任务 ID 不能为空'}, broadcast=False)
            return

        # 更新任务数据
        query = """
            UPDATE tasks 
            SET task_description = %s, due_date = %s, status = %s, assigned_doctor_id = %s
            WHERE task_id = %s
        """
        cursor.execute(query, (task_description, due_date, status, assigned_doctor_id, task_id))
        connection.commit()

        print(f"任务更新成功: {data}")

        # 发送任务更新通知给相关医生
        emit('task_updated', {'assigned_doctor_id': assigned_doctor_id, 'task_id': task_id}, broadcast=True)

    except pymysql.MySQLError as e:
        connection.rollback()
        logging.error(f"更新任务失败: {e}")
        emit('task_update_failed', {'error': str(e)}, broadcast=False)
    finally:
        cursor.close()
        connection.close()

@socketio.on('create_task')
def create_task(data):
    """批量创建任务，并分配给多个医生"""
    try:
        connection = get_connection()
        cursor = connection.cursor()
        # 提取数据
        task_title = data['task_title']
        task_description = data['task_description']
        due_date = data['due_date']
        status = data['status']
        assigned_doctor_ids = data['assigned_doctor_ids']  # 这里是一个列表
        patient_id = data['patient_id']

        # 记录插入的任务 ID
        created_task_ids = []

        # 循环遍历所有医生，插入多条任务
        for doctor_id in assigned_doctor_ids:
            query = """
                INSERT INTO tasks (task_title, task_description, due_date, status, assigned_doctor_id, patient_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (task_title, task_description, due_date, status, doctor_id, patient_id))
            created_task_ids.append(cursor.lastrowid)  # 记录任务 ID

        connection.commit()  # 提交事务

        # 发送成功创建的任务列表给客户端
        emit('task_created', {'task_ids': created_task_ids, 'message': '任务创建成功'}, broadcast=False)

    except pymysql.MySQLError as e:
        connection.rollback()  # 发生错误时回滚
        logging.error(f"创建任务失败: {e}")
        emit('task_creation_failed', {'error': str(e)}, broadcast=False)
    finally:
        cursor.close()
        connection.close()


@socketio.on('get_task_details')
def get_task_details(data):
    """获取指定医生的任务，并查询相同任务标题的所有医生及任务状态"""
    try:
        connection = get_connection()
        cursor = connection.cursor()

        assigned_doctor_id = data['assigned_doctor_id']
        task_title = data['task_title']
        # 获取该医生的任务
        cursor.execute("""
            SELECT task_id, task_title, task_description, due_date, status, assigned_doctor_id, patient_id
            FROM tasks 
            WHERE assigned_doctor_id = %s
        """, (assigned_doctor_id,))
        doctor_tasks = cursor.fetchall()
        if not doctor_tasks:
            print(f"医生 {assigned_doctor_id} 没有分配的任务")
            emit('task_details', {'tasks': []}, broadcast=False)
            return
        # 在该医生的任务中，查找 task_title 匹配的任务
        matching_tasks = [task for task in doctor_tasks if task[1] == task_title]

        if not matching_tasks:
            print(f"医生 {assigned_doctor_id} 没有 task_title 为 {task_title} 的任务")
            emit('task_details', {'tasks': []}, broadcast=False)
            return

        # 取出该任务的描述、截止日期、病人 ID（以第一个匹配的任务为准）
        task_description = matching_tasks[0][2]
        due_date = matching_tasks[0][3].strftime("%Y-%m-%d %H:%M:%S") if matching_tasks[0][3] else "无截止日期"
        patient_id = matching_tasks[0][6]

        #查询所有任务中相同 task_title 的任务
        cursor.execute("""
            SELECT assigned_doctor_id, status 
            FROM tasks 
            WHERE task_title = %s
        """, (task_title,))
        all_tasks = cursor.fetchall()  # 获取所有任务的 doctor_id 和状态
        doctor_status_list = []
        doctor_ids = [task[0] for task in all_tasks]  # 取出所有医生 ID
        if not doctor_ids:
            print(f"没有找到 task_title 为 {task_title} 的任务")
            emit('task_details', {'tasks': []}, broadcast=False)
            return

        # 通过 assigned_doctor_id 查询医生名字
        cursor.execute(f"""
            SELECT doctor_id, doctor_name 
            FROM doctors 
            WHERE doctor_id IN ({','.join(['%s'] * len(doctor_ids))})
        """, tuple(doctor_ids))
        doctor_name_map = {row[0]: row[1] for row in cursor.fetchall()}  # 转换为字典 {doctor_id: doctor_name}

        # 组合数据
        for task in all_tasks:
            doctor_id, status = task
            doctor_name = doctor_name_map.get(doctor_id, "未知医生")
            doctor_status_list.append({
                'assigned_doctor_id': doctor_id,
                'doctor_name': doctor_name,
                'status': status
            })
        # 发送任务详情
        emit('task_details', {
            'task_title': task_title,
            'task_description': task_description,
            'due_date': due_date,
            'patient_id': patient_id,
            'tasks': doctor_status_list
        }, broadcast=False)
    except pymysql.MySQLError as e:
        logging.error(f"获取任务列表失败: {e}")
        emit('task_details_failed', {'error': str(e)}, broadcast=False)
    finally:
        cursor.close()
        connection.close()


if __name__ == '__main__':
    print("Starting WebSocket server on ws://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000,allow_unsafe_werkzeug=True)
