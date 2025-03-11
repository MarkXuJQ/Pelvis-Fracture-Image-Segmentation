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

'''@socketio.on('update_task_title')
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
'''

@socketio.on('update_task_title')
def update_task_title(data):
    task_id = data['task_id']
    new_task_title = data['new_task_title'].strip()
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # 检查 new_task_title 是否已存在
        cursor.execute("""
                   SELECT COUNT(*) FROM tasks WHERE task_title = %s
               """, (new_task_title,))
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            print(f"❌ 任务标题 '{new_task_title}' 已存在，更新失败")
            emit('task_title_updated', {
                'task_id': None,
                'error': '任务已存在，你可以在 \'add task\' 里输入该标题并选择是否加入该任务'
            }, broadcast=False)
            return

            # 先找到 task_id 对应的任务信息
        cursor.execute("""
                    SELECT task_title, task_description, due_date, patient_id 
                    FROM tasks 
                    WHERE task_id = %s
                """, (task_id,))
        result = cursor.fetchone()

        if not result:
            print(f"❌ 任务 ID {task_id} 不存在")
            emit('task_title_updated', {'task_id': None, 'error': 'Task not found'}, broadcast=False)
            return

        old_task_title, task_description, due_date, patient_id = result
        # 更新所有相同 task_description、due_date、patient_id 的任务标题
        cursor.execute("""
                    UPDATE tasks
                    SET task_title = %s
                    WHERE task_description = %s 
                      AND due_date = %s 
                      AND patient_id = %s
                """, (new_task_title, task_description, due_date, patient_id))

        updated_rows = cursor.rowcount  # 获取被更新的行数
        connection.commit()
        # 3. 发送更新信息给客户端
        emit('task_title_updated', {
            'task_id': task_id,
            'new_task_title': new_task_title,
            'updated_rows': updated_rows
        }, broadcast=True)

    except pymysql.IntegrityError:
        print(f"❌ 任务标题 '{new_task_title}' 已存在，更新失败")
        emit('task_title_updated', {
            'task_id': None, 'error': 'Task title already exists'
        }, broadcast=False)
    except pymysql.MySQLError as e:
        logging.error(f"❌ Error updating task title: {e}")
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
    """ 任务创建请求：先检查任务是否已存在 """
    try:
        connection = get_connection()
        cursor = connection.cursor()
        # 提取数据
        task_title = data['task_title']
        task_description = data['task_description']
        due_date = data['due_date']
        status = data['status']
        assigned_doctor_ids = data['assigned_doctor_ids']
        patient_id = data['patient_id']
        created_task_ids = []
        current_doctor_id = str(data.get('current_doctor_id'))  # 只检查当前医生

        # 先检查数据库是否已有相同任务标题的任务
        cursor.execute("""
            SELECT task_id, assigned_doctor_id, task_description, due_date, status, patient_id 
            FROM tasks 
            WHERE task_title = %s
        """, (task_title,))
        existing_tasks = cursor.fetchall()

        if existing_tasks:
            # **任务已存在，只检查当前医生是否已加入**
            existing_doctor_ids = {row[1] for row in existing_tasks}  # {医生ID}

            if current_doctor_id in existing_doctor_ids:
                # **医生已加入该任务**
                emit('task_creation_failed', {'message': f'医生 {current_doctor_id} 已经加入了任务 "{task_title}"'},
                     broadcast=False)
            else:
                # **医生未加入，询问是否要加入**
                existing_task_info = [
                    {'task_id': row[0], 'task_description': row[2], 'due_date': str(row[3]), 'status': row[4],
                     'patient_id': row[5]}
                    for row in existing_tasks
                ]
                emit('existing_task_found', {
                    'task_title': task_title,
                    'tasks': existing_task_info,
                    'doctor_id': current_doctor_id
                }, broadcast=False)

        else:
            # **任务不存在，遍历所有传入的医生并创建**
            for doctor_id in assigned_doctor_ids:
                new_task_id = insert_new_task(cursor, task_title, task_description, due_date, status, doctor_id,
                                              patient_id)
                created_task_ids.append(new_task_id)

            connection.commit()
            # 任务创建成功，更新前端
            if created_task_ids:
                emit('task_created', {'task_ids': created_task_ids, 'message': '任务创建成功'}, broadcast=True)

    except pymysql.MySQLError as e:
        connection.rollback()
        logging.error(f"❌ 创建任务失败: {e}")
        emit('task_creation_failed', {'error': str(e)}, broadcast=False)
    finally:
        cursor.close()
        connection.close()

def insert_new_task(cursor, task_title, task_description, due_date, status, doctor_id, patient_id):
    """ 插入新任务，并返回 task_id """
    query = """
        INSERT INTO tasks (task_title, task_description, due_date, status, assigned_doctor_id, patient_id)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (task_title, task_description, due_date, status, doctor_id, patient_id))
    return cursor.lastrowid  # 返回插入的任务 ID


@socketio.on('confirm_join_task')
def confirm_join_task(response):
    """ 处理医生确认加入已有任务 """
    try:
        connection = get_connection()
        cursor = connection.cursor()

        doctor_id = response['doctor_id']
        task_title = response['task_title']
        confirm = response['confirm']

        if confirm:
            # **医生确认加入任务**
            cursor.execute("""
                SELECT task_description, due_date, patient_id 
                FROM tasks 
                WHERE task_title = %s 
                LIMIT 1
            """, (task_title,))
            task_data = cursor.fetchone()
            if task_data:
                task_description, due_date, patient_id = task_data
                new_task_id = insert_new_task(cursor, task_title, task_description, due_date, "pending", doctor_id,
                                              patient_id)
                connection.commit()

                emit('task_updated', {'task_title': task_title, 'updated_tasks': [new_task_id]}, broadcast=True)
            else:
                emit('task_creation_failed', {'message': '任务不存在，无法加入'}, broadcast=False)

        else:
            # **医生拒绝加入任务**
            emit('task_creation_failed', {'message': f'医生 {doctor_id} 拒绝加入任务 "{task_title}"'}, broadcast=False)

    except pymysql.MySQLError as e:
        connection.rollback()
        logging.error(f"❌ 医生加入任务失败: {e}")
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
