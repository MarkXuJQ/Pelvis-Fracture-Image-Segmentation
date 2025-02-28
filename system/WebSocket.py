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


# 获取聊天记录
# 假设有一个 WebSocket 服务器：
@socketio.on('get_chat_history')
def get_chat_history(data):
    sender_id = data['sender_id']
    receiver_id = data['receiver_id']
    print(345)
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
        print(567)
        emit('chat_history', {'history': chat_history_data}, broadcast=False)
    except Exception as e:
        logging.error(f"Error fetching chat history: {e}")
    finally:
        cursor.close()
        connection.close()
    print('Requesting chat history:', data)
    # 执行获取聊天记录的操作并返回
    emit('chat_history', {'history': []})
    print(999)

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
        print("cnm")
        # 获取任务的任务标题
        #cursor.execute("SELECT task_id, task_title FROM tasks WHERE assigned_doctor_id = %s")  # 获取任务ID和任务标题
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



@socketio.on('get_task_details')
def get_task_details(data):
    task_id = data['task_id']
    print(345)  # 打印调试信息

    # 从数据库中获取任务信息
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            SELECT task_title, task_description, assigned_doctor_id, patient_id, due_date, status 
            FROM tasks
            WHERE task_id = %s
        """, (task_id,))

        task_details = cursor.fetchone()

        if task_details:
            # 将任务详情返回给客户端
            task_details_data = {
                'task_title': task_details[0],
                'task_description': task_details[1],
                'assigned_doctor_id': task_details[2],
                'patient_id': task_details[3],
                'due_date': task_details[4].strftime("%Y-%m-%d %H:%M:%S"),
                'status': task_details[5]
            }
            logging.info(f"Sending task details: {task_details_data}")
            print(567)  # 打印调试信息
            emit('task_details', {'task': task_details_data}, broadcast=False)
        else:
            # 如果没有找到任务详情，返回空字典
            emit('task_details', {'task': {}})

    except Exception as e:
        logging.error(f"Error fetching task details: {e}")
    finally:
        cursor.close()
        connection.close()

    print('Requesting task details:', data)  # 打印请求的任务数据
    print(999)  # 打印调试信息


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


if __name__ == '__main__':
    print("Starting WebSocket server on ws://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000,allow_unsafe_werkzeug=True)
