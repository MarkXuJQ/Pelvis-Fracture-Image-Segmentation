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

'''@app.route('/')
def index():
    return 'Hello, World!'  # 或者你可以返回一个模板，或者静态文件

# favicon 路由
@app.route('/favicon.ico')
def favicon():
    return '', 204  # 返回 No Content
'''
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


if __name__ == '__main__':
    print("Starting WebSocket server on ws://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000,allow_unsafe_werkzeug=True)
