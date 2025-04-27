import os
import sys
from PyQt5.QtWidgets import QApplication
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

if os.path.basename(current_dir) == 'system':
    from login_window import LoginWindow
    from WebSocket import socketio, app
else:
    from system.login_window import LoginWindow
    from system.WebSocket import socketio, app

def run_websocket():
    app.run(host='localhost', port=5000, debug=False)

def main():
    # 启动 WebSocket 服务器线程
    websocket_thread = threading.Thread(target=run_websocket)
    websocket_thread.daemon = True
    websocket_thread.start()

    # 启动主应用程序
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()