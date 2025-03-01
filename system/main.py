import sys
from PyQt5.QtWidgets import QApplication
from doctor_window import DoctorUI
import threading
import socketio
from WebSocket import socketio, app  # 导入 WebSocket 服务器

def run_websocket():
    """在单独的线程中运行 WebSocket 服务器"""
    app.run(host='localhost', port=5000, debug=False)

def main():
    # 启动 WebSocket 服务器线程
    websocket_thread = threading.Thread(target=run_websocket)
    websocket_thread.daemon = True  # 设置为守护线程，这样主程序退出时会自动结束
    websocket_thread.start()

    # 启动主应用程序
    app = QApplication(sys.argv)
    login_window = DoctorUI()
    login_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

