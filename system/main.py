import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
#from login_window import LoginWindow
from doctor_window import DoctorUI


def main():
    app = QApplication(sys.argv)
    #login_window = MainWindow() #原来的
    login_window = DoctorUI()  #最新的
    login_window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
