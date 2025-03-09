def apply_stylesheet(self):
    dark_theme = """
    QWidget {
        background-color: #20232A;
        color: #FFFFFF;
        font-family: "Arial";
        font-size: 16px;
    }

    QLabel {
        color: #E0E0E0;
    }

    QPushButton {
        background-color: #444;
        color: #FFFFFF;
        border: 1px solid #5C5C5C;
        border-radius: 5px;
        padding: 8px;
    }

    QPushButton:hover {
        background-color: #505357;
    }

    QPushButton:pressed {
        background-color: #606366;
    }

    QLineEdit {
        background-color: #2E3138;
        color: #FFFFFF;
        border: 1px solid #5C5C5C;
        padding: 5px;
        border-radius: 4px;
    }

    QTableWidget {
        background-color: #2E3138;
        color: #FFFFFF;
        border: 1px solid #444;
        gridline-color: #5C5C5C;
        alternate-background-color: #282C34;
    }

    QHeaderView::section {
        background-color: #444;
        color: #E0E0E0;
        border: 1px solid #5C5C5C;
        padding: 4px;
    }

    QListWidget {
        background-color: #2E3138;
        color: #FFFFFF;
        border: 1px solid #444;
        padding: 5px;
    }

    QFrame#detailsFrame {
        background-color: #2E3138;
        border: 2px solid #5C5C5C;
        border-radius: 10px;
        padding: 15px;
    }
    """
    self.setStyleSheet(dark_theme)
