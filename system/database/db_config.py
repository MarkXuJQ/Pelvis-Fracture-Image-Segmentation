# MySQL 数据库配置
db_config = {
    'host': '106.75.62.163',    # NAT 网关的公网 IP
    'user': 'root',             # MySQL 用户名
    'password': '123456!',      # MySQL 密码
    'database': 'pelvis_db',    # 数据库名称
    'port': 3306,              # MySQL 端口号
    'charset': 'utf8mb4',
    'use_unicode': True,       # 使用 Unicode
    'connect_timeout': 10,     # 连接超时设置
    'ssl': {
        'verify_cert': False    # 禁用 SSL 验证
    }
}

