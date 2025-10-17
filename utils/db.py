# utils/db.py
import mysql.connector
from mysql.connector import Error
from config import Config

def get_db_connection():
    try:
        # 复制一份参数，避免修改原 Config.DATABASE
        params = dict(Config.DATABASE or {})
        # 只有在未提供时才设置字符集、unicode，避免重复传参导致错误
        params.setdefault('charset', 'utf8mb4')
        params.setdefault('use_unicode', True)
        # 可选：设置连接超时、autocommit 等
        params.setdefault('connect_timeout', 10)

        conn = mysql.connector.connect(**params)
        if conn.is_connected():
            print(f"✅ 成功连接到数据库: {params.get('host')}:{params.get('database')}")
            return conn
        else:
            print("❌ 数据库连接失败")
            return None
    except Error as e:
        print(f"❌ 连接数据库时出错: {e}")
        return None
