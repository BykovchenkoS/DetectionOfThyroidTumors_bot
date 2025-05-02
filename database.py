import logging
import pymysql
from config import host, user, password, db_name


class MySQLDatabase:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=host,
                port=3306,
                user=user,
                password=password,
                database=db_name,
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True
            )
            logging.info("Connected to MySQL successfully")
            return True
        except pymysql.Error as error:
            logging.critical(f"MySQL connection failed: {error}")
            raise

    def is_connected(self):
        try:
            if self.connection and self.connection.open:
                self.connection.ping(reconnect=False)
                return True
            return False
        except pymysql.Error:
            return False

    def execute_query(self, query, args=None, fetch=False):
        if not self.is_connected():
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, args or ())

                if fetch:
                    result = cursor.fetchall()
                    return result if result else None

                self.connection.commit()
                return cursor.lastrowid

        except pymysql.Error as error:
            logging.error(f"Query failed: {error}\nQuery: {query}")
            raise

    def fetch_one(self, query, args=None):
        result = self.execute_query(query, args, fetch=True)
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return None

    def close(self):
        if self.is_connected():
            self.connection.close()
            logging.info("MySQL connection closed")


db = MySQLDatabase()


if __name__ == "__main__":
    db = MySQLDatabase()
    try:
        print("Проверяем соединение...")
        print("Статус:", "активно" if db.is_connected() else "не активно")

    finally:
        db.close()
