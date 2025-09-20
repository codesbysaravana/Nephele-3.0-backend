import sqlite3

DB_NAME = "attendance.db"

def show_attendance():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance")
    rows = cursor.fetchall()

    print("Attendance Records:")
    print("-" * 80)
    for row in rows:
        print(row)

    conn.close()

if __name__ == "__main__":
    show_attendance()
