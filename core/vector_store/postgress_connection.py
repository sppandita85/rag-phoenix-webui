import psycopg2

def create_connection():
    try: 
        db_connection = psycopg2.connect(
           host="localhost",
           port=5432,
           dbname="poc",
           user="postgres",
           password="password123",
        )
        return db_connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None