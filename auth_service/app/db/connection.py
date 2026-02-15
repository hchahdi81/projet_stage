import snowflake.connector

def get_connection():
    conn = snowflake.connector.connect(
        user='HATIM',
        password='@Wata@2000',
        account='epocmeg-pgwebmobile',
        warehouse='MEDICAL_PRJ_WAREHOUSE',
        database='medical_project',
        schema='medical_data'
    )
    return conn
