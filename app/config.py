from decouple import config

SQL_ALCHEMY_CONN=config('SQL_ALCHEMY_CONN')
SQL_POSTGRES_CONN = config('SQL_POSTGRES_CONN')
DB_CONFIG = {
    'dbname': config('DB_NAME'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'host': config('DB_HOST'),
    'port': int(config('DB_PORT'))
}