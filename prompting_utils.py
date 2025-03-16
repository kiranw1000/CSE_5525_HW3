import os, re, utils


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    with open(schema_path, "r") as f:
        schema = f.read()
    return schema

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    result = re.search(r'SELECT(.|\n)*;', response)
    return result.group(0)

def get_schema():
    return utils.compute_record(
        "",
        "SELECT table_name, column_name\nFROM INFORMATION_SCHEMA.COLUMNS\nWHERE table_schema = 'YourDBName'\nORDER BY table_name, ordinal_position"
        )

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")