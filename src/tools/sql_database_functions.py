import mysql.connector
import pandas as pd
import time
import mysql.connector

def check_table_exists(db_name, table_name):
    conn = mysql.connector.connect(user='yourusername', password='yourpassword', host='localhost', database=db_name)
    cursor = conn.cursor()
    cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
    result = cursor.fetchone()
    conn.close()
    return result is not None

def get_table_names(database):
	cnx = mysql.connector.connect(user='Dante', password='1Tsnufink+', host='localhost', database=database)
	mycursor = cnx.cursor()

	mycursor.execute("SHOW TABLES")
	tables = mycursor.fetchall()
	return tables

def add_to_database(data, tablename, database, value_column = 'value'):
	# Step 1: Connect to the MySQL database

	# Check if the database exists, if not, create it
	cnx = mysql.connector.connect(user='Dante', password='1Tsnufink+', host='localhost')
	mycursor = cnx.cursor()
	mycursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
	cnx.database = database

	# Step 2: Drop the table if it exists
	mycursor.execute(f"DROP TABLE IF EXISTS `{tablename}`")

	# Step 3: Create the table with initial columns
	# Assuming 'Hour' and 'value' are initial columns, adjust as needed
	mycursor.execute(f"CREATE TABLE `{tablename}` (ID DOUBLE)")

	# Step 4: Add columns from the DataFrame to the table
	columns = data.columns.tolist()
	
	# Ensure column names are unique
	unique_columns = []
	seen = set()
	for col in columns:
		if col.lower() in seen:
			count = 1
			new_col = f"{col}_{count}"
			while new_col.lower() in seen:
				count += 1
				new_col = f"{col}_{count}"
			unique_columns.append(new_col)
			seen.add(new_col.lower())
		else:
			unique_columns.append(col)
			seen.add(col.lower())
	
	data.columns = unique_columns

	# Determine the maximum length of each column's data
	max_lengths = data.astype(str).applymap(len).max()

    # Increase the length to avoid truncation
	max_lengths = max_lengths.apply(lambda x: x + 10)

	alter_table_sql = f"ALTER TABLE `{tablename}` "
	alter_table_sql += ", ".join([f"ADD COLUMN `{col}` VARCHAR({max_lengths[col]})" for col in unique_columns if col != 'ID'])
	mycursor.execute(alter_table_sql)

    # Step 5: Insert the data from the DataFrame into the table
	cols = "`,`".join([str(i) for i in unique_columns])
	insert_sql = f"INSERT INTO `{tablename}` (`" + cols + "`) VALUES (" + "%s,"*(len(unique_columns)-1) + "%s)"
	for i, row in data.iterrows():
		mycursor.execute(insert_sql, tuple(row))

	# Step 6: Commit the transaction and close the connection
	cnx.commit()
	mycursor.close()
	cnx.close()

def add_individual_database(data, tablename, database):
	# Step 1: Connect to the MySQL database
	cnx = mysql.connector.connect(user='Dante', password='1Tsnufink+', host='localhost', database=database)
	mycursor = cnx.cursor()

	# Step 2: Drop the table if it exists
	mycursor.execute(f"DROP TABLE IF EXISTS `{tablename}`")

	# Step 3: Create the table with initial columns
	# Assuming 'Hour' and 'value' are initial columns, adjust as needed
	mycursor.execute(f"CREATE TABLE `{tablename}` (Hour INT)")

	# Step 4: Add columns from the DataFrame to the table
	columns = data.columns.tolist()
	alter_table_sql = f"ALTER TABLE `{tablename}` "
	alter_table_sql += ", ".join([f"ADD COLUMN `{col}` VARCHAR(255)" for col in columns ])
	mycursor.execute(alter_table_sql)

	# Step 5: Insert the data from the DataFrame into the table
	cols = "`,`".join([str(i) for i in data.columns.tolist()])
	insert_sql = f"INSERT INTO `{tablename}` (`" + cols + "`) VALUES (" + "%s,"*(len(data.columns)-1) + "%s)"
	for i, row in data.iterrows():
		mycursor.execute(insert_sql, tuple(row))

	# Step 6: Commit the transaction and close the connection
	cnx.commit()
	mycursor.close()
	cnx.close()

def extract_table_to_dataframe(tablename, database):
    # Step 1: Connect to the MySQL database
    cnx = mysql.connector.connect(user='Dante', password='1Tsnufink+', host='localhost', database=database)
    mycursor = cnx.cursor()

    # Step 2: Execute the SELECT query to retrieve the table data
    mycursor.execute(f"SELECT * FROM `{tablename}`")
    rows = mycursor.fetchall()

    # Step 3: Get the column names
    column_names = [i[0] for i in mycursor.description]

    # Step 4: Convert the result into a DataFrame
    df = pd.DataFrame(rows, columns=column_names)

    # Step 5: Close the connection
    mycursor.close()
    cnx.close()
    return df

if __name__ == '__main__':
	tablename = 'nodal_coordinates'
	data = pd.read_csv(r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\TJ Sectorial Model\Data Prepration\Topology\nodal_coordinates.csv')	
	data.fillna(0, inplace=True)
	database = 'joule_model'
	add_to_database(data, tablename, database)
