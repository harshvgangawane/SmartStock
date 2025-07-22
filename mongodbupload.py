import pandas as pd
from pymongo import MongoClient

csv_file = 'data/retail_store_inventory.csv'  
df = pd.read_csv(csv_file)


print("Sample data:\n", df.head())

client = MongoClient("mongodb://localhost:27017/")  
db = client['retailDB']                             
collection = db['inventory']                        


data_dict = df.to_dict("records")                   
insert_result = collection.insert_many(data_dict)  


print(f"{len(insert_result.inserted_ids)} records inserted into MongoDB collection '{collection.name}'")


