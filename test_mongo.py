from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()
url = os.getenv("MONGO_DB_URL")
# print(url)

# create a client and connect to the server
client = MongoClient(url)

# Send a ping to confirm a successful connection 

try:
    client.admin.command('ping')
    print("Ping your deployment, You successfully connected to MongoDB")
except Exception as e:
    print(e)