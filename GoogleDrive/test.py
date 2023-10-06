import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Replace with the path to your service account key file
SERVICE_ACCOUNT_FILE = '/home/lnv-242/projects/AI/petpotrait/pet-potrairt-docker/pet-potrait-docker/GoogleDrive/gclouds.json'

# Replace with the ID of the folder you want to move
FOLDER_ID = '18WSJyIP-auVmau6x2GxW7_3XR8YoWARM'

# Replace with the ID of the destination folder
DESTINATION_FOLDER_ID = '1t5X6QGw2eadVUgQqjAviMAsUz1xHyiQ8'

# Authenticate using the service account key file
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
service = build('drive', 'v3', credentials=creds)

# Retrieve the folder's current parents
folder = service.files().get(fileId=FOLDER_ID, fields='parents').execute()
previous_parents = ",".join(folder.get('parents'))

# Move the folder to the new location
folder = service.files().update(fileId=FOLDER_ID, addParents=DESTINATION_FOLDER_ID, removeParents=previous_parents, fields='id, parents').execute()

print(f"Folder with ID '{FOLDER_ID}' has been moved to '{folder['parents']}'")