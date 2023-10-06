import io
import os 
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account


class DriveConnection:   
    """
    pass

    """
    current_folder_id = None
    _drive_connection_obj = None

    def __init__(self, order_number):
        self.order_number = order_number
        try:
            
            creds = service_account.Credentials.from_service_account_file(
                "/root/GoogleDrive/gclouds.json"
            )

            scoped_credentials = creds.with_scopes(
                ["https://www.googleapis.com/auth/drive"]
            )

            self.google_service = build("drive", "v3", credentials=scoped_credentials)
            self._drive_connection_obj = self.google_service.files()
        except HttpError as error:
            print(f"An error occurred in connection: {error}")

    def download_files(self,folder_path, folder_id):
        """
        pass
        
        """
        for file_folder in self.__get_all_files_from_folder(top_folder_id=folder_id):
            if file_folder['name'] == self.order_number:
                file_details =self.__get_all_files_from_folder(top_folder_id=file_folder['id'])
                self.current_folder_id = file_folder['id']
                for file in file_details:
                    self.__download_files_from_folder(file['id'],folder_path)

    def upload(self, folder_path):
        """
        pass
        """
        folder_id = self.__create_folder()

        dir_path = folder_path

        for file in os.listdir(dir_path):
            if file.endswith(".png"):
                file_metadata = {'name': file,'parents': [folder_id] }
                print(dir_path+file)
                media = MediaFileUpload(dir_path+file,
                                        mimetype='image/jpeg')
                # pylint: disable=maybe-no-member
                file = self._drive_connection_obj.create(body=file_metadata, media_body=media,
                                            fields='id').execute()
                print(file)
        # print("#"*25, "Moving Folder",  "#"*25)
        # self.change_folder()

    def __create_folder(self):
        """
        pass
        """
        file_meta = {
            'name': self.order_number,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': ['1_gjuYpYHnyWxnYZI023QY_7eBKkIGnX6']
        }

        folder_details = self._drive_connection_obj.create(body= file_meta).execute()

        return folder_details['id']

    def __get_all_files_from_folder(self, top_folder_id):
        page_token = None
        response = self._drive_connection_obj.list(q="'" + top_folder_id + "' in parents", pageSize=1000,
        pageToken=page_token, fields="nextPageToken, files(id, name)").execute()
        page_token = response.get('nextPageToken')
        return response['files']


    def __download_files_from_folder(self, file_id, folder_path):
        """
        pass
        """
        file = self._drive_connection_obj.get(fileId=file_id).execute()

        file_content = io.BytesIO()

        # Download the file
        downloader = MediaIoBaseDownload(file_content, self._drive_connection_obj.get_media(fileId=file_id))
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Downloading {file['name']}  {int(status.progress() * 100)}%.")
            
        # Save the downloaded file to disk
        with open(folder_path+file['name'], 'wb') as f:
            f.write(file_content.getbuffer())

    def change_folder(self):
        """
        pass
        """
        # Retrieve the folder's current parents
        folder = self._drive_connection_obj.get(fileId=self.current_folder_id, fields='parents').execute()
        previous_parents = ",".join(folder.get('parents'))

        # Move the folder to the new location
        folder = self._drive_connection_obj.update(fileId=self.current_folder_id, addParents='15RX4a2cFaO_qkHk7ZykUBgCuhuTJIhxy', removeParents=previous_parents, fields='id, parents').execute()

        print("FolderMoved")