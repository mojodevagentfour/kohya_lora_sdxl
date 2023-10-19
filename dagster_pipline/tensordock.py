import subprocess
import requests
import os
import json
import time

# TensorDock Core Cloud API Documentation: https://documenter.getpostman.com/view/10732984/UVC3j7Kz#intro

class TensorDock:
    def __init__(self) -> None:
        # List to store the instances deployed
        self.instances = []
        # Dictionary to track the running instances
        self.running = {}

    def deploy_server(self):
        """
        Deploys servers on TensorDock based on the count provided.

        Parameters:
        - count (int): Number of servers to be deployed

        Returns:
        - list: List of instance details
        """
        deploy_url = "https://console.tensordock.com/api/deploy/single/custom"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        deploy_payload = {
            "api_key": "recabQkXrkbei2crf",
            "api_token": "FCACNCSdkGHeVJOommokqIJlNnCOFO",
            "admin_user": "user",
            "admin_pass": "test@1pass",
            "instance_type": "gpu",
            "gpu_model": "A4000",
            "gpu_count": 1,
            "vcpus": 6,
            "ram": 16,
            "storage": 50,
            "storage_class": "io1",
            "os": "recvcExWca1Pac7V3",
            "location": "na-us-chi-1",
            "name": "ray_worker",
            """cloud_init""": """runcmd:
            - sudo docker login -u mojocreator -p dckr_pat_gYXAFOgplKc7Rgrz5p8Qk2KsHhE
            """,
        }   #- sudo docker run --name my_container -d mojocreator/sdxl:0.1.1 tail -f /dev/null
            # - '#!/bin/bash'
            # - set -e
            # - git clone https://github.com/mojodevagentfour/uploader_form.git /home/user/uploader_form
            # - sudo chmod 777 /home/user/uploader_form
            # - sudo apt-get update && sudo apt-get install -y git curl ffmpeg libsm6 libxext6 python3-pip
            # - sudo /usr/bin/pip3 install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
        deploy_req = requests.post(deploy_url, headers=headers, data=deploy_payload)
        print(deploy_req.json(),"+===+++===="*10)
        self.id = deploy_req.json()["server"]["id"]
        self.ip = deploy_req.json()["server"]["ip"]
        self.status = deploy_req.json()["success"]
        return self.id, self.ip, self.status

    def wait_until_deployed(self, _id, _ip):
        """
        Continuously checks if the server is in a running state.
        """

        self.id = _id
        self.ip = _ip
        print(self.id, self.ip, "+++====+++===")

        url = "https://console.tensordock.com/api/deploy/status"

        payload = {
            "api_key": "recabQkXrkbei2crf",
            "api_token": "FCACNCSdkGHeVJOommokqIJlNnCOFO",
            "server": self.id,
        }

        response = requests.request("POST", url, data=payload)
        print(response.json(), "+==+++++===+++")
        if response.json()["status"] == "Success Or Started":
            self.running[self.id] = "success"
        else:
            time.sleep(15)
            self.wait_until_deployed(self.id, self.ip)

        return self.running

    def delete_server(self, server_id):
        """
        Deletes the servers that are successfully running.
        """
        url = f"https://console.tensordock.com/api/delete/single?api_key=recabQkXrkbei2crf&api_token=FCACNCSdkGHeVJOommokqIJlNnCOFO&server={server_id}"
        response = requests.request("GET", url=url)
        return response.json()
