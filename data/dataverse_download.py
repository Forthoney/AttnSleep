import sys
import urllib.request

import requests

SERVER_URL = "https://researchdata.ntu.edu.sg"

if sys.argv[1] == "20":
    persistent_url = "doi:10.21979/N9/MA1AVG"
    version_no = "1.2"
    file_prefix = "20"
elif sys.argv[1] == "78":
    persistent_url = "doi:10.21979/N9/EUHGHS"
    version_no = "1.1"
    file_prefix = "78"
else:
    persistent_url = "doi:10.21979/N9/EAMYFO"
    version_no = "1.4"
    file_prefix = "shhs"

response = requests.get(
    f"{SERVER_URL}/api/datasets/:persistentId/?persistentId={persistent_url}"
)
response = response.json()

dataset_id = response["data"]["id"]

datainfo = requests.get(
    f"{SERVER_URL}/api/datasets/{dataset_id}/versions/{version_no}/files"
)
datainfo = datainfo.json()

data_ids = [datum["dataFile"]["id"] for datum in datainfo["data"]]
for id in data_ids:
    urllib.request.urlretrieve(
        f"{SERVER_URL}/api/access/datafile/{id}", f"{file_prefix}_{id}"
    )
