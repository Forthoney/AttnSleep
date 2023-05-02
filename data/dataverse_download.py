import re
import sys

import requests
from tqdm import tqdm

SERVER_URL = "https://researchdata.ntu.edu.sg"

if sys.argv[1] == "20":
    persistent_url = "doi:10.21979/N9/MA1AVG"
    version_no = "1.2"
    file_prefix = "20"
    folder = "edf20"
elif sys.argv[1] == "78":
    persistent_url = "doi:10.21979/N9/EUHGHS"
    version_no = "1.1"
    file_prefix = "78"
    folder = "edf78"
else:
    persistent_url = "doi:10.21979/N9/EAMYFO"
    version_no = "1.4"
    file_prefix = "shhs"
    folder = "shhs"

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
for id in tqdm(data_ids):
    response = requests.get(f"{SERVER_URL}/api/access/datafile/{id}")
    content_disposition = response.headers.get("Content-Disposition")
    filename = re.search(r'filename="(.+)"', content_disposition).group(1)
    with open(filename, "wb+") as f:
        f.write(response.content)
