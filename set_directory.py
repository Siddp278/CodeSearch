# Setting the project directory and importing the data
# C:\Users\siddp\OneDrive\Desktop\project-venv\codesearch\Scripts

import os
from pathlib import Path
import pandas as pd
import pickle

import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "AbstractSearch"

list_of_files = [
    ".github/workflows/.gitkeep",
    # constructor file needed for packaging this project for deployment.
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/jupyter_notebooks/__init__.py",
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/app.py"
]

for filepath in list_of_files:
    # Handling the / slashes since windows take \ slashes unline linux-EC2
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        # size!=0 means there is content in the file, so do not replace with write operation.
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")


# Importing raw test data
csv_url = "https://raw.githubusercontent.com/kstathou/vector_engine/master/data/misinformation_papers.csv"

try:

    df = pd.read_csv(csv_url, error_bad_lines=False)

    data = pd.DataFrame(df)
    data.reset_index(drop=True, inplace=True)  # Reset index to ensure alignment

    data.to_csv(f"src/{project_name}/data/test_data.csv", index=False)
    logging.info("Data imported for query searching")

except Exception as err:
    logging.warning("Error occurred during processing data")
    print(err)