import os
import sys
import torch
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("ROOT_PATH"))

import miso_utils.datasets


print(os.getenv("ROOT_PATH"))
# print(miso_utils.datasets.TestingFunction())
print(torch.cuda.is_available())