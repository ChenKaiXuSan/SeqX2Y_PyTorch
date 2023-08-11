import sys, os 

current_path = os.getcwd()

try:
    from utils import * 
except:
    sys.path.append(current_path)
    from utils import * 