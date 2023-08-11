import sys, os

current_path = os.getcwd()

try:
    from data_loader import *
except:
    sys.path.append(current_path)
    from data_loader import *