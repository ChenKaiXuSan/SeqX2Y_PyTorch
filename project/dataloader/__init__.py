import sys, os, pathlib

current_path = pathlib.Path(__file__)

try:
    from data_loader import *
except:
    sys.path.append(str(current_path.parent))
    from data_loader import *