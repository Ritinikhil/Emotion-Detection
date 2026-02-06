

import sys, os, glob
root = sys.path[0]
print("PROJECT FOLDER:", root)
print("Local mediapipe* files:", glob.glob(os.path.join(root, "mediapipe*")))