import urllib.request
import os

os.system("unrar x Roms.rar")
os.system("mkdir rars")
os.system("mv HC\ROMS.zip   rars")
os.system("mv ROMS.zip  rars")
os.system("python -m atari_py.import_roms rars")