import os

from natsort import natsorted

def readfileslist(pathtofiles, fileext):
    if not os.path.exists(pathtofiles):
        raise Exception('Path does not exist!')

    lstFiles = []
    for dirName, _, filelist in os.walk(pathtofiles):
        for filename in natsorted(filelist):
            if fileext in filename.lower():
                lstFiles.append(os.path.join(dirName, filename))
    return lstFiles        
