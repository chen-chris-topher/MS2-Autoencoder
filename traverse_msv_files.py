import os
import sys
import json
import signal
import requests
from ftplib import FTP
import argparse

folders_to_list = ['ccms_peak', 'peak', 'raw', 'ccms_raw']
host = 'massive.ucsd.edu'

def signal_handler(signum, frame):
    raise Exception("Timed out!")

def get_filenames(folder):
    filenames = []
    filenames = ftp.nlst(folder)
    return(filenames)

def get_dirs_ftp(folder):
    contents = ftp.nlst(folder) 
    folders = []
    for item in contents:
        
        #this block narrows search to peak files
        if item.startswith('peak'):
            pass
        elif item.startswith('ccms_peak'):      
            pass
        else:
            continue
        
        if "." not in item:
            folders.append(item)
    return folders

def get_all_dirs_ftp(folder=''):
    dirs = []
    new_dirs = []
    new_dirs = get_dirs_ftp(folder) #call above function
    while len(new_dirs) > 0:
        for dir in new_dirs:
            dirs.append(dir)

        old_dirs = new_dirs[:]
        new_dirs = []
        for dir in old_dirs:
            for new_dir in get_dirs_ftp(dir):
                new_dirs.append(new_dir)

    dirs.sort()
    return dirs

#set a sginal alarm
signal.signal(signal.SIGALRM, signal_handler)
signal.alarm(300)

#open the list of massive ids to look at 
with open('possible_dataset.json', 'r') as f:
    msv_ids = json.load(f)['data']

download_list = []

#loop over all our msv ids 
for msvid in msv_ids:
    print(msvid)
    try:  
        ftp = FTP(host)
        ftp.login()
        ftp.cwd(msvid)  #switch into the massive id directory
    
    except:
        print("Dir change failed")
    
    try:
        all_dirs = get_all_dirs_ftp()   
    except:
        print("All dirs failed")

    files_list = []
    #go through each directory to find the files
    for dir in all_dirs:
        files = get_filenames(dir)  
        
        #keep not None in a master list
        if len(files) != 0:
            for temp_file in files:
                if str(temp_file) != None:
                    files_list.append(os.path.join("ftp://massive.ucsd.edu", msvid, temp_file))
                    download_list.extend(files_list)

with open('all_file_ftp_list.txt', 'w') as fp:
    for item in download_list:
        fp.write("%s\n" % item)
