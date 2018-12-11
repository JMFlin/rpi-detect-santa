# importing os module 
import argparse
import os 
import string
import random

def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# Function to rename multiple files 
def main(args): 
    
    i = 0

    direc = args["path"]
      
    for filename in os.listdir(direc):
        try:
            dst = str(i) + ".jpg"
            src = direc + filename 
            dst = direc + dst 
              
            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 
            i += 1
        except:
            random_id = id_generator()
            dst = str(i) + "_" + random_id + ".jpg"
            src = direc + filename 
            dst = direc + dst 
              
            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 
            i += 1

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--path", required=True,
        help="path images for renaming")
    args = vars(ap.parse_args())

    main(args)

    #directory = os.getcwd() + "\\images"
    #main(directory, folders = ["negative"])