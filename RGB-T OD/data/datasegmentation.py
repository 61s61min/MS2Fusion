import json
import os
import cv2
import shutil
with open('/home/shen2/dataset_zhb/M3FD_Detection/instances_val2014.json','r',encoding='utf-8') as files:
    data = json.load(files)
# print(data['images'])
imagefile= '/home/shen2/dataset_zhb/M3FD_Detection'
save='/home/shen2/dataset_zhb/M3FD_Detection'
file = data['images']
for i in file:
    irimagefilename=os.path.join(imagefile,'ir',i['file_name'])
    irsavefilename = os.path.join(save,'infrared/test',i['file_name'])
    visimagefilename = os.path.join(imagefile,'vi',i['file_name'])
    vissavefilename = os.path.join(save,'visible/test',i['file_name'])
    label = os.path.join(imagefile,'labels',i['file_name'].split('.')[0]+'.txt')
    labelsave=os.path.join(imagefile,'labels/test',i['file_name'].split('.')[0]+'.txt')
    # irimg = cv2.imread(irimagefilename)
    # visimg = cv2.imwrite(visimagefilename)
    shutil.copy(irimagefilename,irsavefilename)
    shutil.copy(visimagefilename,vissavefilename)
    shutil.copy(label,labelsave)


    # print(irimagefilename)
    # print(irsavefilename)
