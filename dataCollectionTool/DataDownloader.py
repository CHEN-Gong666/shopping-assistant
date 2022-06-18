import requests
import os

os.chdir(R"/")
targetPath = "data/googleImage/ai-stanford-edu-mugs/"

imageDir = 'http://ai.stanford.edu/~asaxena/robotdatacollection/real/mug/'
imageNameFormat = 'mug.{}.{}'
# three digits with leading zeros
numberFormat = "{:03d}"
postFixList = ["jpg", "txt"]
for postFix in postFixList:
    for i in range(0, 200):
        imageNumber = numberFormat.format(i)
        imageName = imageNameFormat.format(imageNumber, postFix)
        imagePath = imageDir + imageName
        img_data = requests.get(imagePath).content
        targetFile = targetPath + imageName
        with open(targetFile, 'wb') as handler:
            handler.write(img_data)
