import os
import shutil

def sortFileClasses(targetIndexListFile):
    # env
    os.chdir("./ai-stanford-edu-mugs")
    srcFolder = "./mugs_to_sort"
    targetFolder = "./mug_without_logo"
    restFolder = "./mug_with_logo"

    fileNameFormat = "mug.{}.{}"
    indexFormat = "{:03d}"
    suffixList = ['jpg']
    imageNumber = 200

    # get index
    targetIndexList = []
    with open(targetIndexListFile) as file:
        for line in file:
            index = line.rstrip()
            targetIndexList.append(int(index))

    # sort files
    for suffix in suffixList:
        for index in range(0, imageNumber):
            fileIndex = indexFormat.format(index)
            fileName = fileNameFormat.format(fileIndex, suffix)
            filePath = '/'.join([srcFolder, fileName])
            if index in targetIndexList:
                shutil.copy(filePath, targetFolder)
            else:
                shutil.copy(filePath, restFolder)
