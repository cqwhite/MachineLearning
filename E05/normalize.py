from operator import delitem
from tkinter.font import names
import pandas as pd
import numpy as np
import csv
import io

# x=the data u=mean of data s=range
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def scaleAndNormilizeFile(file):
    outFileName = "E05/FishersIrisStandardHeaderOutFile.txt"
    rowsToSkip = 6
    iterator = 0
    topDataFile = open(file, "r")
    topFileData = ""

    #Get the first 6 lines of the data file
    for line in topDataFile:
        if iterator <= 5:
            topFileData = (topFileData) + (line)
        if iterator > 5:
            break
        iterator = iterator + 1
    #get the columnNames for the dataFile
    columnNames = pd.read_csv(file, skiprows=4, nrows=0)
    #opne the data file as a data frame and normalize the dataFile features
    df = pd.read_csv(file, skiprows=rowsToSkip,
                     names=columnNames.columns, delimiter=r"[,\s+|,|\s+]")

    df = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))
                  if x.name != "Sample Number" and x.name != "Price" and x.name != "Classification" else x)
    print(df)

    #combine the datafile top and data section
    outputData = topFileData + str(df)
    outputData = outputData.split("\n")

    #output the new file
    skipIterator = 0
    with open(outFileName, 'w') as file:
        for line in outputData:
            if skipIterator == 6:
                print("Skipped")
            else:
                file.write(line)
                file.write("\n")
            skipIterator = skipIterator + 1


scaleAndNormilizeFile("E05/FishersIrisStandardHeader.txt")
