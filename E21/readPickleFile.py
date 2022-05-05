def readData(fileName):
    if fileName[-4:]==".pkl":
        print("Reading distance matrix from .pkl file")
        pklIn=open(fileName,"rb")
        distanceLOL=pickle.load(pklIn)
        pklIn.close()
        return distanceLOL
    else:
        print("Expects a pickled file containing a list of lists for the distance matrix!")
