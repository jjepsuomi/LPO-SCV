
import numpy as np
import sklearn.metrics.pairwise as pw
import matplotlib.pyplot as plt
import random

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 --- CALCULATION OF LEAVE-PAIR-OUT FOLDS ---

 DESCRIPTION: 
 - This function will produce fold set for leave-pair-out cross-validation

 INPUT: 
 'Ydata': a n-by-1 matrix of output values
 'neg_samples': integer specifying the number of data points with 'negative'
 label. This function is needed by the LPO-SCV method. 

 OUTPUT: 
 'lpo_folds': list of leave-pair-out folds

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def lpo_folds(Ydata, neg_samples):
    posind = np.where(Ydata>0)[0]
    negind = np.where(Ydata==0)[0]
    lpo_folds = []
    for i in posind:
        negsample = random.sample(list(negind), neg_samples)
        for j in range(neg_samples):
            fold = [i, negsample[j]]
            lpo_folds.append(fold)
    return lpo_folds

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 --- USED FOR CALCULATING SORTED DISTANCE MATRIX ---

 DESCRIPTION: 
 - This function will calculate all pairwise geographical distances between 
 the data points and returns a sorted matrix consisting from the distances 
 and additional matrix containing corresponding data point indices. This 
 function is needed by the SKCV method. 

 INPUT: 
 'coordinates': a n-by-2 array consisting from data point coordinates

 OUTPUT: 
 'data_distances': a sorted n-by-n matrix containing all pairwise distances 
 'data_distance_indexes': a corresponding n-by-n matrix of data point indexes

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def distanceMatrix(coordinates):
    
    number_of_data_points = coordinates.shape[0]
    data_distances = np.float32(pw.euclidean_distances(coordinates, coordinates))
    data_distance_indexes = np.int32(np.zeros([number_of_data_points, number_of_data_points], dtype=np.int))
    index_m = np.array(range(0, number_of_data_points, 1))
    for i in range(0, number_of_data_points):
        sorted_m = np.transpose(np.array([data_distances[:,i], index_m]))
        sorted_m = sorted_m[sorted_m[:, 0].argsort()]
        data_distances[:,i] = sorted_m[:,0]
        data_distance_indexes[:,i] = sorted_m[:,1]  
    return data_distances, data_distance_indexes

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 --- USED FOR CALCULATING DEAD ZONE FOLDS FOR A GIVEN RADIUS r ---

 DESCRIPTION: 
 - This function will produce dead zone folds used in SKCV given some radius
 r by including into the provided previosuly calculated folds the indices of
 data points which are "too close" (distance <= r) to the test data points. 
 These folds are used when forming the reduced training data sets. Reduced 
 in the sense that they do not contain data points too near to test points

 INPUT: 
 'r': dead zone radius
 'folds': array of cross-validation folds to be updated
 'data_distances': n-by-n matrix of pairwise distances
 'data_distance_indexes': corresponding matrix of indices to previous input

 OUTPUT: 
 'dz_folds': updated array of 'folds' that includes data points too close to 
 test data points

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def dzfolds(r, folds, data_distances, data_distance_indexes):
    dz_folds = list()
    # We loop through every previously calculated fold
    for fold in folds:
        tooCloseList = list()
        # Find data points too close to test data points
        for i in fold:
            closeInds = np.where(data_distances[:,i] <= r)[0]
            tooCloseList = list(set(tooCloseList).union(data_distance_indexes[closeInds,i]))       
        dzfold = fold[:]   
        # In addition to test data points, include into dzfold the indices of "too close" data
        for j in tooCloseList:
            if j not in dzfold:
                dzfold.append(j)
        dz_folds.append(dzfold)
    return dz_folds

"""""""""""""""""""""""""""""""""""""""

--- VISUALIZATION OF SKCV PROCEDURE ---

"""""""""""""""""""""""""""""""""""""""
def visualize_skcv(coords, testcoords, dzcoords, r):   
    plt.figure(0)
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+0+0")
    plt.clf()
    plt.scatter(coords[:,0], coords[:,1], c='purple')
    for t in range(0, testcoords.shape[0]):
        circle1=plt.Circle((testcoords[t,0], testcoords[t,1]), r, color='orange', fill=True, alpha=0.2)
        fig = plt.gcf()
        fig.gca().add_artist(circle1)
    plt.scatter(dzcoords[:,0], dzcoords[:,1], c='yellow')
    plt.scatter(testcoords[:,0], testcoords[:,1], c='red')
    plt.xlabel("EUREF-TM35FIN E")
    plt.ylabel("EUREF-TM35FIN N")
    plt.legend(['Training data', 'Omitted data', 'Test data'])
    plt.title("SKCV PROCEDURE")
    plt.axis('equal')
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.draw()
    plt.pause(0.1)
    
""""""""""""""""""""""""""""""""""""""""

 --- PLOT AND SAVE SKCV RESULTS ---

"""""""""""""""""""""""""""""""""""""""
def plotRes_skcv(performanceTable, ind, nfolds, method):
    plt.figure(1)
    mngr = plt.get_current_fig_manager()
    window = plt.get_current_fig_manager().window
    screen_x, screen_y = window.wm_maxsize()
    mngr.window.wm_geometry("+"+str(int(screen_x/float(2)))+"+0")
    plt.clf()
    plt.plot(performanceTable[range(0, ind), 0], performanceTable[range(0, ind), 1], c='blue')
    plt.plot(performanceTable[range(0, ind), 0], performanceTable[range(0, ind), 1], 'go')
    plt.grid()
    pcal = np.around(ind/float(performanceTable.shape[0]-1)*100, 1)
    plt.title(str(nfolds) + "-fold SKCV C-index performance graph (" + str(method) + ")")
    plt.xlabel("Dead zone radius (meters)")
    plt.ylabel("C-index")
    plt.draw()
    plt.pause(0.1)
    if pcal == 100:
        plt.savefig(str(nfolds) + '_Fold_SKCV_Results_' + method + '.pdf')
        
"""""""""""""""""""""""""""""""""""""""""""""

--- VISUALIZATION OF LPO-SCV PROCEDURE ---

"""""""""""""""""""""""""""""""""""""""""""""
def visualize_lposcv(coords, testcoords, dzcoords, r):   
    plt.figure(0)
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+0+0")
    plt.clf()
    plt.scatter(coords[:,0], coords[:,1], c='purple')
    circle1=plt.Circle((testcoords[0,0], testcoords[0,1]), r, color='orange', fill=True, alpha=0.2)
    circle2=plt.Circle((testcoords[1,0], testcoords[1,1]), r, color='orange', fill=True, alpha=0.2)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)
    plt.scatter(dzcoords[:,0], dzcoords[:,1], c='yellow')
    plt.scatter(testcoords[0,0], testcoords[0,1], c='red')
    plt.scatter(testcoords[1,0], testcoords[1,1], c='blue')
    plt.title("LPO-SCV PROCEDURE")
    plt.xlabel("EUREF-TM35FIN E")
    plt.ylabel("EUREF-TM35FIN N")
    plt.legend(['Training data', 'Omitted data', 'Test (+) data', 'Test (-) data'])
    plt.axis('equal')
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.draw()
    plt.pause(0.1)
    
""""""""""""""""""""""""""""""""""""""""

 --- PLOT AND SAVE LPO-SCV RESULTS ---

"""""""""""""""""""""""""""""""""""""""
def plotRes_lposcv(performanceTable, ind, nfolds, method):
    plt.figure(1)
    mngr = plt.get_current_fig_manager()
    window = plt.get_current_fig_manager().window
    screen_x, screen_y = window.wm_maxsize()
    mngr.window.wm_geometry("+"+str(int(screen_x/float(2)))+"+0")
    plt.clf()
    plt.plot(performanceTable[range(0, ind), 0], performanceTable[range(0, ind), 1], c='blue')
    plt.plot(performanceTable[range(0, ind), 0], performanceTable[range(0, ind), 1], 'go')
    plt.grid()
    pcal = np.around(ind/float(performanceTable.shape[0]-1)*100, 1)
    plt.title(str(nfolds) + "-fold LPO-SCV C-index performance graph (" + str(method) + ")")
    plt.xlabel("Dead zone radius (meters)")
    plt.ylabel("C-index")
    plt.draw()
    plt.pause(0.1)
    if pcal == 100:
        plt.savefig(str(nfolds) + '_Fold_LPO-SCV_Results_' + method + '.pdf')