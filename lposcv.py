def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
from rlscore.learner import RLS
from sklearn.neighbors.regression import KNeighborsRegressor
from rlscore.measure import cindex
from utils import distanceMatrix, lpo_folds, dzfolds, visualize_lposcv, plotRes_lposcv

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 --- LEAVE-PAIR-OUT SPATIAL CROSS VALIDATION FOR RIDGE REGRESSION ---

 DESCRIPTION: 
 - This function will implement leave-pair-out spatial cross validation for ridge
 regression model and produces a performance table with three columns: 
 prediction range, optimal concordance index, optimal regularization parameter.

 INPUT: 
 'coordinates': a n-by-2 array consisting from data point coordinates
 'Xdata': a n-by-m matrix containing predictor data (columns as features)
 'Ydata': a n-by-1 matrix of output values
 'dzradii': list of dead zone radiuses to be used
 'regparams': list of regularization parameters to be tried
 'visualization': boolean on whether visualization wanted or not. If True,
  result in longer calculation
  'saveResultToFile': boolean whether to show analysis results and save them 
  to file.

 OUTPUT: 
 'performanceTable': a r-by-3 matrix containing prediction results. The number
 of rows (r) is determined by the number of dead zone radiuses in 'dzradii'. 
 First corresponds to prediction range, second to concordance index, third to 
 optimal regularization parameter

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def lposcv_rls(coordinates, Xdata, Ydata, dzradii, regparams, visualization, saveResultToFile):
    print("Starting lposcv-rls analysis...")
    # Calculate sorted pairwise distance matrix and indexes
    performanceTable = np.zeros([len(dzradii), 3])
    data_distances, data_distance_indexes = distanceMatrix(coordinates)
    negcount = len(np.where(Ydata==0)[0])
    folds = lpo_folds(Ydata, negcount)
    for rind, dzradius in enumerate(dzradii):
        print("Analysis on going, dead zone radius: " + str(dzradius) + "m / " + str(dzradii[len(dzradii)-1]) + "m")
        # Calculate dead zone folds
        dz_folds = dzfolds(dzradius, folds, data_distances, data_distance_indexes)
        # Initialize performance variables
        best_regparam = None
        best_cindex = 0.5
        for regparam in regparams: # Find best regularization parameter
            perfs = []
            for dz_fold in dz_folds:
                X_tr = np.delete(Xdata, dz_fold, axis=0)
                Y_tr = np.delete(Ydata, dz_fold, axis=0)
                learner = RLS(X_tr, Y_tr, measure=cindex, regparam=regparam)
                preds = learner.predict(Xdata[dz_fold])
                if preds[0] > preds[1]:
                    perfs.append(1.)
                elif preds[0] == preds[1]:
                    perfs.append(0.5)
                else:
                    perfs.append(0.)
                if visualization: # Check for visualization 
                    testcoords = coordinates[dz_fold[0:2], :]
                    dzcoords = coordinates[dz_fold, :]
                    visualize_lposcv(coordinates, testcoords, dzcoords, dzradius)
            perf = np.mean(perfs)
            if perf >= best_cindex:
                best_cindex = perf 
                best_regparam = regparam
        performanceTable[rind, 0] = dzradius
        performanceTable[rind, 1] = best_cindex
        performanceTable[rind, 2] = best_regparam
        if saveResultToFile:
            plotRes_lposcv(performanceTable, rind, len(folds), 'rls')
    print("Analysis done.")
    return performanceTable

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 --- LEAVE-PAIR-OUT SPATIAL CROSS VALIDATION FOR K-NEAREST NEIGHBOR ---

 DESCRIPTION: 
 - This function will implement leave-pair-out spatial cross validation for k-
 nearest neighbor model and produces a performance table with three columns: 
 prediction range, optimal concordance index, optimal regularization parameter.

 INPUT: 
 'coordinates': a n-by-2 array consisting from data point coordinates
 'Xdata': a n-by-m matrix containing predictor data (columns as features)
 'Ydata': a n-by-1 matrix of output values
 'dzradii': list of dead zone radiuses to be used
 'klist': list of k values to be tried
 'visualization': boolean on whether visualization wanted or not. If True,
  result in longer calculation
  'saveResultToFile': boolean whether to show analysis results and save them 
  to file.

 OUTPUT: 
 'performanceTable': a r-by-3 matrix containing prediction results. The number
 of rows (r) is determined by the number of dead zone radiuses in 'dzradii'. 
 First corresponds to prediction range, second to concordance index, third to 
 optimal regularization parameter

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def lposcv_knn(coordinates, Xdata, Ydata, dzradii, klist, visualization, saveResultToFile):
    print("Starting lposcv-knn analysis...")
    # Calculate sorted pairwise distance matrix and indexes
    performanceTable = np.zeros([len(dzradii), 3])
    data_distances, data_distance_indexes = distanceMatrix(coordinates)
    negcount = len(np.where(Ydata==0)[0])
    folds = lpo_folds(Ydata, negcount)
    for rind, dzradius in enumerate(dzradii):
        print("Analysis on going, dead zone radius: " + str(dzradius) + "m / " + str(dzradii[len(dzradii)-1]) + "m")
        # Calculate dead zone folds
        dz_folds = dzfolds(dzradius, folds, data_distances, data_distance_indexes)
        # Initialize performance variables
        best_k = None
        best_cindex = 0.5
        for k_neighbors in klist: # Find best k parameter
            perfs = []
            for dz_fold in dz_folds:
                X_tr = np.delete(Xdata, dz_fold, axis=0)
                Y_tr = np.delete(Ydata, dz_fold, axis=0)
                learner = KNeighborsRegressor(n_neighbors=k_neighbors)
                learner.fit(X_tr, Y_tr)
                preds = learner.predict(Xdata[dz_fold])
                if preds[0] > preds[1]:
                    perfs.append(1.)
                elif preds[0] == preds[1]:
                    perfs.append(0.5)
                else:
                    perfs.append(0.)
                if visualization: # Check for visualization 
                    testcoords = coordinates[dz_fold[0:2], :]
                    dzcoords = coordinates[dz_fold, :]
                    visualize_lposcv(coordinates, testcoords, dzcoords, dzradius)
            perf = np.mean(perfs)
            if perf >= best_cindex:
                best_cindex = perf 
                best_k = k_neighbors
        performanceTable[rind, 0] = dzradius
        performanceTable[rind, 1] = best_cindex
        performanceTable[rind, 2] = best_k
        if saveResultToFile:
            plotRes_lposcv(performanceTable, rind, len(folds), 'knn')
    print("Analysis done.")
    return performanceTable

