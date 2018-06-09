def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
from rlscore.learner import RLS
from sklearn.neighbors.regression import KNeighborsRegressor
from rlscore.measure import cindex
from rlscore.utilities.cross_validation import random_folds
from utils import distanceMatrix, dzfolds, visualize_skcv, plotRes_skcv

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 --- SPATIAL K-FOLD CROSS VALIDATION FOR RIDGE REGRESSION ---

 DESCRIPTION: 
 - This function will implement spatial k-fold cross validation for ridge
 regression model and produces a performance table with three columns: 
 prediction range, optimal concordance index, optimal regularization parameter.

 INPUT: 
 'coordinates': a n-by-2 array consisting from data point coordinates
 'Xdata': a n-by-m matrix containing predictor data (columns as features)
 'Ydata': a n-by-1 matrix of output values
 'number_of_folds': integer number of cross validation folds
 'dzradii': list of dead zone radiuses to be used
 'regparams': list of regularization parameters to be tried
 'visualization': boolean on whether visualization wanted or not. If True,
  results in longer calculation
  'saveResultToFile': boolean whether to show analysis results and save them 
  to file. 

 OUTPUT: 
 'performanceTable': a r-by-3 matrix containing prediction results. The number
 of rows (r) is determined by the number of dead zone radiuses in 'dzradii'. 
 First corresponds to prediction range, second to concordance index, third to 
 optimal regularization parameter

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def skcv_rls(coordinates, Xdata, Ydata, number_of_folds, dzradii, regparams, visualization, saveResultToFile):
    print("Starting skcv-rls analysis...")
    # Calculate sorted pairwise distance matrix and indexes
    performanceTable = np.zeros([len(dzradii), 3])
    data_distances, data_distance_indexes = distanceMatrix(coordinates)
    folds = random_folds(len(Ydata), number_of_folds)
    for rind, dzradius in enumerate(dzradii): 
        print("Analysis on going, dead zone radius: " + str(dzradius) + "m / " + str(dzradii[len(dzradii)-1]) + "m")
        # Calculate dead zone folds
        dz_folds = dzfolds(dzradius, folds, data_distances, data_distance_indexes)  
        # Initialize performance variables   
        best_regparam = None
        best_cindex = 0.5
        for regparam in regparams: # Find best regularization parameter
            P = np.zeros(Ydata.shape)
            for fold_id, dz_fold in enumerate(dz_folds):
                X_tr = np.delete(Xdata, dz_fold, axis=0)
                Y_tr = np.delete(Ydata, dz_fold, axis=0)
                learner = RLS(X_tr, Y_tr, measure=cindex, regparam=regparam)
                preds = learner.predict(Xdata[dz_fold])
                if preds.ndim == 0:
                    P[folds[fold_id]] = preds         
                else:
                    P[folds[fold_id]] = preds[0:len(folds[fold_id])]
                if visualization: # Check for visualization 
                    testcoords = coordinates[folds[fold_id],:]
                    dzcoords = coordinates[dz_fold, :]
                    visualize_skcv(coordinates, testcoords, dzcoords, dzradius)
            perf = cindex(Ydata, P)
            if perf >= best_cindex:
                best_cindex = perf 
                best_regparam = regparam
        performanceTable[rind, 0] = dzradius
        performanceTable[rind, 1] = best_cindex
        performanceTable[rind, 2] = best_regparam
        if saveResultToFile:
            plotRes_skcv(performanceTable, rind, number_of_folds, "rls")
    print("Analysis done.")
    return performanceTable
        
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 --- SPATIAL K-FOLD CROSS VALIDATION FOR K-NEAREST NEIGHBOR ---

 DESCRIPTION: 
 - This function will implement spatial k-fold cross validation for k-nearest
 neighbor model and produces a performance table with three columns: 
 prediction range, optimal concordance index, optimal regularization parameter.

 INPUT: 
 'coordinates': a n-by-2 array consisting from data point coordinates
 'Xdata': a n-by-m matrix containing predictor data (columns as features)
 'Ydata': a n-by-1 matrix of output values
 'number_of_folds': integer number of cross validation folds
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
 optimal number of neighbors 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def skcv_knn(coordinates, Xdata, Ydata, number_of_folds, dzradii, klist, visualization, saveResultToFile):
    print("Starting skcv-knn analysis...")
    # Calculate sorted pairwise distance matrix and indexes
    performanceTable = np.zeros([len(dzradii), 3])
    data_distances, data_distance_indexes = distanceMatrix(coordinates)
    folds = random_folds(len(Ydata), number_of_folds)
    for rind, dzradius in enumerate(dzradii):
        print("Analysis on going, dead zone radius: " + str(dzradius) + "m / " + str(dzradii[len(dzradii)-1]) + "m")
        # Calculate dead zone folds
        dz_folds = dzfolds(dzradius, folds, data_distances, data_distance_indexes)
        # Initialize performance variables   
        best_k = None
        best_cindex = 0.5
        for k_neighbors in klist: # Find best k parameter
            P = np.zeros(Ydata.shape)
            for fold_id, dz_fold in enumerate(dz_folds):
                X_tr = np.delete(Xdata, dz_fold, axis=0)
                Y_tr = np.delete(Ydata, dz_fold, axis=0)
                learner = KNeighborsRegressor(n_neighbors=k_neighbors)
                learner.fit(X_tr, Y_tr)
                preds = learner.predict(Xdata[dz_fold])
                if preds.ndim == 0:
                    P[folds[fold_id]] = preds         
                else:
                    P[folds[fold_id]] = preds[0:len(folds[fold_id])]
                if visualization: # Check for visualization
                    testcoords = coordinates[folds[fold_id],:]
                    dzcoords = coordinates[dz_fold, :]
                    visualize_skcv(coordinates, testcoords, dzcoords, dzradius)                
            perf = cindex(Ydata, P)
            if perf >= best_cindex:
                    best_cindex = perf 
                    best_k = k_neighbors
        performanceTable[rind, 0] = dzradius
        performanceTable[rind, 1] = best_cindex
        performanceTable[rind, 2] = best_k
        if saveResultToFile:
            plotRes_skcv(performanceTable, rind, number_of_folds, 'knn')
    print("Analysis done.")
    return performanceTable