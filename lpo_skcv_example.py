
import numpy as np
from skcv import skcv_knn, skcv_rls
from lposcv import lposcv_knn, lposcv_rls

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

 --- EXAMPLE SCRIPT ---

 DESCRIPTION: 
 - In this script we will test and visualize 
both SKCV and LPO-SCV methods with suitable data sets. 
SKCV refers to the spatial K-fold cross-validation method,
whereas LPO-SCV corresponds to the leave-pair-out spatial
cross-validation method.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if __name__=="__main__":
    
    ########################################################
    ## STEP 1: SKCV EXPERIMENT
    ##
    ## - In this part we will implement SKCV-analysis for 
    ## open natural data. The output variable of the analysis
    ## is soil water permeability.  
    ## 
    ########################################################
    # First we load data coordinates
    coordinates_skcv =  np.loadtxt("data_sets/coordinates_skcv.txt", delimiter=",")
    # Predictor features
    Xdata_skcv =  np.loadtxt("data_sets/Xdata_skcv.txt", delimiter=",")
    # Water permeability output variable
    Ydata_skcv =  np.loadtxt("data_sets/Ydata_skcv.txt", delimiter=",")
    # Next we set analysis parameters. We try first ridge regression as the prediction model. 
    regparam = 1. # Regularization parameter 
    dead_zone_radiuses = range(0, 2500, 500) # Produce SKCV for dead zone radiuses 0, 500, ..., 2000 meters. 
    number_of_folds = 10 # Do 10-fold cross-validation
    # If visualization is wanted, set 'visualization = True'. Visualization will result in significantly longer calculation time.  
    visualization = True
    # Run the analysis
    #skcv_rls(coordinates_skcv, Xdata_skcv, Ydata_skcv, number_of_folds, dead_zone_radiuses, regparam, visualization)
    # Next we try the same analysis, but for k-nearest neighbor
    k = 10 # Number of neighbors
    # Run the analysis
    #skcv_knn(coordinates_skcv, Xdata_skcv, Ydata_skcv, number_of_folds, dead_zone_radiuses, k, visualization)
    

    ########################################################
    ## STEP 2: LPO-SCV EXPERIMENT
    ##
    ## - In this part we will implement LPO-SCV-analysis for 
    ## natural data sets. The output variable of the analysis
    ## is soil gold occurrence.  
    ## 
    ########################################################
    # First we load data coordinates
    coordinates_lposcv =  np.loadtxt("data_sets/coordinates_lposcv.txt", delimiter=",")
    # Predictor features
    Xdata_lposcv =  np.loadtxt("data_sets/Xdata_lposcv.txt", delimiter=",")
    Xdata_lposcv = Xdata_lposcv[:,2:]
    # Output variable
    Ydata_lposcv =  np.loadtxt("data_sets/Ydata_lposcv.txt", delimiter=",")
    # Next we set analysis parameters. We try first ridge regression as the prediction model. 
    regparam = 1. # Regularization parameter 
    dead_zone_radiuses = range(0, 30000, 5000) # Produce SKCV for dead zone radiuses 0, 1000, ..., 4000 meters.
    # If visualization is wanted, set 'visualization = True'. Visualization will result in significantly longer calculation time.  
    visualization = False
    # Run the analysis
    lposcv_rls(coordinates_lposcv, Xdata_lposcv, Ydata_lposcv, dead_zone_radiuses, regparam, visualization)
    # Next we try the same analysis, but for k-nearest neighbor
    k = 10 # Number of neighbors
    # Run the analysis
    lposcv_knn(coordinates_lposcv, Xdata_lposcv, Ydata_lposcv, dead_zone_radiuses, k, visualization)
    

