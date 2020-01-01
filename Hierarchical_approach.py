####################################################################################
### Written and Updated by Prashant Shekhar on date: 27 December 2019
####################################################################################

### Importing the libraries
import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy import random, linalg, stats



def Hierarchical_learn_basic(Data_temp,err):
    ## Input:
    ## 1: Data_temp: Data matrix in XY (for univariate functions) or XYZ (for bivariate functions).
    ## 2: err: The error tolerance provided by the user.
    
    ## Output:
    ## 1: (x_s, y_s, z_s): Sparse representation coordinates
    ## 2: S_conv: Convergence Scale
    ## 3: coor: Coordinate of projection on the chosen sparse bases
    ## 4: epsilon: Length scale of the squared exponential kernel
    
    ## Formatting the Data matrix to be in 3D
    if Data_temp.shape[1] == 2:
        Data = np.zeros([Data_temp.shape[0],3])
        Data[:,0] = Data_temp[:,0]
        Data[:,2] = Data_temp[:,1]
    
    if Data_temp.shape[1] == 3:
        Data = Data_temp
        
    ## Initializations
    points = Data.shape[0]
    s=0;
    X = Data[:,0]
    Y = Data[:,1]
    H = Data[:,2]
    P1 = 2
    l_s = 0
    
    ## Computing T
    dist = squareform(pdist(Data[:,0:2], 'euclidean'))
    max_d = np.amax(dist)
    T = 2*(max_d/2)**2;

    ## Starting the main loop
    while l_s<points:
        epsilon = T/(P1**s)
        Gaussian = np.exp(-((dist**2)/epsilon))

        ## Finding the rank of the Gaussian Kernel
        l_s = LA.matrix_rank(Gaussian)
        k = l_s+8

        ## Calculating the W matrix
        A = random.randn(k,points)
        W = A.dot(Gaussian)
        W = np.matrix(W) 

        ## Pivoted QR decomposition for W
        Q,R,P = linalg.qr(W, mode='full', pivoting = True)
        Perm = np.zeros([len(P),len(P)],dtype=int)
        for w in range(0,len(P)):
            Perm[P[w],w] = 1


        ## Selecting Relevant columns of Kernel into B Matrix
        Mat = Gaussian.dot(Perm)
        B = Mat[:,0:l_s];

        ## Coordinate vector of orthogonal projection
        coor = np.linalg.lstsq(B, H,rcond=None)[0]

        ## Orthogonal Projections on columns of B
        f = B.dot(coor)

        ## Getting the sparse representation
        vec = np.empty([1,points])
        for w in range(0,points):
            vec[0,w] = w
        permv = vec.dot(Perm)
        permv = permv.T
        x_s = np.empty([1,l_s])
        y_s = np.empty([1,l_s])
        z_s = np.empty([1,l_s])
        for i in range(l_s):
            coor1 = int(permv[i])
            x_s[0,i] = X[coor1]
            y_s[0,i] = Y[coor1]
            z_s[0,i] = H[coor1]

        ## Printing relevant quantities
        print('Scale: '+str(s)+', '+'Points selected: '+str(l_s)+'/'+str(points)+' Error: '+str(LA.norm(f-H)) )
        
        ## Computing quantities for next iteration
        if LA.norm(f-H)<=err:
            print('Convergence Scale: '+str(s))
            S_conv = s
            break
        
        ## Updating scale parameter 
        s = s+1
        
    return (x_s,y_s,z_s,S_conv,coor,epsilon)



def Hierarchical_predict_basic(x_s,y_s,coor,epsilon,Data_pred):
    ## Input
    ## 1: x_s: The X coordinates of sparse representation
    ## 2: y_s: The Y coordinates of sparse representation
    ## 3: coor: Coordinate of projection on the chosen sparse bases
    ## 4. Data_pred: Prediction points
    
    ## Output
    ## f_star: prediction at Data_pred
    ## gauss_vec: The basis function generated for doing predictions

    l_s = max(x_s.shape)
    siz = Data_pred.shape[0]
    dist_vec = np.zeros([siz,l_s])
    gauss_vec = np.zeros([siz,l_s])
    for i in range(siz):
        for j in range(l_s):
            dist_vec[i,j] = np.sqrt((Data_pred[i,0] - x_s[0,j])**2 + (Data_pred[i,1] - y_s[0,j])**2 )
            gauss_vec[i,j] = np.exp(-(dist_vec[i,j]**2)/epsilon)

    f_star = gauss_vec.dot(coor)
    return f_star,gauss_vec







def figure2(Data_temp,err):
    ## Input:
    ## 1: Data_temp: Data matrix in XY (for univariate functions) or XYZ (for bivariate functions).
    ## 2: err: The error tolerance provided by the user.
    
    ## Output:
    ## 1: (x_s, y_s, z_s): Sparse representation coordinates
    ## 2: S_conv: Convergence Scale
    ## 3: S_crit: Critical Scale
    ## 3: coor: Coordinate of projection on the chosen sparse bases
    ## 4: epsilon: Length scale of the squared exponential kernel
    ## 5: conv_res: record of prediction errors across different scales
    
    
    ## Formatting the Data matrix to be in 3D
    if Data_temp.shape[1] == 2:
        Data = np.zeros([Data_temp.shape[0],3])
        Data[:,0] = Data_temp[:,0]
        Data[:,2] = Data_temp[:,1]
    
    if Data_temp.shape[1] == 3:
        Data = Data_temp
        
    ## Initializations
    points = Data.shape[0]
    s=0;
    X = Data[:,0]
    Y = Data[:,1]
    H = Data[:,2]
    P1 = 2
    l_s = 0
    flag = 0
    
    ## Computing T
    dist = squareform(pdist(Data[:,0:2], 'euclidean'))
    max_d = np.amax(dist)
    T = 2*(max_d/2)**2;

    ## Starting the main loop
    conv_res = []
    while l_s<points:
        epsilon = T/(P1**s)
        Gaussian = np.exp(-((dist**2)/epsilon))

        ## Finding the rank of the Gaussian Kernel
        l_s = LA.matrix_rank(Gaussian)
        k = l_s+8

        ## Calculating the W matrix
        A = random.randn(k,points)
        W = A.dot(Gaussian)
        W = np.matrix(W) 

        ## Pivoted QR decomposition for W
        Q,R,P = linalg.qr(W, mode='full', pivoting = True)
        Perm = np.zeros([len(P),len(P)],dtype=int)
        for w in range(0,len(P)):
            Perm[P[w],w] = 1


        ## Selecting Relevant columns of Kernel into B Matrix
        Mat = Gaussian.dot(Perm)
        B = Mat[:,0:l_s];

        ## Coordinate vector of orthogonal projection
        coor = np.linalg.lstsq(B, H,rcond=None)[0]

        ## Orthogonal Projections on columns of B
        f = B.dot(coor)

        ## Getting the sparse representation
        vec = np.empty([1,points])
        for w in range(0,points):
            vec[0,w] = w
        permv = vec.dot(Perm)
        permv = permv.T
        x_s = np.empty([1,l_s])
        y_s = np.empty([1,l_s])
        z_s = np.empty([1,l_s])
        for i in range(l_s):
            coor1 = int(permv[i])
            x_s[0,i] = X[coor1]
            y_s[0,i] = Y[coor1]
            z_s[0,i] = H[coor1]

        ## Printing relevant quantities
        print('Scale: '+str(s)+', '+'Points selected: '+str(l_s)+'/'+str(points)+' Error: '+str(LA.norm(f-H)) )
        conv_res.append([s,l_s/points,LA.norm(f-H)])
        
        ## Computing quantities for next iteration
        if LA.norm(f-H)<=err and flag ==0:
            print('Convergence Scale: '+str(s))
            S_conv = s
            flag =1

        
        ## Updating scale parameter 
        s = s+1
    S_crit = s-1    
    print('Critical Scale: '+str(S_crit))
    conv_res = np.array(conv_res)
    return (x_s,y_s,z_s,S_conv,S_crit,coor,epsilon,conv_res)




def figure3(Data_temp):
    ## Input:
    ## 1: Data_temp: Data matrix in XY (for univariate functions) or XYZ (for bivariate functions).
    
    ## Output:
    ## 1: bounds: The hilbert space bounds

    
    ## Formatting the Data matrix to be in 3D
    if Data_temp.shape[1] == 2:
        Data = np.zeros([Data_temp.shape[0],3])
        Data[:,0] = Data_temp[:,0]
        Data[:,2] = Data_temp[:,1]
    
    if Data_temp.shape[1] == 3:
        Data = Data_temp
        
    ## Initializations
    points = Data.shape[0]
    s=0;
    X = Data[:,0]
    Y = Data[:,1]
    H = Data[:,2]
    P1 = 2
    l_s = 0
    flag = 0
    
    ## Computing T
    dist = squareform(pdist(Data[:,0:2], 'euclidean'))
    max_d = np.amax(dist)
    T = 2*(max_d/2)**2;

    ## Starting the main loop
    bounds = []
    while l_s<points:
        epsilon = T/(P1**s)
        Gaussian = np.exp(-((dist**2)/epsilon))

        ## Finding the rank of the Gaussian Kernel
        l_s = LA.matrix_rank(Gaussian)
        k = l_s+8

        ## Calculating the W matrix
        A = random.randn(k,points)
        W = A.dot(Gaussian)
        W = np.matrix(W) 

        ## Pivoted QR decomposition for W
        Q,R,P = linalg.qr(W, mode='full', pivoting = True)
        Perm = np.zeros([len(P),len(P)],dtype=int)
        for w in range(0,len(P)):
            Perm[P[w],w] = 1


        ## Selecting Relevant columns of Kernel into B Matrix
        Mat = Gaussian.dot(Perm)
        B = Mat[:,0:l_s];

        ## Coordinate vector of orthogonal projection
        coor = np.linalg.lstsq(B, H,rcond=None)[0]

        ## Orthogonal Projections on columns of B
        f = B.dot(coor)

        ## Computing the upper limit on the inner product
        cc = []
        for u in coor:
            cc.append(abs(u))
        c_infty = np.max(cc)
        
        err1 = 0
        for r in range(len(f)):
            err1 = err1 + abs(f[r]-H[r])
            
        
        bounds.append([s, c_infty*err1])
        # Computing quantities for next iteration
            
        s = s+1
    bounds = np.array(bounds)  
    return (bounds)




def figure_4_5_7_9(Data_temp,s):
    ## Input:
    ## 1: Data_temp: Data matrix in XY (for univariate functions) or XYZ (for bivariate functions).
    ## 2: s: Scale of analysis
    
    ## Output:
    ## 1: (x_s, y_s, z_s): Sparse representation coordinates
    ## 2: coor: Coordinate of projection on the chosen sparse bases
    ## 3: epsilon: Length scale of the squared exponential kernel
    ## 4: permv: importance ranking for the points sampled in the sparse representation
    
    
    ## Formatting the Data matrix to be in 3D
    if Data_temp.shape[1] == 2:
        Data = np.zeros([Data_temp.shape[0],3])
        Data[:,0] = Data_temp[:,0]
        Data[:,2] = Data_temp[:,1]
    
    if Data_temp.shape[1] == 3:
        Data = Data_temp
        
    ## Initializations
    points = Data.shape[0]
    X = Data[:,0]
    Y = Data[:,1]
    H = Data[:,2]
    P1 = 2
    
    ## Computing T
    dist = squareform(pdist(Data[:,0:2], 'euclidean'))
    max_d = np.amax(dist)
    T = 2*(max_d/2)**2;

    ## Computing the Kernel
    epsilon = T/(P1**s)
    Gaussian = np.exp(-((dist**2)/epsilon))

    ## Finding the rank of the Gaussian Kernel
    l_s = LA.matrix_rank(Gaussian)
    k = l_s+8

    ## Calculating the W matrix
    A = random.randn(k,points)
    W = A.dot(Gaussian)
    W = np.matrix(W) 

    ## Pivoted QR decomposition for W
    Q,R,P = linalg.qr(W, mode='full', pivoting = True)
    Perm = np.zeros([len(P),len(P)],dtype=int)
    for w in range(0,len(P)):
        Perm[P[w],w] = 1


    ## Selecting Relevant columns of Kernel into B Matrix
    Mat = Gaussian.dot(Perm)
    B = Mat[:,0:l_s];

    ## Coordinate vector of orthogonal projection
    coor = np.linalg.lstsq(B, H,rcond=None)[0]

    ## Orthogonal Projections on columns of B
    f = B.dot(coor)

    ## Getting the sparse representation
    vec = np.empty([1,points])
    for w in range(0,points):
        vec[0,w] = w
    permv = vec.dot(Perm)
    permv = permv.T
    x_s = np.empty([1,l_s])
    y_s = np.empty([1,l_s])
    z_s = np.empty([1,l_s])
    for i in range(l_s):
        coor1 = int(permv[i])
        x_s[0,i] = X[coor1]
        y_s[0,i] = Y[coor1]
        z_s[0,i] = H[coor1]
        
    return (x_s,y_s,z_s,coor,epsilon,permv)




def figure6(Data_temp,Data_pred,s,confidence = 0.95):
    ## Input:
    ## 1: Data_temp: Data matrix in XY (for univariate functions) or XYZ (for bivariate functions).
    ## 2: Data_pred: Prediction locations
    ## 3: s: Scale of analysis
    ## 4: confidence: %confidence required for the generated bounds (defaults to 95%)
        
    ## Output:
    ## 1: mean_coor: The coordinate of orthogonal projection
    ## 2: std_coor_t: Standard deviation for the estimate of coordinates
    ## 3: mean_y: Mean prediction
    ## 4: std_y_t: t confidence interval for prediction
    ## 5: std_pred_t: t-prediction interal
    ## 6: (x_s, y_s, z_s): Sparse representation coordinates
    
 
    
    ## Formatting the Data matrix to be in 3D
    if Data_temp.shape[1] == 2:
        Data = np.zeros([Data_temp.shape[0],3])
        Data[:,0] = Data_temp[:,0]
        Data[:,2] = Data_temp[:,1]
    
    if Data_temp.shape[1] == 3:
        Data = Data_temp
        
    ## Initializations
    points = Data.shape[0]
    X = Data[:,0]
    Y = Data[:,1]
    H = Data[:,2]
    P1 = 2
    
    ## Computing T
    dist = squareform(pdist(Data[:,0:2], 'euclidean'))
    max_d = np.amax(dist)
    T = 2*(max_d/2)**2;

    ## Computing the Kernel
    epsilon = T/(P1**s)
    Gaussian = np.exp(-((dist**2)/epsilon))

    ## Finding the rank of the Gaussian Kernel
    l_s = LA.matrix_rank(Gaussian)
    k = l_s+8

    ## Calculating the W matrix
    A = random.randn(k,points)
    W = A.dot(Gaussian)
    W = np.matrix(W) 

    ## Pivoted QR decomposition for W
    Q,R,P = linalg.qr(W, mode='full', pivoting = True)
    Perm = np.zeros([len(P),len(P)],dtype=int)
    for w in range(0,len(P)):
        Perm[P[w],w] = 1


    ## Selecting Relevant columns of Kernel into B Matrix
    Mat = Gaussian.dot(Perm)
    B = Mat[:,0:l_s];

    ## Coordinate vector of orthogonal projection
    coor = np.linalg.lstsq(B, H,rcond=None)[0]

    ## Orthogonal Projections on columns of B
    f = B.dot(coor)

    ## Getting the sparse representation
    vec = np.empty([1,points])
    for w in range(0,points):
        vec[0,w] = w
    permv = vec.dot(Perm)
    permv = permv.T
    x_s = np.empty([1,l_s])
    y_s = np.empty([1,l_s])
    z_s = np.empty([1,l_s])
    for i in range(l_s):
        coor1 = int(permv[i])
        x_s[0,i] = X[coor1]
        y_s[0,i] = Y[coor1]
        z_s[0,i] = H[coor1]
        
    
    ### Getting the prediction from the sparse representation
    siz = Data_pred.shape[0]
    dist_vec = np.zeros([siz,l_s])
    gauss_vec = np.zeros([siz,l_s])
    for i in range(siz):
        for j in range(l_s):
            dist_vec[i,j] = np.sqrt((Data_pred[i,0] - x_s[0,j])**2 + (Data_pred[i,1] - y_s[0,j])**2 )
            gauss_vec[i,j] = np.exp(-(dist_vec[i,j]**2)/epsilon)

    f_star = gauss_vec.dot(coor);


    ### Computing \hat{sigma}
    n = Data.shape[0]
    p = l_s
    sig = (np.linalg.norm(f-H)**2)/(n-p)  ## unbiased estimator of variance


    ### Getting t-confidence Intervals with 100(1-alpha) confidence
    ## 1. For confidence on coordinate
    r1 = linalg.inv(B.T.dot(B)+1.0e-6*np.eye(B.T.shape[0]))
    mean_coor = coor
    std_coor = np.sqrt(np.diag(sig*r1))
    std_coor_t = stats.t._ppf((1+confidence)/2.,n-p)*std_coor
    
    ## 2. For Confidence on y
    r2 = gauss_vec.dot(r1).dot(gauss_vec.T)
    mean_y = f_star
    std_y = np.sqrt(np.diag(sig*r2))
    std_y_t = stats.t._ppf((1+confidence)/2.,n-p)*std_y
    
    ### Getting t-prediction intervals with 100(1-alpha) confidence
    r3 = np.eye(gauss_vec.shape[0]) + gauss_vec.dot(r1).dot(gauss_vec.T)
    std_pred = np.sqrt(np.diag(sig*r3))
    std_pred_t = stats.t._ppf((1+confidence)/2.,n-p)*std_pred
    return (mean_coor,std_coor_t,mean_y,std_y_t,std_pred_t,x_s,y_s,z_s)




def figure7(Data,s,freq):
    ## Input:
    ## 1: Data: Dataset for learning
    ## 2: s: scale of study
    ## 3: freq: number of simulations
    
    ## Output:
    ## 1: Histograms for the first second and third most important points
    
    
    ## Creating an empty dictionary
    hist1 = {}
    hist2 = {}
    hist3 = {}
    for i in range(Data.shape[0]):
        hist1[i] = 0
        hist2[i] = 0
        hist3[i] = 0
    
    hist11 = []
    hist22 = []
    hist33 = []
    for h in range(freq):
        x1,y1,z1,coor,epsilon,permv = figure_4_5_7_9(Data,s)
        zero = int(permv[0][0])
        one = int(permv[1][0])
        two = int(permv[2][0])
        hist1[zero] = hist1[zero] + 1
        hist2[one] = hist2[one] + 1
        hist3[two] = hist3[two] + 1
        
        
        hist11.append(zero)
        hist22.append(one)
        hist33.append(two)
        
      
    return (hist1,hist2,hist3,hist11,hist22,hist33)



################################################################################################################

################################################################################################################


