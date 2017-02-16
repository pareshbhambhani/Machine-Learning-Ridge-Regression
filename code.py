import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

"""Create the class ridge """
class Ridge:
    
    def fit(self, X, y, L):
        G = L * np.eye(X.shape[1])
        G[0, 0] = 0        # We don't want to regularize the bias term
        self.params = np.dot(np.linalg.inv(np.dot(X.T, X) + G),
                             np.dot(X.T, y))

    def predict(self, X):
        return np.dot(X, self.params)

"""Standardization implementation. Calculates mean and std deviation to find standardized values
   Input: Given array
   Returns: Standardized array"""        
def standardization (X_train_ref):
    sum_add=np.zeros(len(X_train_ref[0]))
    mean=np.zeros(len(X_train_ref[0]))
    sum_std=np.zeros(len(X_train_ref[0]))
    mean_std=np.zeros(len(X_train_ref[0]))
    std_dev=np.zeros(len(X_train_ref[0]))
    X_temp = np.zeros(shape=(len(X_train_ref),len(X_train_ref.T)))
    X_out = np.zeros(shape=(len(X_train_ref),len(X_train_ref.T)))
    for i in range(0,len(X_train_ref.T)):
        sum_temp=0
        for j in range(0,len(X_train_ref)):
            sum_temp=sum_temp+ X_train_ref[j][i]
        sum_add[i]=sum_temp
        mean[i]=sum_add[i]/len(X_train_ref)
        temp=0
        for j in range(0,len(X_train_ref)):
            temp=float(((X_train_ref[j][i]-mean[i])**2))
            X_temp[j][i] = np.float(temp)
        for j in range(0,len(X_train_ref)):
            sum_std[i]=sum_std[i]+X_temp[j][i] 
        mean_std[i]=sum_std[i]/(len(X_train_ref))
        std_dev[i]=mean_std[i]**0.5
        for j in range(0,len(X_train_ref)):
            X_out[j][i] = (X_train_ref[j][i] - mean[i])/std_dev[i]
    return X_out

"""Root mean square error calculation.
   Input: Actual and predicted data
   Returns: rmse"""
def Rmse (actual_data,predicted_data):
    N = len(actual_data)
    temp=0
    temp_sum=0
    for i in range(N):
        temp=(actual_data[i]-predicted_data[i])**2
        temp_sum = temp_sum+temp
    return float(np.sqrt(temp_sum/N))

"""Maximum error deviation calculation.
   Inputs: Actual and predicted data
   Returns: MAD, sorted array with the absolute values of difference betweeen actual and predicted data"""        
def MAD (actual_data_M,predicted_data_M):
    N = len(actual_data_M)
    MAD_err = []
    MAD_sum = 0
    for i in range(N):
        MAD_abs_val = abs(actual_data_M[i]-predicted_data_M[i])
        MAD_err.append(float(MAD_abs_val))
        MAD_sum = MAD_sum + MAD_abs_val
    MAD_err.sort()
    return float((MAD_sum/N)),MAD_err

"""To plot the rmse and MAD errors.
   Inputs: x-axis values, y-axis values, label_flag(label=0-->red wine, label=1-->white wine)"""    
def plot(x_axis_val,y_axis_val1,y_axis_val2,label):
    plt.figure()
    plt.plot(x_axis_val, y_axis_val1, marker='o',color='r')
    plt.plot(x_axis_val, y_axis_val2, marker='d',color='b')   
    plt.xscale('log') 
    plt.ylim(0.45,0.95) 
    plt.xticks(x_axis_val) 
    plt.xlabel('Regularization paramter: Lambda')
    plt.ylabel('Error')
    if (label==0):
        plt.title('Measures  of error for red wine dataset')
    else:
        plt.title('Measures  of error for white wine dataset')
    plt.show()

"""Plot the REC curve.
   Inputs: absolute error values, label_flag(red/white wine), Regularization parameter"""
def rec(MAD_err,label,lambda_val):
    error_prev = 0
    error_prev_array = []
    correct = 0
    correct_array = []
    lambda_val_str = str(lambda_val)
    for i in range(len(MAD_err)):
        if (MAD_err[i] > error_prev):
            error_prev_array.append(float(error_prev))
            correct_array.append(float(correct)/len(MAD_err))
            error_prev = deepcopy(MAD_err[i])
        correct += 1
    if (label == 0):
        plt.subplot(2,1,1)
        plt.plot(error_prev_array, correct_array,color='r')
        plt.legend = ('%s' %lambda_val_str)
        plt.xlim(0,MAD_err[len(MAD_err)-1])
        plt.ylabel("Accuracy")
        plt.title("REC curve for red wine dataset")
    else:
        plt.subplot(2,1,2)
        plt.plot(error_prev_array, correct_array,color='g')
        plt.legend = ('%s' %lambda_val_str)
        plt.xlim(0,MAD_err[len(MAD_err)-1])
        plt.xlabel("Absolute deviation")
        plt.ylabel("Accuracy")
        plt.title("REC curve for white wine dataset")

"""Calculation of Pearson Coefficient
   Inputs: Data and label values
   Returns: Pearson coefficient"""        
def pearson_coeff(X,y):
    Rxy = []
    X_no_bias = X[:,1:13]
    mean_arr_X=np.mean(X_no_bias,axis=0)
    mean_arr_Y=np.mean(y)
    for i in range (len(X_no_bias[0])):
        numerator_prev=0
        denom1_prev=0
        denom2_prev=0
        for j in range (len(X_no_bias)):
            numerator = float(((X_no_bias[j][i] - mean_arr_X[i])*((y[j]-mean_arr_Y))))
            numerator_prev = numerator_prev + numerator
            denom1 = float(X_no_bias[j][i] - mean_arr_X[i])**2
            denom1_prev=denom1_prev + denom1
            denom2 = float(y[j] - mean_arr_Y)**2
            denom2_prev=denom2_prev + denom2
        Rxy.append(float(numerator_prev/((np.sqrt(denom1_prev))*np.sqrt(denom2_prev)))) 
    return Rxy
    
"""Read data from file. Shuffle, standardize and separate into training and testing datasets.
   Input: Path location for the data
   Returns: Training and testing datasets"""
def get_data (path):
    data=np.genfromtxt(path, delimiter=";", skip_header = 1)
    I = np.arange(len(data))
    np.random.shuffle(I)
    data = data[I]
    X = data[:,0:11]
    X = standardization(X)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = data[:,11:12]
    X_train = X[0:((len(X)/3)*2),:]
    X_test =  X[((len(X)/3)*2):len(X),0::]
    y_train = y[0:((len(y)/3)*2),0::]
    y_test = y[((len(y)/3)*2):len(y),0::]
    return X_train,X_test,y_train,y_test


if __name__ == '__main__':
    """Get data"""    
    X_red_train,X_red_test,y_red_train,y_red_test = get_data('/data/Work_CSU/Course_Work/Dropbox/Masters/Fall15/Machine_learning/Assignment2/Data/winequality-red.csv')
    X_white_train,X_white_test, y_white_train,y_white_test = get_data('/data/Work_CSU/Course_Work/Dropbox/Masters/Fall15/Machine_learning/Assignment2/Data/winequality-white.csv')
    r = Ridge() 
    """Calculate rmse and MAD for various values of regularization parameter"""
    rmse_red=[]
    MAD_red=[]
    rmse_white=[]
    MAD_white=[]
    MAD_err_temp=[]
    log_scale = np.logspace(-2,7,base=10.0,num=10)
    for L in (log_scale):
        MAD_abs_red_err = [] 
        MAD_abs_white_err = []
        r.fit(X_red_train,y_red_train,L)
        y_red_predicted = r.predict(X_red_test)
        rmse_red.append(Rmse(y_red_test,y_red_predicted))
        MAD_max_red_err,MAD_abs_red_err = MAD(y_red_test,y_red_predicted)
        MAD_red.append(MAD_max_red_err)
        r.fit(X_white_train,y_white_train,L)
        y_white_predicted = r.predict(X_white_test)
        rmse_white.append(Rmse(y_white_test,y_white_predicted))
        MAD_max_white_err,MAD_abs_white_err = MAD(y_white_test,y_white_predicted)
        MAD_white.append(MAD_max_white_err)
    """Plot the rmse and MAD"""
    plot(log_scale,rmse_red,MAD_red,0)
    plot(log_scale,rmse_white,MAD_white,1)   
    """Find the lambda for which MAD error is minimum in red wine dataset"""
    min_MAD_red=np.amin(MAD_red)
    for i,j in enumerate(MAD_red):
        if (j == (min_MAD_red)):
            index_red_MAD=i
    """Find the lambda for which MAD error is minimum in white wine dataset"""
    min_MAD_white=np.amin(MAD_white)
    for i,j in enumerate(MAD_white):
        if (j == (min_MAD_white)):
            index_white_MAD=i
    """Use the above found value of lambda to plot the REC curves for red and white wine dataset"""
    plt.figure()    
    r.fit(X_red_train,y_red_train,index_red_MAD)
    y_red_predicted = r.predict(X_red_test)
    MAD_max_red_err,MAD_abs_red_err = MAD(y_red_test,y_red_predicted)
    rec(MAD_abs_red_err,0,index_red_MAD)
    r.fit(X_white_train,y_white_train,index_white_MAD)
    y_white_predicted = r.predict(X_white_test)
    MAD_max_white_err,MAD_abs_white_err = MAD(y_white_test,y_white_predicted)
    rec(MAD_abs_white_err,1,index_white_MAD)
    """Use the above found value of lambda to find the Pearson coefficients for both red and white wine"""
    X_red_train_scatter = deepcopy(X_red_train[0:(len(X_red_train))/2,:])
    X_red_train_decrement = deepcopy(X_red_train[((len(X_red_train))/2):(len(X_red_train)),:])
    y_red_train_scatter = deepcopy(y_red_train[0:(len(y_red_train))/2,:])
    y_red_train_decrement = deepcopy(y_red_train[(len(y_red_train))/2:(len(y_red_train)),:])
    r.fit(X_red_train_scatter,y_red_train_scatter,index_red_MAD)
    X_red_test_decrement = deepcopy(X_red_test)
    Weight_vector_red_wine=deepcopy(r.params)
    pearson_coeff_red = pearson_coeff(X_red_train_scatter,y_red_train_scatter)
    """Scatter plot of weight vector vs pearson coefficients"""
    plt.figure()
    colors = np.random.rand(11)
    area = np.pi * (4)**2
    plt.subplot(2,1,1)
    plt.ylabel('Pearson Coefficient')
    plt.scatter(r.params[1:12],pearson_coeff_red,s=area,c=colors,alpha=1) 
    plt.title('Scatter plot for red wine dataset')
    X_white_train_scatter = deepcopy(X_white_train[0:(len(X_white_train))/2,:])
    X_white_train_decrement = deepcopy(X_white_train[((len(X_white_train))/2):(len(X_white_train)),:])
    y_white_train_scatter = deepcopy(y_white_train[0:(len(y_white_train))/2,:])
    y_white_train_decrement = deepcopy(y_white_train[(len(y_white_train))/2:(len(y_white_train)),:])
    r.fit(X_white_train_scatter,y_white_train_scatter,index_white_MAD)  
    X_white_test_decrement = deepcopy(X_white_test)
    Weight_vector_white_wine=deepcopy(r.params) 
    pearson_coeff_white = pearson_coeff(X_white_train,y_white_train)
    plt.subplot(2,1,2)
    plt.xlabel('Weight Vector')
    plt.ylabel('Pearson Coefficient')
    plt.scatter(r.params[1:12],pearson_coeff_white,s=area,c=colors,alpha=1)
    plt.title('Scatter plot for white wine dataset')
    """Feature removal experiment for red wine dataset"""
    rmse_red_decrement=[]
    MAD_red_decrement=[]
    red_scale=[]
    for i in range(len(X_red_train[0])-1):
        y_red_predicted_decrement = np.dot(X_red_test_decrement, Weight_vector_red_wine)  
        rmse_red_decrement.append(Rmse(y_red_test,y_red_predicted_decrement))
        MAD_max_red_err,MAD_abs_red_err = MAD(y_red_test,y_red_predicted_decrement)
        MAD_red_decrement.append(MAD_max_red_err)
        Weight_vector_red_wine_absolute = np.absolute(Weight_vector_red_wine)
        min_wt_red = np.amin(Weight_vector_red_wine_absolute)
        for i,j in enumerate(Weight_vector_red_wine_absolute):
            if (j == (min_wt_red)):
                index_wt_red_min=i    
        X_red_train_decrement = np.delete(X_red_train_decrement,np.s_[index_wt_red_min],axis=1)
        X_red_test_decrement = np.delete(X_red_test_decrement,np.s_[index_wt_red_min],axis=1)         
        r.fit(X_red_train_decrement,y_red_train_decrement,index_red_MAD)
        Weight_vector_red_wine=deepcopy(r.params)
        red_scale.append(i) 
    """Feature removal experiment for white wine dataset"""
    rmse_white_decrement=[]
    MAD_white_decrement=[]
    white_scale=[]
    for i in range(len(X_white_train[0])-1):
        y_white_predicted_decrement = np.dot(X_white_test_decrement, Weight_vector_white_wine)  
        rmse_white_decrement.append(Rmse(y_white_test,y_white_predicted_decrement))
        MAD_max_white_err,MAD_abs_white_err = MAD(y_white_test,y_white_predicted_decrement)
        MAD_white_decrement.append(MAD_max_white_err)
        Weight_vector_white_wine_absolute = np.absolute(Weight_vector_white_wine)
        min_wt_white = np.amin(Weight_vector_white_wine_absolute)
        for i,j in enumerate(Weight_vector_white_wine_absolute):
            if (j == (min_wt_white)):
                index_wt_white_min=i    
        X_white_train_decrement = np.delete(X_white_train_decrement,np.s_[index_wt_white_min],axis=1)
        X_white_test_decrement = np.delete(X_white_test_decrement,np.s_[index_wt_white_min],axis=1)          
        r.fit(X_white_train_decrement,y_white_train_decrement,index_white_MAD)
        Weight_vector_white_wine=deepcopy(r.params)
        white_scale.append(i)
    """Plot of rmse and MAD with respect to the no. of features"""
    plt.figure()
    plt.subplot(2,1,1)  
    plt.plot(red_scale,rmse_red_decrement, marker='o',color='r')
    plt.plot(red_scale,MAD_red_decrement, marker='d',color='b') 
    plt.xlim(11,1)
    plt.ylim(0.4,0.8)
    plt.xticks(red_scale) 
    plt.ylabel('Rmse_MAD_Error')
    plt.title("Error vs no of features for red wine dataset")
    plt.subplot(2,1,2)
    plt.plot(white_scale,rmse_white_decrement, marker='o',color='r')   
    plt.plot(white_scale,MAD_white_decrement, marker='d',color='b') 
    plt.xlim(11,1)
    plt.ylim(0.5,0.9)
    plt.xticks(white_scale) 
    plt.xlabel('No. of features')
    plt.ylabel('Rmse_MAD_Error')
    plt.title("Error vs no of features for white wine dataset")
