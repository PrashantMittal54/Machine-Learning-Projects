import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def cost_function(X_var, Y_actual, B_var):
    m = len(Y_actual)
    Y_pred = np.dot(X_var, B_var)
    J = sum((Y_pred - Y_actual) **2)/(2*m)
    return J


def gradient_descent_su(X_var, Y_actual, B_var, alpha, iterations):
    m = len(Y_actual)
    cost_history = np.zeros(iterations)
    temp_B_var = B_var.copy()

    for iteration in range(0, iterations):
        Y_pred = np.dot(X_var, B_var)
        cost = cost_function(X_var, Y_actual, B_var)
        cost_history[iteration] = cost

        # threshold for the cost reduction
        if abs(cost_history[iteration] - cost_history[iteration - 1]) <= 0.0001:
            print(iteration, cost_history[iteration], " ", cost_history[iteration - 1])
            break

        # Because we have added added X0 = 1 for beta(0), we are updating all together
        for j in range(0, 15):
            temp_B_var[j] = B_var[j] - (alpha / m) * sum((Y_pred - Y_actual) * X_var[:, j])
        B_var = temp_B_var.copy()

    return B_var, cost_history


def linearRegression():
    #loading data
    df = pd.read_csv(r"C:\Prashant\UTD_Semesters\Second\Machine_Learning\Assignemnts\Assignment1\sgemm_product.csv")
    # Creating a new column with average of 4 runs.
    df['AvgRun'] = round((df['Run1 (ms)']+df['Run2 (ms)']+df['Run3 (ms)']+df['Run4 (ms)'])/4, 2)
    df.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], inplace=True, axis=1)
    df['SB'] = pd.Categorical(df['SB'])
    df['SA'] = pd.Categorical(df['SA'])
    df['STRM'] = pd.Categorical(df['STRM'])
    df['STRN'] = pd.Categorical(df['STRN'])
    y = pd.DataFrame(df.iloc[:,-1])
    x = df.iloc[:,:-1]
    x0 = np.ones((x.shape[0],1))
    x = np.hstack((x0,x))
    y = y.values
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
    B = np.zeros((x_train.shape[1],1))
    maxiter = 10
    alpha = 0.0001
    beta, cost = gradient_descent_su(x_train, y_train, B, alpha, maxiter)
    a = 1

def gradientDescent(x,y,beta,alpha,iter):
    cost_final = []
    beta_list = []
    B = beta
    # B_temp = B.copy()
    for i in range(iter):
        e = (np.dot(x, beta) - y)
        # for j in range(x.shape[1]):
        #     B_temp[j] = B[j] - (alpha / len(x)) * sum(e * x[:, j])
        # B = B_temp.copy()
        # gradient = np.sum(e*x, axis=0)
        gradient = np.dot(x.T,e)/len(x)
        # gradient = np.round(gradient.astype(np.double), 3)
        # gradient = np.dot
        B = B - (alpha) * gradient
        cost = computeCost(x, y, B)
        beta_list.append(B)
        cost_final.append(cost)

    itera = np.arange(iter)
    # itera.shape
    plt.style.use('seaborn-whitegrid')
    plt.plot(itera, cost_final, color='blue')
    plt.show()
    min_cost = min(cost_final)
    min_index = cost_final.index(min_cost)
    b = beta_list[min_index]
    return b, min_cost


def computeCost(X,Y,B):
    total = (np.dot(X,B)-Y)**2
    j = np.sum(total)
    j = j/(2 * len(X))
    return j

linearRegression()


def logisticRegression():
    df = pd.read_csv(r"C:\Prashant\UTD_Semesters\Second\Machine_Learning\Assignemnts\sgemm_product.csv")
    df['AvgRun'] = round((df['Run1 (ms)']+df['Run2 (ms)']+df['Run3 (ms)']+df['Run4 (ms)'])/4, 2)
    df.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], inplace=True, axis=1)
    df['SB'] = pd.Categorical(df['SB'])
    df['SA'] = pd.Categorical(df['SA'])
    df['STRM'] = pd.Categorical(df['STRM'])
    df['STRN'] = pd.Categorical(df['STRN'])
    y = pd.DataFrame(df.iloc[:,-1])
    x = df.iloc[:,:-1]
    x0 = np.ones((x.shape[0],1))
    x = np.hstack((x0,x))
    y = y.values
    y_median = np.median(y)
    y = np.where(y < y_median, 0, 1)
    # y[y >= y_median] = 1
    # y[y < y_median] = 0
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
    B = np.zeros((x_train.shape[1],1))
    maxiter = 100
    alpha = 0.00001
    beta, cost = gradientDescent1(x_train, y_train, B, alpha, maxiter)
    a = 1

def gradientDescent1(x,y,beta,alpha,iter):
    cost_final = []
    beta_list = []
    B = beta
    for i in range(iter):
        xB = np.round(np.dot(x, beta).astype(np.double), 0)
        xB = xB.astype(int)
        z = 1 / (1 + np.exp(-xB))
        e = z - y
        gradient = np.dot(x.T,e)/len(x)
        gradient = np.round(gradient.astype(np.double), 3)
        B = B - alpha * gradient
        cost = computeCost1(x, y, B)
        beta_list.append(B)
        cost_final.append(cost)
    min_cost = min(cost_final)
    min_index = cost_final.index(min_cost)
    b = beta_list[min_index]
    return b, min_cost


def computeCost1(X,Y,B):
    xB = np.round(np.dot(X, B).astype(np.double), 0)
    xB = xB.astype(int)
    yhat = 1 / (1 + np.exp(-xB))
    identity = np.ones((len(yhat),1))
    sum1 = (np.multiply(Y, np.log(yhat)) + np.multiply((identity-Y), np.log((identity-yhat))))
    # totalsum = np.power((np.dot(X,B)-Y),2)
    return np.sum(sum1)/(-len(X))

# logisticRegression()
