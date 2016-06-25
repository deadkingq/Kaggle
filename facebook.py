#Import libraries:
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
import xgboost
from sklearn import preprocessing
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def bindata(data, NX, NY):
    NX=int(NX)
    NY=int(NY)
    deltaX = 10 / float(NX)
    deltaY = 10 / float(NY)
    xbins = np.floor(data.x.values/ deltaX).astype(np.int32)
    ybins = np.floor(data.y.values/ deltaY).astype(np.int32)
    xbins[xbins == NX] = NX-1
    ybins[ybins == NY] = NY-1
    data['xlabel'] = xbins
    data['ylabel'] = ybins
    return


def CV_fit(train,kfold=5, keep_fraction=0.5):
    '''Performs a fit using XGBoost. Applies kfold cross validation to
    estimate error.

    Parameters
    ----------
    train
        The training dataset as a pandas DataFrame
    test
        The testing dataset as a pandas DataFrame
    kfold
        The number of folds to use for K Fold Validation
    keep_fraction
        A float between 0 and 1. The fraction of events in each bin
        to keep while minimizing the number of place_ids in the training
        set. Low values throw away a lot of infrequent place_ids, and
        values near 1 retain almost all place_ids.
    '''

    rs = np.random.RandomState(42)
    bin_numbers = zip(rs.randint(0, 50, size=10), rs.randint(0, 50, size=10))

    predictions = []
    map3s = []

    for i_bin_x, i_bin_y in bin_numbers:
        print("Bin {},{}".format(i_bin_x, i_bin_y))
        train_in_bin = train[(train.xlabel == i_bin_x)
                             & (train.ylabel == i_bin_y)].sort_values('time')
        #test_in_bin = test[(test.xlabel == i_bin_x)
        #                  & (test.ylabel == i_bin_y)].sort_values('time')

        N_total_in_bin = train_in_bin.shape[0]
        keep_N = int(float(N_total_in_bin)*keep_fraction)
        vc = train_in_bin.place_id.value_counts()
        vc = vc[np.cumsum(vc.values) < keep_N]
        df1 = pd.DataFrame({'place_id': vc.index, 'freq': vc.values})
        train_in_bin_2 = pd.merge(train_in_bin, df1, on='place_id',how='inner')
        le = preprocessing.LabelEncoder()
        le.fit(train_in_bin_2.place_id.values)
        y_train = le.transform(train_in_bin_2.place_id.values)
        x_train = train_in_bin_2['x y accuracy hour weekday month year'.split()].as_matrix()
        #x_test = test_in_bin['x y accuracy hour weekday month year'.split()].as_matrix()
        dm_train = xgboost.DMatrix(x_train, label=y_train)
        #dm_test = xgboost.DMatrix(x_test)
        res = xgboost.cv(
            {'eta': 0.1, 'objective': 'multi:softprob',
             'num_class': len(le.classes_),
             'alpha': 0.1, 'lambda': 0.1, 'booster': 'gbtree'},
            dm_train, num_boost_round=200, nfold=kfold, seed=42,
            early_stopping_rounds=10
            # For some reason, verbose_eval seems to be broken on my install
            )

        print(res)
        N_epochs = res.shape[0]
        booster = xgboost.train(
            {'eta': 0.1, 'objective': 'multi:softprob',
             'num_class': len(le.classes_),
             'alpha': 0.1, 'lambda': 0.1, 'booster': 'gbtree'},
            dm_train, num_boost_round=N_epochs)
        predict_y_train = booster.predict(dm_train)
        #predict_y_test = booster.predict(dm_test)

    return predict_y_train


def main():

    print('Reading csv files from disk.')
    df_train=pd.read_csv("/users/seven/downloads/train1.csv",sep='\t')
    print("Calculating time features")
    df_train['hour'] = (df_train['time']//60) % 24+1
    df_train['weekday'] = df_train['hour']//24+1
    df_train['month'] = df_train['weekday']//30+1
    df_train['year'] = (df_train['weekday']//365+1)
    #df_train = df_train.drop(['time'], axis=1)
    print("Binning data")
    bindata(df_train, 50, 50)
    #bindata(test, 50, 50)

    print("Starting CV")
    predict_y_train= CV_fit(df_train, kfold=5)
    print(predict_y_train)

if __name__ == '__main__':
    main()