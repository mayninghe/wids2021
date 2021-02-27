import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix
from sklearn import preprocessing
import lightgbm as lgb
import warnings
from bayes_opt import BayesianOptimization as BO

warnings.filterwarnings('ignore')


def lgb_bayes(data):
    def lgb_eval(learning_rate, feature_fraction, max_depth, lambda_l1,
                 lambda_l2, scale_pos_weight, num_leaves, min_data_in_leaf):
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'max_depth': int(max_depth),
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'scale_pos_weight': int(scale_pos_weight),
            'num_leaves': int(num_leaves),
            'min_data_in_leaf': int(min_data_in_leaf),
            'verbose': -1,
            'feature_pre_filter': False
        }

        cv_bst = lgb.cv(lgb_params, data, nfold=5, num_boost_round=10000, early_stopping_rounds=200,
                        return_cvbooster=True)

        return cv_bst['auc-mean'][-1]

    lgb_param_space = {
        'learning_rate': (0.001, 0.01),
        'feature_fraction': (0.4, 0.8),  # opt by optuna: 0.62
        'lambda_l1': (0, 10),  # opt by optuna: 0
        'lambda_l2': (0, 10),  # opt by optuna: 3.17
        'max_depth': (3, 6),
        'scale_pos_weight': (1, 10),
        'num_leaves': (15, 255),  # opt by optuna: 21
        'min_data_in_leaf': (10, 50)  # opt by optuna: 20
    }

    return lgb_eval, lgb_param_space


def run_bayes_opt(eval_func, param_space):
    """
    This function is to run Bayesian optimization.
    'init_points' is the number of initializations - random search.
    'n_iter' is the number of iterations after your random initializations.
    """

    bo = BO(eval_func, param_space)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        bo.maximize(init_points=10,  # can increase if performance not good
                    n_iter=30,  # can increase if performance not good
                    acq='ucb',  # can try 'ei' or 'poi'
                    alpha=1e-6,
                    kappa=5#,  # can try 10 with acq='ucb'
                    #kappa_decay=0.99,  # can tune with acq='ucb'
                    #kappa_decay_delay=3,  # can increase if kappa decays too fast
                    #xi=0.1,  # can 0.001 with acq='ei' or acq='poi'
                    )



    return bo


if __name__ == '__main__':
    train = pd.read_csv('../data/train_x.csv')
    y = pd.read_csv('../data/train_y.csv')
    test = pd.read_csv('../data/test_x.csv')

    xs = [np.concatenate([train[f].values, test[f].values]).reshape(-1, 1) for f in train.columns]

    xs = hstack([coo_matrix(x) for x in xs]).tocsr().astype('float32')
    xs_train = xs[0:train.shape[0]]
    xs_test = xs[train.shape[0]:]
    del xs

    print(xs_train.shape, xs_test.shape)
    lgb_data = lgb.Dataset(xs_train, y)

    lgb_eval, lgb_param_space = lgb_bayes(lgb_data)
    lgb_bo = run_bayes_opt(lgb_eval, lgb_param_space)
    max_bo_params = lgb_bo.max['params']
    print(max_bo_params)

    # ucb, kappa=1
    # max_bo_params = {'feature_fraction': 0.45414325040277254,
    #                  'lambda_l1': 9.966978044815114,
    #                  'lambda_l2': 6.559649182324128,
    #                  'learning_rate': 0.01,
    #                  'max_depth': 6.0,
    #                  'min_data_in_leaf': 41.69284408221577,
    #                  'num_leaves': 19.761695908394966,
    #                  'scale_pos_weight': 1.1120954162106733}


    opt_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 10000,
        'learning_rate': max_bo_params['learning_rate'] / 2,
        'feature_fraction': max_bo_params['feature_fraction'],
        'max_depth': int(max_bo_params['max_depth']),
        'lambda_l1': max_bo_params['lambda_l1'],
        'lambda_l2': max_bo_params['lambda_l2'],
        'scale_pos_weight': int(max_bo_params['scale_pos_weight']),
        'min_data_in_leaf': int(max_bo_params['min_data_in_leaf']),
        'early_stopping_round': 300,
    }

    print(opt_params)

    # retrain model with optimal parameters
    bo_bst = lgb.cv(opt_params, lgb_data, nfold=5, num_boost_round=10000, early_stopping_rounds=200,
                    return_cvbooster=True)

    preds = bo_bst['cvbooster'].predict(xs_test, num_iteration=bo_bst['cvbooster'].best_iteration)
    submission = pd.read_csv('../data/UnlabeledWiDS2021.csv', usecols=['encounter_id'])
    submission['diabetes_mellitus'] = np.mean(preds, axis=0)
    submission.to_csv("../data/submission_afterDrop.csv", header=True, index=False)
