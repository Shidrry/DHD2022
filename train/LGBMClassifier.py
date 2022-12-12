import datetime
import joblib
import lightgbm as lgb
import optuna

from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

class LGBMClassifier():
    def __init__(
        self, 
        params={},
        path=None
    ):
        if path:
            self.model = self.load_model(path)
        else:
            self.model = lgb.LGBMClassifier(**params)

    def save_model(self, path):
        joblib.dump(self, path)

    def load_model(self, path):
        return joblib.load(path)

    def train(self, x_train, y_train):
        print('training starts')
        print(datetime.datetime.now())
        self.model.fit(x_train, y_train)
        print('training ends')
        print(datetime.datetime.now())
        
    def tune_and_train(self, x_train, y_train, param_distributions, n_cv):
        print('tuning starts')
        print(datetime.datetime.now())
        tuner = optuna.integration.OptunaSearchCV(
            estimator=self.model, 
            param_distributions=param_distributions,
            cv=StratifiedKFold(n_splits=n_cv),
            scoring=make_scorer(f1_score, pos_label=1),
            verbose=2
        )
        tuner.fit(x_train, y_train)
        self.tuned_params = tuner.best_params_
        print('tuning ends')
        print(datetime.datetime.now())

        print('training starts')
        print(datetime.datetime.now())
        self.model.set_params(**self.tuned_params)
        self.model.fit(x_train, y_train)
        print('training ends')
        print(datetime.datetime.now())

    def predict(self, x_test):
        prediction = self.model.predict(x_test)
        return prediction

    def predict_proba(self, x_test):
        prob = self.model.predict_proba(x_test)
        return prob

    def return_score(self, x_test, y_test):
        return f1_score(y_test, self.model.predict(x_test))