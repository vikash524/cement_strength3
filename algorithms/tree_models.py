from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import warnings
import logging

warnings.filterwarnings('ignore')  # Ignore warnings

# Configure logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')

class TreeModelsReg:
    """Class for building and tuning regression models using tree-based and ensemble algorithms.
    Author: Vikash Chauhan 😊
    """

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def decision_tree_regressor(self):
        """Builds a Decision Tree Regressor with RandomizedSearchCV for hyperparameter tuning."""
        try:
            dt = DecisionTreeRegressor(random_state=42)

            params = {'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                      'max_depth': [2, 5, 10, 20],
                      'min_samples_split': [2, 4, 8, 12],
                      'min_samples_leaf': [2, 4, 6, 8, 10]}

            rcv = RandomizedSearchCV(estimator=dt, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2, random_state=42, n_jobs=-1,
                                     return_train_score=True)

            rcv.fit(self.x_train, self.y_train)
            best_dt = rcv.best_estimator_
            best_dt.fit(self.x_train, self.y_train)

            # Feature importance
            dt_feature_imp = pd.DataFrame(best_dt.feature_importances_, index=self.x_train.columns,
                                          columns=['Feature_importance'])
            dt_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print(f"Decision Tree Regressor Feature Importance:\n{dt_feature_imp}\n")

            logging.info("Decision Tree Regressor model built successfully.")

            return best_dt
        except Exception as e:
            logging.error(f"Error in decision_tree_regressor: {e}")
            return None

    def random_forest_regressor(self):
        """Builds a Random Forest Regressor with RandomizedSearchCV for hyperparameter tuning."""
        try:
            rf = RandomForestRegressor(random_state=42)

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'criterion': ['mse', 'mae'],
                      'max_depth': [2, 5, 10, 20],
                      'min_samples_split': [2, 4, 8, 12],
                      'min_samples_leaf': [2, 4, 6, 8, 10],
                      'oob_score': [True]}

            rcv = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2, random_state=42, n_jobs=-1,
                                     return_train_score=True)

            rcv.fit(self.x_train, self.y_train)
            best_rf = rcv.best_estimator_
            best_rf.fit(self.x_train, self.y_train)

            # Feature importance
            rf_feature_imp = pd.DataFrame(best_rf.feature_importances_, index=self.x_train.columns,
                                          columns=['Feature_importance'])
            rf_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print(f"Random Forest Regressor Feature Importance:\n{rf_feature_imp}\n")

            logging.info("Random Forest Regressor model built successfully.")

            return best_rf
        except Exception as e:
            logging.error(f"Error in random_forest_regressor: {e}")
            return None

    def adaboost_regressor(self):
        """Builds an AdaBoost Regressor with RandomizedSearchCV for hyperparameter tuning."""
        try:
            adb = AdaBoostRegressor(random_state=42)

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                      'loss': ['linear', 'square', 'exponential']}

            rcv = RandomizedSearchCV(estimator=adb, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2, random_state=42, n_jobs=-1,
                                     return_train_score=True)

            rcv.fit(self.x_train, self.y_train)
            best_adb = rcv.best_estimator_
            best_adb.fit(self.x_train, self.y_train)

            # Feature importance
            adb_feature_imp = pd.DataFrame(best_adb.feature_importances_, index=self.x_train.columns,
                                           columns=['Feature_importance'])
            adb_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print(f"AdaBoost Regressor Feature Importance:\n{adb_feature_imp}\n")

            logging.info("AdaBoost Regressor model built successfully.")

            return best_adb
        except Exception as e:
            logging.error(f"Error in adaboost_regressor: {e}")
            return None

    def gradientboosting_regressor(self):
        """Builds a GradientBoosting Regressor with RandomizedSearchCV for hyperparameter tuning."""
        try:
            gbr = GradientBoostingRegressor(random_state=42)

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'learning_rate': [0.1, 0.2, 0.5, 0.8, 1],
                      'loss': ['ls', 'lad', 'huber'],
                      'subsample': [0.1, 0.3, 0.5, 1],
                      'max_depth': [3, 4, 5],
                      'min_samples_split': [2, 4, 8]}

            rcv = RandomizedSearchCV(estimator=gbr, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2, random_state=42, n_jobs=-1,
                                     return_train_score=True)

            rcv.fit(self.x_train, self.y_train)
            best_gbr = rcv.best_estimator_
            best_gbr.fit(self.x_train, self.y_train)

            # Feature importance
            gbr_feature_imp = pd.DataFrame(best_gbr.feature_importances_, index=self.x_train.columns,
                                           columns=['Feature_importance'])
            gbr_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print(f"GradientBoosting Regressor Feature Importance:\n{gbr_feature_imp}\n")

            logging.info("GradientBoosting Regressor model built successfully.")

            return best_gbr
        except Exception as e:
            logging.error(f"Error in gradientboosting_regressor: {e}")
            return None

    def xgb_regressor(self):
        """Builds an XGBoost Regressor with RandomizedSearchCV for hyperparameter tuning."""
        try:
            xgbr = XGBRegressor(random_state=42)

            params = {'learning_rate': [0.1, 0.2, 0.5, 0.8, 1],
                      'max_depth': [2, 3, 4, 5, 6],
                      'subsample': [0.5, 0.7, 0.9, 1.0],
                      'min_child_weight': [1, 2, 3, 4],
                      'gamma': [0, 0.1, 0.2, 0.3],
                      'colsample_bytree': [0.3, 0.5, 0.7, 1.0]}

            rcv = RandomizedSearchCV(estimator=xgbr, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2, random_state=42, n_jobs=-1,
                                     return_train_score=True)

            rcv.fit(self.x_train, self.y_train)
            best_xgbr = rcv.best_estimator_
            best_xgbr.fit(self.x_train, self.y_train)

            # Feature importance
            xgbr_feature_imp = pd.DataFrame(best_xgbr.feature_importances_, index=self.x_train.columns,
                                            columns=['Feature_importance'])
            xgbr_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print(f"XGBoost Regressor Feature Importance:\n{xgbr_feature_imp}\n")

            logging.info("XGBoost Regressor model built successfully.")

            return best_xgbr
        except Exception as e:
            logging.error(f"Error in xgb_regressor: {e}")
            return None

    def model_predict(self, model, X):
        """Helper method to predict using a trained model."""
        try:
            predictions = model.predict(X)
            return predictions
        except Exception as e:
            logging.error(f"Error in model_predict: {e}")
            return None

