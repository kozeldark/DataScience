import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np 
import pandas as pd 
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,ConfusionMatrixDisplay,precision_score,recall_score,f1_score,classification_report,roc_curve,plot_roc_curve,auc,precision_recall_curve,plot_precision_recall_curve,average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv('stroke7.csv')
print(df.head())
print(df.isnull().sum())

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

def visualizationData(df):
    # show missing values by heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.isna(), cbar=False)
    plt.xticks(rotation=45)
    plt.yticks(ticks=[])
    plt.subplots_adjust(left=0.05, bottom=0.24, right=0.95, top=0.95)
    plt.show()

    # show number of values in each column
    for feature in ["gender","hypertension","heart_disease","ever_married","work_type","Residence_type","smoking_status"]:
        sns.countplot(x=df[feature],hue=df["stroke"])
        plt.xlabel(feature)
        plt.title(feature)
        plt.show()

    for feature in ["age","avg_glucose_level","bmi"]:
        sns.countplot(x=df[feature],hue=df["stroke"])
        plt.xlabel(feature)
        plt.title(feature)
        plt.show()

    # show boxplot for columns with continuous values rather than specific values.
    for feature in ["avg_glucose_level", "bmi"]:
        sns.boxplot(x=feature, data=df)
        plt.show()

def visualizationCorrelation(df):
    # compute the corr matrix
    corr = df.corr()

    # generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 6))

    # generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # draw the heatpmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})
    plt.subplots_adjust(left=0, bottom=0.24, right=1, top=1)
    plt.show()


def preprocessing(data, encoder, scaler):
    data.drop('id', axis=1, inplace=True)  # 필요없는 id값 드랍
    data['age'] = data['age'].apply(lambda x: round(x))  # 나이 반올림
    data['bmi'] = data['bmi'].apply(lambda bmi_value: bmi_value if 12 < bmi_value < 45 else np.nan)  # bmi 아웃라이어 처리
    data['gender'] = data['gender'].apply(
        lambda gender: gender if gender == 'Female' or gender == 'Male' else np.nan)  # other 값 처리
    data.dropna(axis=0, inplace=True)  # 결측값
    data.reset_index(drop=True, inplace=True)

    print(encoder)
    # categorical data convert to numeric
    if(encoder == 'Label'):
        encoder = LabelEncoder()
        data['gender'] = encoder.fit_transform(data['gender'])
        data['work_type'] = encoder.fit_transform(data['work_type'])
        data['Residence_type'] = encoder.fit_transform(data['Residence_type'])
        data['smoking_status'] = encoder.fit_transform(data['smoking_status'])
        data['ever_married'] = encoder.fit_transform(data['ever_married'])
    else:
        data = pd.get_dummies(data,columns=['gender','work_type','Residence_type','smoking_status','ever_married'])
    print(data)
    # OverSampling to balance the Data
    target = data["stroke"]
    feat = data.drop('stroke', axis=1)

    feat, target = SMOTE(random_state=2, sampling_strategy=0.2).fit_resample(feat, target)

    target.drop(feat[feat.duplicated(keep="first") == True].index, inplace=True)
    feat = feat.drop_duplicates(keep="first")

    feat['bmi'] = feat['bmi'].apply(lambda bmi_value: round(bmi_value,1))  # bmi 아웃라이어 처리
    feat['avg_glucose_level'] = feat['avg_glucose_level'].apply(lambda avg_glucose_level: round(avg_glucose_level,2))  # bmi 아웃라이어 처리

    X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=0.2, random_state=0,stratify=target)

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.fit_transform(X_test)

    return X_train, y_train, X_test, y_test

def visualizationAccuracy(result):
    # bar graph the accuracy for each model.
    sns.barplot(x='Accuracy', y='Model', data=result, color='b')
    for i, v in enumerate(result['Accuracy']):
        plt.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
    plt.show()

    sns.barplot(x='K-Fold Mean Accuracy', y='Model', data=result, color='b')
    for i, v in enumerate(result['K-Fold Mean Accuracy']):
        plt.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
    plt.show()

    sns.barplot(x='Precision', y='Model', data=result, color='b')
    for i, v in enumerate(result['Precision']):
        plt.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
    plt.show()

    sns.barplot(x='Recall', y='Model', data=result, color='b')
    for i, v in enumerate(result['Recall']):
        plt.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
    plt.show()

    sns.barplot(x='F1 Score', y='Model', data=result, color='b')
    for i, v in enumerate(result['Score']):
        plt.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
    plt.show()

def visualizationConfusionMatrix(matrix):
    # confusion matrix
    plt.figure(figsize=(8, 5))
    sns.heatmap(matrix, cmap='Blues', annot=True, fmt='d', linewidths=5, cbar=False, annot_kws={'fontsize': 15},
                yticklabels=['No stroke', 'Stroke'], xticklabels=['Predicted no stroke', 'Predicted stroke'])
    plt.yticks(rotation=0)
    plt.show()

def visualizationROC(false_positive_rate, true_positive_rate):
    # ROC Curve
    roc_auc = auc(false_positive_rate, true_positive_rate)

    sns.set_theme(style='white')
    plt.figure(figsize=(8, 8))
    plt.plot(false_positive_rate, true_positive_rate, color='#b01717', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#174ab0')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Models to be used for running.
models = []
models.append(['XGB Classifier', XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')])
models.append(['K-Nearest-Neigbors', KNeighborsClassifier()])
models.append(['Decision Tree', DecisionTreeClassifier()])
models.append(['Random Forest', RandomForestClassifier()])
models.append(['ExtraTrees Forest', ExtraTreesClassifier()])
models.append(['AdaBoost Classifier', AdaBoostClassifier()])
models.append(['GradientBoost Classifier', GradientBoostingClassifier()])

# Models that have undergone hyperparameter tuning to be used in grid search.
g_models = [
            (XGBClassifier(use_label_encoder=False, eval_metric='merror'), [{'learning_rate': [0.1, 0.5, 1], 'n_estimators':[10,100,1000],
                                'booster':['gbtree','gblinear'] }]),
            (KNeighborsClassifier(), [{'n_neighbors':[2,3,4],'leaf_size':[15,30],
                                    'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree', 'brute']}]),
            (DecisionTreeClassifier(), [{'criterion':['gini','entropy'],'splitter':['best','random'],
                                        'max_depth':[None,2,3],'max_features':[None, 'sqrt','log2']}]),
            (RandomForestClassifier(), [{'n_estimators':[10,100,1000],'criterion':['gini','entropy'],'max_depth': [None, 2, 3],
                                        'max_features': [None, 'sqrt', 'log2']}]),
            (ExtraTreesClassifier(), [{'n_estimators':[10,100,1000],'criterion':['gini','entropy'],'max_depth': [None, 2, 3],
                                        'max_features': [None, 'sqrt', 'log2']}]),
            (AdaBoostClassifier(), [{'algorithm':['SAMME','SAMME.R'],
                                    'learning_rate': [0.1, 0.5, 1], 'n_estimators':[10,50,100]}]),
            (GradientBoostingClassifier(), [{'n_estimators':[10,100,1000],'criterion':['friedman_mse', 'mse'], 'learning_rate': [0.1, 0.5, 1],
                                            'loss':['deviance', 'exponential']}])]

encoder_scaler = [
    # ('Onehot',StandardScaler()),
    # ('Onehot',MinMaxScaler()),
    # ('Onehot',MaxAbsScaler()),
    # ('Onehot',RobustScaler()),
    ('Label',StandardScaler()),
    ('Label',MinMaxScaler()),
    ('Label',MaxAbsScaler()),
    ('Label',RobustScaler())
]
# X_train_res, y_train_res, X_test, y_test = preprocessing(df.copy(),'Label',StandardScaler())

# def runningModels(models):
#     result_list = []
#     for m in range(len(models)):
#         result = []
#         model = models[m][1]
#         model.fit(X_train_res, y_train_res)
#         y_pred = model.predict(X_test)

#         # Proceed to cross-validation with 10 folds
#         accuracies = cross_val_score(estimator=model, X=X_train_res, y=y_train_res, cv=10)
#         print('<<<<<<<<<<', models[m][0], '>>>>>>>>>>')
#         print(confusion_matrix(y_test, y_pred))
#         print('Accuracy Score: ', accuracy_score(y_test, y_pred))
#         print('K-Fold Score: ', accuracies)
#         print('K-Fold Validation Mean Accuracy: {:.2f} %'.format(accuracies.mean() * 100))
#         print('ROC AUC Score: {:.2f} %'.format(roc_auc_score(y_test, y_pred) * 100))
#         print('Precision: {:.2f} %'.format(precision_score(y_test, y_pred) * 100))
#         print('Recall: {:.2f} %'.format(recall_score(y_test, y_pred) * 100))
#         print('F1 Score: {:.2f} %'.format(f1_score(y_test, y_pred) * 100))
#         print(classification_report(y_test, y_pred))
#         print('\n')

#         result.append(models[m][0])
#         result.append(accuracy_score(y_test, y_pred) * 100)
#         result.append(accuracies.mean() * 100)
#         result.append(roc_auc_score(y_test, y_pred) * 100)
#         result.append(precision_score(y_test, y_pred) * 100)
#         result.append(recall_score(y_test, y_pred) * 100)
#         result.append(f1_score(y_test, y_pred) * 100)
#         result_list.append(result)

#     result_df = pd.DataFrame(result_list, columns=['Model', 'Accuracy', 'K-Fold Mean Accuracy', 'ROC_AUC', 'Precision', 'Recall', 'F1 Score'])
#     return result_df


def gridsearch(enc_scal,g_models):
    result_list = []
    # Proceed with hyperparameter tuning through grid search.
    for encoder, scaler in enc_scal:
        X_train_res, y_train_res, X_test, y_test = preprocessing(df.copy(),encoder,scaler)
        m=0
        result_list = []
        for model, param in g_models:
            result = []
            grid = GridSearchCV(estimator=model, param_grid=param, scoring='accuracy', cv=5, n_jobs=-1)
            grid.fit(X_train_res, y_train_res)
            print(' {}: \n Best Accuracy: {:.2f} %'.format(model, grid.best_score_ * 100))
            print('\n Best Parameter : {}', grid.best_params_)

            # predict with best model and calculate MSE
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)

            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            # visualizationROC(false_positive_rate,true_positive_rate)

            result.append(models[m][0])
            result.append(accuracy_score(y_test, y_pred) * 100)
            result.append(roc_auc_score(y_test, y_pred) * 100)
            result.append(precision_score(y_test, y_pred) * 100)
            result.append(recall_score(y_test, y_pred) * 100)
            result.append(f1_score(y_test, y_pred) * 100)
            result_list.append(result)
            m = m + 1
            result_df = pd.DataFrame(result_list, columns=['Model', 'Accuracy', 'ROC_AUC', 'Precision', 'Recall', 'F1 Score'])
        print(encoder, scaler)
        print(result_df)


# visualizationAccuracy(runningModels(models))
gridsearch(encoder_scaler,g_models)