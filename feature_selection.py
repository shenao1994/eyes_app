# logistic regression for feature importance
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import csv
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, accuracy_score, auc, confusion_matrix, \
    classification_report, precision_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def train_model(excel_path):
    # excel_path = 'Data/5. 术前眼球内陷、复视组合（697人）-不含v2.0内容1xlsx.xlsx'
    features_data = pd.read_excel(excel_path, sheet_name='Sheet1')
    X = features_data.iloc[:, 2:]
    # X = StandardScaler().fit_transform(X)
    # X = pd.DataFrame(X)
    y = features_data['label']
    # print(X.shape)
    # print(y.shape)
    # X = X.reshape(1, -1)
    # y = np.array(y).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg = LogisticRegression()
    reg.fit(X, y)
    y_pred = reg.predict_proba(X_test)
    # Visualize all regression plots
    # wandb.sklearn.plot_roc(y_test, y_pred, ['0', '1'])
    # return y_test, y_pred
    # skplt.metrics.plot_calibration_curve(y_test, [y_pred], ['LR'])
    prob_true, prob_pred = calibration_curve(y_test, y_pred[:, 1], normalize=True, n_bins=10)
    return prob_true, prob_pred
    # plt.plot(prob_pred, prob_true, marker='.', label='LR Curve')
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Ideal Curve')
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('True Probability in each Bin')
    # plt.legend()
    # plt.show()


def feat_sel(excel_path):
    # excel_path = 'Data/task5_train.csv'
    features_data = pd.read_csv(excel_path)
    X = features_data.iloc[:, 2:]
    feature_names = X.columns
    scalered_X = StandardScaler().fit_transform(X)
    scalered_X = pd.DataFrame(scalered_X)
    y = features_data['label']
    # print(X)
    scalered_X.columns = feature_names
    # print(feature_names)
    # define the model
    # model = LogisticRegression()
    MI_score = mutual_info_classif(scalered_X, y, random_state=0)
    # fit the model
    # model.fit(X, y)
    # get importance
    # importance = model.coef_[0]
    # summarize feature importance
    # print(np.argsort(MI_score)[::-1])
    sorted_index = np.argsort(MI_score)[::-1]
    feature_names = feature_names[sorted_index]
    MI_score = MI_score[sorted_index]
    return feature_names, MI_score


def read_output(csv_path):
    label_list = []
    pred_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, dialect='excel')
        for row in reader:
            # print(row['label'])
            label_list.append(int(row['label']))
            pred_list.append(float(row['output']))
    return label_list, pred_list


def draw_model_roc():
    model_name_list = ['svm', 'lr', 'knn']
    # 循环读取数据
    # modal = 'svm'
    auc_list = []
    fpr_list = []
    tpr_list = []
    for model_name in model_name_list:
        # csv_path = 'result/singleVSall_results/ML_results_temp/{model}_test_output_{num}.csv'.format(model=modal, num=model_name)
        # csv_path = 'result/ann_validation_output_{cls}.csv'.format(cls=model_name)
        # csv_path = 'result/singleVSall_results/ML_img_results/training_output_{num}.csv'.format(num=model_name)
        # csv_path = 'result/20231101/DL_results_temp/RDINet+Clinic_validation_output_{model}.csv'.format(model=model_name)
        csv_path = 'Data/results/{model}_test_output_task5.csv'.format(model=model_name)
        y, x = read_output(csv_path)
        x = np.array(x)
        # test_pred = np.where((x > 0.5) | (x == 0.5), 1, 0)
        # test_pred = test_pred.tolist()
        # print(test_pred)
        # test_accuracy = accuracy_score(y, test_pred)
        # test_auc = roc_auc_score(y, x)
        # test_sen = recall_score(y, test_pred)
        # test_spe = recall_score(y, test_pred, pos_label=0)
        # ppv = precision_score(y, test_pred)
        # npv = precision_score(y, test_pred, pos_label=0)
        # print(classification_report(y, test_pred, digits=4))
        # print('-----------------------------------')
        fpr, tpr, thresholds = roc_curve(y, x)
        auc_result = roc_auc_score(y, x)
        auc_list.append(auc_result)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    return model_name_list, auc_list, fpr_list, tpr_list


# train_model()