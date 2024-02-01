import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from sklearn import svm
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, classification_report, accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from scipy.stats import levene, ttest_ind


def load_data():
    excel_path = 'data/20231128眼眶骨折数据/1. 眼球内陷（1100人）-不含v2.0的内容1.xlsx'
    clinic_path = 'data/20231031_148/副本20231020 T2影像组学-激素疗效预测.xlsx'
    label_path = 'data/20231031_148/20231020 T2_label.xlsx'
    # label_data = pd.read_excel(label_path)
    # label_data = label_data.iloc[:, :2]
    clinic_data = pd.read_excel(clinic_path)
    print(clinic_data)
    # features_data = pd.read_excel(excel_path, sheet_name='Sheet3')

    # features_data = features_data.merge(clinic_data, on=['name'])
    # features_data = clinic_data.merge(label_data, on=['name'])
    # features_data['name'] = features_data['name'].apply(change_label)
    # print(features_data)
    feature_selection(clinic_data)


# def change_label(x):
#     all_label = x.split('_')[-1]
#     return int(all_label[class_list.index(category_key)])

def change_error_value(x):
    if x == '/':
        return 0


def feature_selection(origin_data):
    # origin_data = shuffle(origin_data, random_state=4)
    features = origin_data[origin_data.columns[1:]]
    # features = pd.concat([features, clinic], axis=1)
    y = origin_data['Response（1=yes）']
    features = features.drop('Response（1=yes）', axis=1)
    colNames = features.columns
    # features = features.apply(func=lambda x: x.replace(x[3:7], "****"))
    # features = features.apply(change_error_value)
    features = features.replace('/', 0)
    features = features.replace('＜0.005', 0)
    features = features.replace('<0.005', 0)
    features = features.replace('1.0-', 0)
    features = features.replace('0.9+', 0)
    features = features.replace('0.9-', 0)
    data = features.astype(np.float64)
    data = data.fillna(0)
    data = StandardScaler().fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = colNames
    data['label'] = y
    print(data)
    selectedFeatutesList = []
    feature_names = []
    p_list = []
    for colName in data.columns[:-1]:
        if levene(data[data['label'] == 0][colName], data[data['label'] == 1][colName])[1] > 0.05 and \
                ttest_ind(data[data['label'] == 0][colName], data[data['label'] == 1][colName])[1] < 0.05:
            selectedFeatutesList.append(colName)
            p_list.append(ttest_ind(data[data['label'] == 0][colName], data[data['label'] == 1][colName])[1])
            feature_names.append(colName)
        elif levene(data[data['label'] == 0][colName], data[data['label'] == 1][colName])[1] <= \
                0.05 and ttest_ind(data[data['label'] == 0][colName],
                                   data[data['label'] == 1][colName], equal_var=False)[1] < 0.05:
            selectedFeatutesList.append(colName)
            p_list.append(ttest_ind(data[data['label'] == 0][colName], data[data['label'] == 1][colName])[1])
            feature_names.append(colName)
    if 'label' not in selectedFeatutesList: selectedFeatutesList = ['label'] + selectedFeatutesList
    print('TTest筛选后特征数量' + str(len(selectedFeatutesList)))
    # print(feature_names)
    # print(p_list)
    # return
    data1 = data[data['label'] == 0][selectedFeatutesList]
    data2 = data[data['label'] == 1][selectedFeatutesList]
    trainData = pd.concat([data1, data2])
    data = trainData[trainData.columns[1:]]
    y = trainData['label']
    # print(data)
    alphas = np.logspace(10, 20, 100)
    model_lassoCV = LassoCV(alphas=alphas, cv=5, max_iter=100000).fit(data, y)
    print(model_lassoCV.alpha_)
    print(model_lassoCV.intercept_)
    coe_list = []
    sel_threshold = 0
    for coe in model_lassoCV.coef_:
        if abs(coe) != sel_threshold:
            coe_list.append(coe)
    coef_table = {'features': data.columns[abs(model_lassoCV.coef_) != sel_threshold].tolist(),
                  'coefficient': coe_list}
    coef_pd = pd.DataFrame.from_dict(coef_table)
    formula_res = 'signature = '
    for col in coef_pd['features'].values.tolist():
        formula_res = formula_res + str(col) + "*" + str(
            coef_pd[coef_pd['features'] == col]['coefficient'].values[0])
    print(formula_res)
    coef = pd.Series(model_lassoCV.coef_, index=data.columns)
    print("Lasso picked " + str(sum(abs(coef) != sel_threshold)) + " variables and eliminated the other " + str(
        sum(abs(coef) == sel_threshold)) + ' variables')
    index = coef[abs(coef) != sel_threshold].index
    X = data[index]
    # X = pd.concat([X, clinic], axis=1)
    print(coef[abs(coef) != sel_threshold])
    MSEs = model_lassoCV.mse_path_
    MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
    MSEs_std = np.apply_along_axis(np.std, 1, MSEs)
    plt.figure()
    plt.errorbar(model_lassoCV.alphas_, MSEs_mean
                 , yerr=MSEs_std
                 , fmt='o'
                 , ms=3  # dot size
                 , mfc='r'  # dot color
                 , mec='r'  # dot margin color
                 , ecolor='lightblue'
                 , elinewidth=2  # error bar width
                 , capsize=4  # cap length of error bar
                 , capthick=1)
    plt.semilogx()
    plt.axvline(model_lassoCV.alpha_, color='black', ls='--', label='%.4e' % model_lassoCV.alpha_)
    plt.xlabel('Lamda')
    plt.ylabel('MSE')
    ax = plt.gca()
    y_major_locator = ticker.MultipleLocator(5)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.legend()
    # plt.savefig(
    #     'ML_result_img/{classModal}_lamda.pdf'.format(classModal=category_key),
    #     dpi=300)
    plt.show()
    coefs = lasso_path(data, y, alphas=alphas)[1].T
    plt.figure()
    plt.semilogx(model_lassoCV.alphas_, coefs, '-')
    plt.axvline(model_lassoCV.alpha_, color='black', ls='--', label='%.4e' % model_lassoCV.alpha_)
    plt.xlabel('Lamda')
    plt.ylabel('Coefficient')
    plt.legend()
    # plt.savefig(
    #     'ML_result_img/{classModal}_co_alphas.pdf'.format(classModal=category_key),
    #     dpi=300)
    plt.show()
    x_values = np.arange(len(index))
    y_values = coef[abs(coef) != sel_threshold]
    plt.bar(x_values, y_values
            , color='lightblue'
            , edgecolor='black'
            , alpha=0.8  # 不透明度
            )  # bar plot
    plt.xticks(x_values, index,
               rotation=45,
               ha='right',
               va='top')
    # plt.xlabel("Features")
    plt.ylabel("Coefficients")
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots_adjust(left=0.09, right=1, wspace=0.25, hspace=0.25, bottom=0.13, top=0.91)
    # plt.savefig('ML_result_img/{classModal}_sel_feat.pdf'.format(
    #     classModal=category_key), dpi=300, bbox_inches='tight')
    plt.show()
    # plt.figure(figsize=(9, 9), dpi=80)
    # print(X.corr())
    corr_sv_path = "ML_result/{modal}_corr0.csv".format(modal=category_key)
    X.corr().to_csv(corr_sv_path, header='true', encoding='utf-8')
    sns.heatmap(X.corr()
                , xticklabels=X.corr().columns
                , yticklabels=X.corr().columns
                , cmap='Blues'
                , center=0.5)  # 计算特征间的相关性
    plt.title('Correlogram of features', fontsize=20)
    plt.xticks(fontsize=10,
               rotation=45,
               ha='right',
               va='top')
    plt.yticks(fontsize=10,
               rotation=45,
               ha='right',
               va='top')
    # plt.savefig('ML_result_img/{classModal}_feats_corr.pdf'.format(
    #     classModal=category_key), dpi=300, bbox_inches='tight')
    plt.show()
    # selected_features = X
    # selected_features['CaseName'] = caseName
    # X['CaseName'] = caseName
    # print(selected_features)
    csv_path = "ML_result/{modal}_features0.csv".format(modal=category_key)
    label_path = "ML_result/{modal}_labels0.csv".format(modal=category_key)
    X['name'] = origin_data['name']
    if os.path.exists(csv_path):  # 如果文件存在
        os.remove(csv_path)
        X.to_csv(csv_path, header='true', encoding='utf-8', index=False)
        y.to_csv(label_path, header='true', encoding='utf-8', index=False)
    else:
        X.to_csv(csv_path, header='true', encoding='utf-8', index=False)
        y.to_csv(label_path, header='true', encoding='utf-8', index=False)
    # train_classifier(X, y, 0)


def train_classifier():
    # excel_path = 'data/20231128眼眶骨折数据/7. 术前眼球内陷、眼球运动障碍、复视组合（589人）-不含v2.0版本内容1.xlsx'
    # features_data = pd.read_excel(excel_path, sheet_name='Sheet1')
    X = pd.read_csv("ML_result/{modal}_features0.csv".format(modal=category_key))
    y = pd.read_csv("ML_result/{modal}_labels0.csv".format(modal=category_key))
    # X = features_data
    # y = features_data['label']
    # ros = RandomOverSampler(random_state=11)
    # x_smote_train, y_smote_train = ros.fit_resample(X, y)
    # smo = SMOTE(random_state=2)
    # x_smote_train, y_smote_train = smo.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    train_path = "ML_result/{modal}_train.csv".format(modal=category_key)
    test_path = "ML_result/{modal}_test.csv".format(modal=category_key)
    # train_path = "data/20231128眼眶骨折数据/train.xlsx"
    # test_path = "data/20231128眼眶骨折数据/test.xlsx"
    # train_data = pd.read_excel(train_path)
    # test_data = pd.read_excel(test_path)
    # smo = SMOTE(random_state=2)
    # x_smote_train, y_smote_train = smo.fit_resample(train_data.iloc[:, 2:], train_data['label'])
    out_train = X_train
    out_train['label'] = y_train
    out_test = X_test
    out_test['label'] = y_test
    if os.path.exists(train_path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(train_path)
        os.remove(test_path)
        out_train.to_csv(train_path, header='true', encoding='utf-8')
        out_test.to_csv(test_path, header='true', encoding='utf-8')
    else:
        out_train.to_csv(train_path, header='true', encoding='utf-8')
        out_test.to_csv(test_path, header='true', encoding='utf-8')
    # X_train.pop('name')
    # X_train.pop('label')
    # X_test.pop('name')
    # X_test.pop('label')
    # balanced Data
    # cc = ClusterCentroids(random_state=0)
    # x_smote_train, y_smote_train = cc.fit_resample(X_train, y_train)
    # print(Counter(y_resampled))
    # smo = SMOTE(random_state=2)
    # x_smote_train, y_smote_train = smo.fit_resample(X_train, y_train)

    # classifier_name = 'svm'
    # if classifier_name == 'svm':
    #     SVM_Classifier(x_smote_train, y_smote_train)
    # elif classifier_name == 'lr':
    #     LR_Classifier(x_smote_train, y_smote_train)
    # elif classifier_name == 'rf':
    #     RF_Classifier(x_smote_train, y_smote_train)
    # else:
    #     KNN_Classifier(x_smote_train, y_smote_train, test_data.iloc[:, 2:], test_data['label'])
    # test_model(classifier_name, test_data.iloc[:, 2:], test_data['label'])


def test_model(clf_name, X_test, y_test):
    model_path = 'ML_model/{classifier}_clf_{classModal}.pickle'.format(
        classifier=clf_name, classModal=category_key)
    with open(model_path, 'rb') as f:
        best_clf = pickle.load(f)
        y_test_probs = best_clf.predict_proba(X_test)
        # print(y_test_probs)
        test_auc = roc_auc_score(y_test, y_test_probs[:, 1])
        test_accuracy = best_clf.score(X_test, y_test)
        test_pred = best_clf.predict(X_test)
        test_sen = recall_score(y_test, test_pred)
        test_spe = recall_score(y_test, test_pred, pos_label=0)
        test_lower_roc, test_upper_roc = cal_ci_auc(np.array(y_test), np.array(y_test_probs[:, 1]), test_pred, 'auc')
        test_lower_acc, test_upper_acc = cal_ci_auc(np.array(y_test), np.array(y_test_probs[:, 1]), test_pred, 'acc')
        test_lower_sen, test_upper_sen = cal_ci_auc(np.array(y_test), np.array(y_test_probs[:, 1]), test_pred, 'sen')
        test_lower_spe, test_upper_spe = cal_ci_auc(np.array(y_test), np.array(y_test_probs[:, 1]), test_pred, 'spe')
        print('test_auc = %.4f(%.4f - %.4f)' % (test_auc, test_lower_roc, test_upper_roc))
        print('test_accuracy = %.4f(%.4f - %.4f)' % (test_accuracy, test_lower_acc, test_upper_acc))
        print('test_sen = %.4f(%.4f - %.4f)' % (test_sen, test_lower_sen, test_upper_sen))
        print('test_spe = %.4f(%.4f - %.4f)' % (test_spe, test_lower_spe, test_upper_spe))
        print(classification_report(y_test, test_pred, digits=4))
        # test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_probs[:, 1])
        # print(np.array(y_test))
        write_output_to_csv(np.array(y_test).flatten().tolist(), y_test_probs[:, 1],
                            'ML_result/{classifier}_test_output_{classModal}.csv'.
                            format(classifier=clf_name, classModal=category_key), 'label', 'output')


def SVM_Classifier(x_smote_train, y_smote_train):
    Cs = np.logspace(-5, 3, 20, base=2)
    gammas = np.logspace(-5, 1, 50, base=2)
    param_grid = dict(C=Cs, gamma=gammas)
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=5, scoring='roc_auc'). \
        fit(x_smote_train, y_smote_train)
    C = grid.best_params_['C']
    gamma = grid.best_params_['gamma']
    print(C, gamma)
    model_best = svm.SVC(C=C, kernel='rbf', gamma=gamma, probability=True).fit(x_smote_train, y_smote_train)
    # model_best = svm.SVC(C=10, gamma='auto', probability=True).fit(x_smote_train, y_smote_train)
    training_metric(model_best, x_smote_train, y_smote_train, 'svm')
    rkf = RepeatedKFold(n_splits=5, n_repeats=1)
    val_accs = []
    val_aucs = []
    val_sen = []
    val_spe = []
    val_lower_rocs = []
    val_upper_rocs = []
    val_lower_accs = []
    val_upper_accs = []
    val_lower_sens = []
    val_upper_sens = []
    val_lower_spes = []
    val_upper_spes = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    val_y_list = []
    val_pred_list = []
    for train_index, val_index in rkf.split(x_smote_train):
        X_cv_train = x_smote_train.iloc[train_index]
        X_val = x_smote_train.iloc[val_index]
        y_cv_train = y_smote_train.iloc[train_index]
        y_val = y_smote_train.iloc[val_index]
        model_svm = svm.SVC(C=C, kernel='rbf', gamma=gamma, probability=True).fit(X_cv_train, y_cv_train)
        # model_svm = svm.SVC(gamma='auto', probability=True).fit(X_cv_train, y_cv_train)
        val_acc = model_svm.score(X_val, y_val)
        y_probs = model_svm.predict_proba(X_val)
        y_pred = model_svm.predict(X_val)
        sen = recall_score(y_val, y_pred)
        spec = recall_score(y_val, y_pred, pos_label=0)
        val_fpr, val_tpr, val_thresholds = roc_curve(y_val, y_probs[:, 1])
        val_auc = roc_auc_score(y_val, y_probs[:, 1])
        val_y_list.extend(np.array(y_val).flatten().tolist())
        val_pred_list.extend(y_probs[:, 1])
        interp_tpr = np.interp(mean_fpr, val_fpr, val_tpr)
        interp_tpr[0] = 0.0
        val_lower_roc, val_upper_roc = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'auc')
        val_lower_acc, val_upper_acc = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'acc')
        val_lower_sen, val_upper_sen = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'sen')
        val_lower_spe, val_upper_spe = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'spe')
        val_lower_rocs.append(val_lower_roc)
        val_upper_rocs.append(val_upper_roc)
        val_lower_accs.append(val_lower_acc)
        val_upper_accs.append(val_upper_acc)
        val_lower_sens.append(val_lower_sen)
        val_upper_sens.append(val_upper_sen)
        val_lower_spes.append(val_lower_spe)
        val_upper_spes.append(val_upper_spe)
        tprs.append(interp_tpr)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        val_sen.append(sen)
        val_spe.append(spec)
    # mean_auc = auc(mean_fpr, mean_tpr)
    val_metric(val_aucs, val_accs, val_sen, val_spe, val_lower_rocs, val_upper_rocs, val_lower_accs, val_upper_accs,
               val_lower_sens, val_upper_sens, val_lower_spes, val_upper_spes, tprs, mean_fpr, 'svm')
    write_output_to_csv(val_y_list, val_pred_list, 'ML_result/{clf}_validation_output_{classModal}.csv'.
                        format(clf='svm', classModal=category_key), 'label', 'output')
    model_path = 'ML_model/svm_clf_{classModal}.pickle'.format(classModal=category_key)
    if os.path.exists(model_path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model_best, f)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model_best, f)


def LR_Classifier(x_train, y_train):
    # 参数设置
    params = {'C': [0.0001, 1, 100, 1000],
              'max_iter': [1, 10, 100, 500],
              'class_weight': ['balanced', None],
              'solver': ['liblinear', 'sag', 'lbfgs', 'newton-cg']
              }
    lr = LogisticRegression()
    clf = GridSearchCV(lr, param_grid=params, cv=5, scoring='roc_auc')
    clf.fit(x_train, y_train)
    model_best = LogisticRegression(**clf.best_params_).fit(x_train, y_train)
    training_metric(model_best, x_train, y_train, 'lr')
    rkf = RepeatedKFold(n_splits=5, n_repeats=1)
    val_accs = []
    val_aucs = []
    val_sen = []
    val_spe = []
    val_lower_rocs = []
    val_upper_rocs = []
    val_lower_accs = []
    val_upper_accs = []
    val_lower_sens = []
    val_upper_sens = []
    val_lower_spes = []
    val_upper_spes = []
    tprs = []
    val_y_list = []
    val_pred_list = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_index, val_index in rkf.split(x_train):
        X_cv_train = x_train.iloc[train_index]
        X_val = x_train.iloc[val_index]
        y_cv_train = y_train.iloc[train_index]
        y_val = y_train.iloc[val_index]
        model_lr = LogisticRegression(**clf.best_params_).fit(X_cv_train, y_cv_train)
        val_acc = model_lr.score(X_val, y_val)
        y_probs = model_lr.predict_proba(X_val)
        y_pred = model_lr.predict(X_val)
        sen = recall_score(y_val, y_pred)
        spec = recall_score(y_val, y_pred, pos_label=0)
        val_fpr, val_tpr, val_thresholds = roc_curve(y_val, y_probs[:, 1])
        val_auc = roc_auc_score(y_val, y_probs[:, 1])
        val_y_list.extend(np.array(y_val).flatten().tolist())
        val_pred_list.extend(y_probs[:, 1])
        interp_tpr = np.interp(mean_fpr, val_fpr, val_tpr)

        interp_tpr[0] = 0.0
        val_lower_roc, val_upper_roc = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'auc')
        val_lower_acc, val_upper_acc = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'acc')
        val_lower_sen, val_upper_sen = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'sen')
        val_lower_spe, val_upper_spe = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'spe')
        val_lower_rocs.append(val_lower_roc)
        val_upper_rocs.append(val_upper_roc)
        val_lower_accs.append(val_lower_acc)
        val_upper_accs.append(val_upper_acc)
        val_lower_sens.append(val_lower_sen)
        val_upper_sens.append(val_upper_sen)
        val_lower_spes.append(val_lower_spe)
        val_upper_spes.append(val_upper_spe)
        tprs.append(interp_tpr)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        val_sen.append(sen)
        val_spe.append(spec)
    # mean_auc = auc(mean_fpr, mean_tpr)
    # val_metric(val_aucs, val_accs, val_sen, val_spe, val_lower_rocs, val_upper_rocs, val_lower_accs, val_upper_accs,
    #            val_lower_sens, val_upper_sens, val_lower_spes, val_upper_spes, tprs, mean_fpr, 'lr')
    write_output_to_csv(val_y_list, val_pred_list, 'ML_result/{clf}_validation_output_{classModal}.csv'.
                        format(clf='lr', classModal=category_key), 'label', 'output')
    model_path = 'ML_model/lr_clf_{classModal}.pickle'.format(classModal=category_key)
    if os.path.exists(model_path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model_best, f)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model_best, f)


def RF_Classifier(x_train, y_train):
    model_best = RandomForestClassifier().fit(x_train, y_train)
    training_metric(model_best, x_train, y_train, 'rf')
    model_path = 'ML_model/rf_clf_{classModal}.pickle'.format(classModal=category_key)
    rkf = RepeatedKFold(n_splits=5, n_repeats=1)
    val_accs = []
    val_aucs = []
    val_sen = []
    val_spe = []
    val_lower_rocs = []
    val_upper_rocs = []
    val_lower_accs = []
    val_upper_accs = []
    val_lower_sens = []
    val_upper_sens = []
    val_lower_spes = []
    val_upper_spes = []
    tprs = []
    val_y_list = []
    val_pred_list = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_index, val_index in rkf.split(x_train):
        X_cv_train = x_train.iloc[train_index]
        X_val = x_train.iloc[val_index]
        y_cv_train = y_train.iloc[train_index]
        y_val = y_train.iloc[val_index]
        model_rf = RandomForestClassifier().fit(X_cv_train, y_cv_train)
        val_acc = model_rf.score(X_val, y_val)
        y_probs = model_rf.predict_proba(X_val)
        y_pred = model_rf.predict(X_val)
        sen = recall_score(y_val, y_pred)
        spec = recall_score(y_val, y_pred, pos_label=0)
        val_fpr, val_tpr, val_thresholds = roc_curve(y_val, y_probs[:, 1])
        val_auc = roc_auc_score(y_val, y_probs[:, 1])
        val_y_list.extend(np.array(y_val).flatten().tolist())
        val_pred_list.extend(y_probs[:, 1])
        interp_tpr = np.interp(mean_fpr, val_fpr, val_tpr)
        interp_tpr[0] = 0.0
        val_lower_roc, val_upper_roc = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'auc')
        val_lower_acc, val_upper_acc = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'acc')
        val_lower_sen, val_upper_sen = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'sen')
        val_lower_spe, val_upper_spe = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'spe')
        val_lower_rocs.append(val_lower_roc)
        val_upper_rocs.append(val_upper_roc)
        val_lower_accs.append(val_lower_acc)
        val_upper_accs.append(val_upper_acc)
        val_lower_sens.append(val_lower_sen)
        val_upper_sens.append(val_upper_sen)
        val_lower_spes.append(val_lower_spe)
        val_upper_spes.append(val_upper_spe)
        tprs.append(interp_tpr)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        val_sen.append(sen)
        val_spe.append(spec)
    # mean_auc = auc(mean_fpr, mean_tpr)
    # val_metric(val_aucs, val_accs, val_sen, val_spe, val_lower_rocs, val_upper_rocs, val_lower_accs, val_upper_accs,
    #            val_lower_sens, val_upper_sens, val_lower_spes, val_upper_spes, tprs, mean_fpr, 'rf')
    write_output_to_csv(val_y_list, val_pred_list, 'ML_result/{clf}_validation_output_{classModal}.csv'.
                        format(clf='rf', classModal=category_key), 'label', 'output')
    if os.path.exists(model_path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model_best, f)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model_best, f)


def KNN_Classifier(x_train, y_train, X_test, y_test):
    best_score = 0.0
    best_k = -1
    best_p = -1
    for k in range(1, 11):
        for p in range(1, 6):
            knn_clf = KNeighborsClassifier(n_neighbors=k, weights="uniform", p=p)
            knn_clf.fit(x_train, y_train)
            y_test_probs = knn_clf.predict_proba(X_test)
            test_auc = roc_auc_score(y_test, y_test_probs[:, 1])
            if test_auc > best_score:
                best_score = test_auc
                best_k = k
                best_p = p
    model_best = KNeighborsClassifier(n_neighbors=best_k, weights="uniform", p=best_p).fit(x_train, y_train)
    training_metric(model_best, x_train, y_train, 'knn')
    model_path = 'ML_model/knn_clf_{classModal}.pickle'.format(classModal=category_key)
    rkf = RepeatedKFold(n_splits=5, n_repeats=1)
    val_accs = []
    val_aucs = []
    val_sen = []
    val_spe = []
    val_lower_rocs = []
    val_upper_rocs = []
    val_lower_accs = []
    val_upper_accs = []
    val_lower_sens = []
    val_upper_sens = []
    val_lower_spes = []
    val_upper_spes = []
    tprs = []
    val_y_list = []
    val_pred_list = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_index, val_index in rkf.split(x_train):
        X_cv_train = x_train.iloc[train_index]
        X_val = x_train.iloc[val_index]
        y_cv_train = y_train.iloc[train_index]
        y_val = y_train.iloc[val_index]
        model_knn = KNeighborsClassifier(n_neighbors=5, weights="uniform", p=3).fit(X_cv_train, y_cv_train)
        val_acc = model_knn.score(X_val, y_val)
        y_probs = model_knn.predict_proba(X_val)
        y_pred = model_knn.predict(X_val)
        sen = recall_score(y_val, y_pred)
        spec = recall_score(y_val, y_pred, pos_label=0)
        val_fpr, val_tpr, val_thresholds = roc_curve(y_val, y_probs[:, 1])
        val_auc = roc_auc_score(y_val, y_probs[:, 1])
        val_y_list.extend(np.array(y_val).flatten().tolist())
        val_pred_list.extend(y_probs[:, 1])
        interp_tpr = np.interp(mean_fpr, val_fpr, val_tpr)
        interp_tpr[0] = 0.0
        val_lower_roc, val_upper_roc = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'auc')
        val_lower_acc, val_upper_acc = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'acc')
        val_lower_sen, val_upper_sen = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'sen')
        val_lower_spe, val_upper_spe = cal_ci_auc(np.array(y_val), np.array(y_probs[:, 1]), y_pred, 'spe')
        val_lower_rocs.append(val_lower_roc)
        val_upper_rocs.append(val_upper_roc)
        val_lower_accs.append(val_lower_acc)
        val_upper_accs.append(val_upper_acc)
        val_lower_sens.append(val_lower_sen)
        val_upper_sens.append(val_upper_sen)
        val_lower_spes.append(val_lower_spe)
        val_upper_spes.append(val_upper_spe)
        tprs.append(interp_tpr)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        val_sen.append(sen)
        val_spe.append(spec)
    # mean_auc = auc(mean_fpr, mean_tpr)
    # val_metric(val_aucs, val_accs, val_sen, val_spe, val_lower_rocs, val_upper_rocs, val_lower_accs, val_upper_accs,
    #            val_lower_sens, val_upper_sens, val_lower_spes, val_upper_spes, tprs, mean_fpr, 'knn')
    write_output_to_csv(val_y_list, val_pred_list, 'ML_result/{clf}_validation_output_{classModal}.csv'.
                        format(clf='knn', classModal=category_key), 'label', 'output')
    if os.path.exists(model_path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model_best, f)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model_best, f)


def val_metric(val_aucs, val_accs, val_sen, val_spe, val_lower_rocs, val_upper_rocs, val_lower_accs, val_upper_accs,
               val_lower_sens, val_upper_sens, val_lower_spes, val_upper_spes, tprs, mean_fpr, model):
    current_auc = sum(val_aucs) / len(val_aucs)
    current_acc = sum(val_accs) / len(val_accs)
    current_sen = sum(val_sen) / len(val_sen)
    current_spe = sum(val_spe) / len(val_spe)
    current_lower_auc = sum(val_lower_rocs) / len(val_lower_rocs)
    current_upper_auc = sum(val_upper_rocs) / len(val_upper_rocs)
    current_lower_acc = sum(val_lower_accs) / len(val_lower_accs)
    current_upper_acc = sum(val_upper_accs) / len(val_upper_accs)
    current_lower_sen = sum(val_lower_sens) / len(val_lower_sens)
    current_upper_sen = sum(val_upper_sens) / len(val_upper_sens)
    current_lower_spe = sum(val_lower_spes) / len(val_lower_spes)
    current_upper_spe = sum(val_upper_spes) / len(val_upper_spes)
    mean_tpr = np.mean(tprs, axis=0)
    write_output_to_csv(mean_fpr.tolist(), mean_tpr.tolist(), 'ML_result/{clf}_validation_output_{classModal}.csv'.
                        format(clf=model, classModal=category_key), 'fpr', 'tpr')
    print(
        "cv_val AUC: {:.4f}({:.4f} - {:.4f})  cv_val Accuracy: {:.4f}({:.4f} - {:.4f}) cv_val "
        "Sen: {:.4f}({:.4f} - {:.4f}) cv_val Spe: {:.4f}({:.4f} - {:.4f})".format(
            current_auc, current_lower_auc, current_upper_auc, current_acc, current_lower_acc, current_upper_acc,
            current_sen, current_lower_sen, current_upper_sen, current_spe, current_lower_spe, current_upper_spe
        ))


def training_metric(model, x_smote_train, y_smote_train, model_name):
    y_train_probs = model.predict_proba(x_smote_train)
    train_auc = roc_auc_score(y_smote_train, y_train_probs[:, 1])
    train_accuracy = model.score(x_smote_train, y_smote_train)
    train_pred = model.predict(x_smote_train)
    train_sen = recall_score(y_smote_train, train_pred)
    train_spe = recall_score(y_smote_train, train_pred, pos_label=0)
    # train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_probs[:, 1])
    training_lower_roc, training_upper_roc = cal_ci_auc(np.array(y_smote_train), np.array(y_train_probs[:, 1]),
                                                        train_pred, 'auc')
    training_lower_acc, training_upper_acc = cal_ci_auc(np.array(y_smote_train), np.array(y_train_probs[:, 1]),
                                                        train_pred, 'acc')
    training_lower_sen, training_upper_sen = cal_ci_auc(np.array(y_smote_train), np.array(y_train_probs[:, 1]),
                                                        train_pred, 'sen')
    training_lower_spe, training_upper_spe = cal_ci_auc(np.array(y_smote_train), np.array(y_train_probs[:, 1]),
                                                        train_pred, 'spe')
    print('train_auc = %.4f(%.4f - %.4f)' % (train_auc, training_lower_roc, training_upper_roc))
    print('train_accuracy = %.4f(%.4f - %.4f)' % (train_accuracy, training_lower_acc, training_upper_acc))
    print('train_sen = %.4f(%.4f - %.4f)' % (train_sen, training_lower_sen, training_upper_sen))
    print('train_spe = %.4f(%.4f - %.4f)' % (train_spe, training_lower_spe, training_upper_spe))
    write_output_to_csv(np.array(y_smote_train).tolist(), y_train_probs[:, 1],
                        'ML_result/{cls}_training_output_{classModal}.csv'.format(cls=model_name,
                                                                                  classModal=category_key),
                        'label', 'output')
    print(classification_report(y_smote_train, model.predict(x_smote_train), digits=4))


def cal_ci_auc(y_true, y_pred, pred_label, metric):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        if metric == 'auc':
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        elif metric == 'acc':
            score = accuracy_score(y_true[indices], pred_label[indices])
            bootstrapped_scores.append(score)
        elif metric == 'sen':
            score = recall_score(y_true[indices], pred_label[indices])
            bootstrapped_scores.append(score)
        elif metric == 'spe':
            score = recall_score(y_true[indices], pred_label[indices], pos_label=0)
            bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower, confidence_upper


def write_output_to_csv(label, predict, csv_name, header1, header2):
    out_df = pd.DataFrame({header1: label, header2: predict})
    csv_path = csv_name
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    if os.path.exists(csv_path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(csv_path)
        out_df.to_csv(csv_path, index=False, sep=',')
    else:
        out_df.to_csv(csv_path, index=False, sep=',')


if __name__ == '__main__':
    # class_list = ['1', '2', '3']
    category_key = 'clinic'
    load_data()
    # train_classifier()
