from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam

import numpy as np
from . import common
from . import cf_util
from tensorflow.keras.callbacks import EarlyStopping

import os
from nice import NICE
import pandas as pd


class MyEnsemble:
    def __init__(self, models, model_scores, aggregate_model_score):
        self.models = models
        self.model_scores = model_scores
        self.aggregate_model_score = aggregate_model_score

    def predict(self, X):
        Ypred = np.zeros(X.shape[0])
        if self.aggregate_model_score == 0:
            for i in range(len(self.models)):
                Ypred += self.models[i].predict(X).reshape(-1, )
            Ypred /= len(self.models)

            return np.array([[1-y, y] for y in Ypred])

        for i in range(len(self.models)):
            if self.model_scores[i] > 0.7:
                Ypred += self.models[i].predict(X).reshape(-1, ) * self.model_scores[i]
        Ypred /= self.aggregate_model_score
        return np.array([[1-y, y] for y in Ypred])

def dense_model(input_shape, denses, activations):
    model = Sequential()
    n = len(denses)
    if n != len(activations):
        raise ValueError('The number of denses and activations should be the same')
    model.add(Dense(denses[0], activation=activations[0], input_shape=input_shape))
    for i in range(1, n):
        model.add(Dense(denses[i], activation=activations[i]))
    return model

def get_model(model_type, dropout, regularization, input_shape):
    if model_type == 'mlp_1':
        denses = [34, 64, 18, 12, 8, 1]
        activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
        return dense_model(input_shape, denses, activations)

    elif model_type == 'mlp_2':
        denses = [34, 64, 32, 18, 12, 8, 1]
        activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
        return dense_model(input_shape, denses, activations)
    elif model_type == 'mlp_3':
        denses = [34, 32, 32, 12, 8, 1]
        activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
        return dense_model(input_shape, denses, activations)
    elif model_type == 'mlp_4':
        denses = [34, 32, 24, 24, 24, 18, 16, 12, 8, 1]
        activations = ['relu', 'relu', 'relu','relu','relu','relu','relu', 'relu', 'relu', 'sigmoid']
        return dense_model(input_shape, denses, activations)
    elif model_type == 'mlp_5':
        denses = [31, 32, 24, 18, 16, 12, 8, 1]
        activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
        return dense_model(input_shape, denses, activations)

def save_history(history, output_folder, datestamp, outcome_name, model_type, data_tag, dropout, num_epochs, learning_rate, regularization, validation_method, fold):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    #epochs = len(train_loss)
    filename = f'{output_folder}/History_{outcome_name}_{datestamp}_MOD_{model_type}_DAT_{data_tag}_DROP_{dropout}_EP_{num_epochs}_LR_{learning_rate}_RG_{regularization}_VAL_{validation_method}_FLD_{fold+1}.csv'
    df = pd.DataFrame(data={'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc})
    df.to_csv(filename, index=False)


def save_weights(model, output_folder, datestamp, outcome_name, model_type, data_tag, dropout, num_epochs, learning_rate, regularization, validation_method, fold):
    filename = f'{output_folder}/Weights_{outcome_name}_{datestamp}_MOD_{model_type}_DAT_{data_tag}_DROP_{dropout}_EP_{num_epochs}_LR_{learning_rate}_RG_{regularization}_VAL_{validation_method}_FLD_{fold+1}.h5'
    model.save_weights(filename)

def train_and_test_AIMEN(Xtrain, Ytrain, Xtest, Ytest, outcome_name, output_path, feature_names, classifier_config, data_tag):
    """
    This function trains and tests the MLP model on the FRI dataset. It uses k-fold cross validation. It also prints the results in a csv file.
    :param Xtrain:
    :param Ytrain:
    :param Xtest:
    :param Ytest:
    :param outcome_name:
    :param basepath:
    :return: None
    """
    #print('Im here in the beginning')
    txt = 'Model,Data tag,Dropout,Epochs,Learning rate,Regularization,Loss function,Optimizer,Validation_Method,Train_val_or_test,Fold,Loss,Accuracy,Abnormal class recall (sensitivity),Normal class recall (specificity),'+\
          'Abnormal class precision,Normal class precision,Abnormal class f1,Normal class f1,Macro average f1,AUROC\n'
    model_type = f"{classifier_config['backbone']}_{classifier_config['mlp_version']}"

    dropout = classifier_config['dropout'] == 'True'
    learning_rate = float(classifier_config['learning_rate'])
    num_epochs = int(classifier_config['num_epochs'])
    regularization = classifier_config['regularization'] == 'True'
    loss_function = 'binary_crossentropy'
    optimizer_name = 'Adam'
    k = int(classifier_config['validation_folds'])
    validation_method=f'{k}-fold XV'

    #k - fold cross validation

    models = [None for i in range(k)]
    model_scores = [0 for i in range(k)]
    test_losses = []
    Ypred_test_weighted = np.zeros(Ytest.shape)
    test_loss_weighted = 0
    Ypred_test_unweighted = np.zeros(Ytest.shape)
    test_loss_unweighted = 0
    aggregate_model_score = 0
    datestamp = common.get_datestamp()
    output_folder = f'{output_path}/fri_classification_results_{datestamp}/'
    os.mkdir(output_folder)
    output_file_path = f'{output_folder}/Summary_{outcome_name}_{datestamp}.csv'
    with open(output_file_path, 'w') as f:
        f.write(txt)
        txt = ''


    for fold in range(k):
        fold_start = fold * int(Xtrain.shape[0] / k)
        fold_end = (fold + 1) * int(Xtrain.shape[0] / k)
        Xtrain_fold = np.concatenate((Xtrain[:fold_start], Xtrain[fold_end:]), axis=0)
        Ytrain_fold = np.concatenate((Ytrain[:fold_start], Ytrain[fold_end:]), axis=0)
        Xval_fold = Xtrain[fold_start:fold_end]
        Yval_fold = Ytrain[fold_start:fold_end]

        models[fold]= get_model(model_type, dropout, regularization, (Xtrain.shape[1],))

        callback = EarlyStopping(monitor='loss', patience = 10)

        models[fold].compile(loss=loss_function, metrics=['accuracy'], optimizer=Adam(learning_rate=learning_rate))
        history = models[fold].fit(Xtrain_fold, Ytrain_fold, validation_data=(Xval_fold,Yval_fold), epochs = num_epochs,  callbacks=[callback])

        save_history(history, output_folder, datestamp, outcome_name, model_type, data_tag, dropout, num_epochs, learning_rate, regularization, validation_method, fold)
        save_weights(models[fold], output_folder, datestamp, outcome_name, model_type, data_tag, dropout, num_epochs, learning_rate, regularization, validation_method, fold)
        common.plot_learning_curve(history, output_folder, datestamp, outcome_name, fold, model_type)

        '''
        Printing the performance on the training set
        '''
        loss, acc = models[fold].evaluate(Xtrain_fold, Ytrain_fold)
        print(f"Fold {fold} Training Loss: {loss}, Acc: {acc}")
        Ypred_train_raw = models[fold].predict(Xtrain_fold).reshape(-1, )
        Ypred_train_fold = np.round(Ypred_train_raw).astype(int)
        report = common.plot_the_confusion_matrix(Ytrain_fold, Ypred_train_fold, output_folder,datestamp,
                                                  f'Training Confusion matrix (fold {fold + 1}) with {model_type} on the FRI dataset',
                                                  trainval_or_test='train_fold_' + str(fold + 1))
        train_auc_fold = common.get_auc(Ytrain_fold, Ypred_train_raw)
        txt+=f'{model_type},{data_tag},{dropout},{num_epochs},{learning_rate},{regularization},{loss_function},{optimizer_name},{validation_method},train,{fold+1},{loss},{report["accuracy"]},{report["1"]["recall"]},{report["0"]["recall"]},{report["1"]["precision"]},{report["0"]["precision"]},{report["1"]["f1-score"]},{report["0"]["f1-score"]},{report["macro avg"]["f1-score"]},{train_auc_fold}\n'

        with open(output_file_path, 'a') as f:
            f.write(txt)
            txt = ''

        '''
        Printing the performance on the validation set
        '''
        loss, acc = models[fold].evaluate(Xval_fold, Yval_fold)
        print(f"Validation (fold {fold + 1}) Loss: {loss}, Acc: {acc}")
        Ypred_val_raw = models[fold].predict(Xval_fold).reshape(-1, )
        Ypred_val_fold = np.round(Ypred_val_raw).astype(int)


        report = common.plot_the_confusion_matrix(Yval_fold, Ypred_val_fold, output_folder,datestamp,
                                                  f'Validation Confusion matrix (fold {fold+1}) with {model_type} on the FRI dataset', trainval_or_test='validation_fold_'+str(fold+1))
        val_auc_fold = common.get_auc(Yval_fold, Ypred_val_raw)
        txt+=f'{model_type},{data_tag},{dropout},{num_epochs},{learning_rate},{regularization},{loss_function},{optimizer_name},{validation_method},validation,{fold+1},{loss},{report["accuracy"]},{report["1"]["recall"]},{report["0"]["recall"]},{report["1"]["precision"]},{report["0"]["precision"]},{report["1"]["f1-score"]},{report["0"]["f1-score"]},{report["macro avg"]["f1-score"]},{val_auc_fold}\n'
        model_scores[fold] = report['macro avg']['f1-score']
        with open(output_file_path, 'a') as f:
            f.write(txt)
            txt = ''

        '''
        Calculating the loss and the predictions on the test set
        '''


        loss, acc = models[fold].evaluate(Xtest, Ytest)
        test_loss_unweighted += loss
        Ypred_test_unweighted += np.round(models[fold].predict(Xtest).reshape(-1, ))

        if model_scores[fold]> 0.7:
            test_loss_weighted += loss*model_scores[fold]
            Ypred_test_weighted += np.round(models[fold].predict(Xtest).reshape(-1, )) * model_scores[fold]
            aggregate_model_score += model_scores[fold]



    '''
    Handling the special cases if all the model scores are zero
    '''
    if aggregate_model_score==0:
        test_loss = test_loss_unweighted / k
        Y_pred_test_raw = Ypred_test_unweighted / k
        Ypred_test = np.round(Y_pred_test_raw).astype(int)

    else:
        test_loss = test_loss_weighted / aggregate_model_score
        Y_pred_test_raw = Ypred_test_weighted / aggregate_model_score
        Ypred_test = np.round(Y_pred_test_raw).astype(int)

    print(f"Test Loss: {test_loss}")
    common.print_metrics(Ytest, Ypred_test)
    report = common.plot_the_confusion_matrix(Ytest, Ypred_test, output_folder,datestamp,
                                              f'Test Confusion matrix with {model_type} on the FRI dataset', trainval_or_test='test')

    test_auc = common.get_auc(Ytest, Y_pred_test_raw)
    test_roc_metrics = common.get_roc_metrics_txt(model_type, data_tag, Ytest, Y_pred_test_raw)
    roc_metrics_path = f'{output_folder}/ROC_metrics_{outcome_name}_{datestamp}.csv'

    txt += f'{model_type},{data_tag},{dropout},{num_epochs},{learning_rate},{regularization},{loss_function},{optimizer_name},{validation_method},test,Weighted voting,{test_loss},{report["accuracy"]},{report["1"]["recall"]},{report["0"]["recall"]},{report["1"]["precision"]},{report["0"]["precision"]},{report["1"]["f1-score"]},{report["0"]["f1-score"]},{report["macro avg"]["f1-score"]},{test_auc}\n'

    ens = MyEnsemble(models, model_scores, aggregate_model_score)

    #print('Im here')
    with open(output_file_path, 'a') as f:
        f.write(txt)
        txt = ''

    with open(roc_metrics_path, 'a') as f:
        f.write(test_roc_metrics)

    cat_feat = list(range(23))+[28,29]
    num_feat = list(range(23,28))+list(range(30,34))
    NICE_FRI = NICE(
        X_train=np.round(Xtrain).astype(int),
        predict_fn=lambda x: ens.predict(x),
        y_train=Ytrain,
        cat_feat=cat_feat,
        num_feat=num_feat,
        distance_metric='HEOM',
        num_normalization='minmax',
        optimization='proximity',
        justified_cf=True
    )

    result_df = pd.DataFrame()
    #print(Ypred_test)
    #Explain only the abnormal classes
    indices = np.where(Ypred_test==1)[0]

    for step in range(len(indices)):
        idx = indices[step]
        print(f"Explaining instance {idx}. Serial: {step+1}/{len(indices)}")
        tempdf = explain_nice(NICE_FRI, Xtest[idx:idx+1], feature_names, Ypred_test[idx], ['Normal', 'Abnormal'])
        if tempdf is not None:
            result_df = pd.concat((result_df, tempdf), axis=0, ignore_index=True)
        else:
            print('No counterfactual found for instance ', idx)

    cf_filename = f'{output_folder}/NICE_{outcome_name}_{datestamp}.csv'
    result_df.to_csv(cf_filename, index=False)
    maxval = np.max(Xtrain, axis=0)
    minval = np.min(Xtrain, axis=0)

    df = pd.read_csv(cf_filename)
    cf_output = cf_util.find_metrics(df, minval, maxval, ens)
    cf_output_filename = f'{output_folder}/CF_metrics_{outcome_name}_{datestamp}.txt'
    with open(cf_output_filename, 'w') as f:
        f.write(cf_output)

def explain_nice(NICE_tool, to_explain, feature_names, predicted_class, class_names):
    CF = NICE_tool.explain(to_explain)
    if CF is None:
        return None
    df = pd.DataFrame(data=[to_explain[0].tolist() + [f'{class_names[predicted_class]} (Predicted)'],
                            CF[0].tolist() + [f'{class_names[1 - predicted_class]} (Counterfactual)']],
                      columns=feature_names + ['class'])
    #print(CF.shape)
    return df

def find_datestamp_from_params(output_path, outcome, model_type, data_tag, dropout, num_epochs, learning_rate, regularization, validation_method):
    history_substring =  f'MOD_{model_type}_DAT_{data_tag}_DROP_{dropout}_EP_{num_epochs}_LR_{learning_rate}_RG_{regularization}_VAL_{validation_method}_FLD_{1}.csv'
    for folder in os.listdir(output_path):
        if not os.path.isdir(f'{output_path}/{folder}'):
            continue
        if not folder.startswith('fri_classification_results'):
            continue

        files = os.listdir(f'{output_path}/{folder}')
        for file in files:
            if file.endswith(history_substring):
                return file.split('_')[2]
    return None

def test_AIMEN(Xtrain, Ytrain, Xtest, Ytest, outcome, output_path, feature_names, classifier_config,data_tag):
    model_type = f"{classifier_config['backbone']}_{classifier_config['mlp_version']}"
    dropout = classifier_config['dropout'] == 'True'
    learning_rate = float(classifier_config['learning_rate'])
    num_epochs = int(classifier_config['num_epochs'])
    regularization = classifier_config['regularization'] == 'True'
    loss_function = 'binary_crossentropy'
    optimizer_name = 'Adam'
    k = int(classifier_config['validation_folds'])
    validation_method = f'{k}-fold XV'

    train_datestamp = find_datestamp_from_params(output_path, outcome, model_type, data_tag, dropout, num_epochs, learning_rate, regularization, validation_method)
    now_datestamp = common.get_datestamp()

    txt = 'Model,Data tag,Dropout,Epochs,Learning rate,Regularization,Loss function,Optimizer,Validation_Method,Train_val_or_test,Fold,Loss,Accuracy,Abnormal class recall (sensitivity),Normal class recall (specificity),' + \
          'Abnormal class precision,Normal class precision,Abnormal class f1,Normal class f1,Macro average f1,AUROC\n'
    if train_datestamp is None:
        print('No model found with the given parameters')
        return
    else:
        print('Found a model with the given parameters. Model was trained on ', train_datestamp)
    output_folder = f'{output_path}/fri_classification_results_{train_datestamp}/'

    #print(datestamp)

    summary_file_path = f'{output_folder}/Summary_{outcome}_{train_datestamp}.csv'
    summary_file_path_for_test = f'{output_folder}/Summary_{outcome}_CLF_trained_{train_datestamp}_Results_saved_{now_datestamp}.csv'
    summary_df = pd.read_csv(summary_file_path)
    # Now we will extract the validation set macro_avg f1-scores from the summary file
    val_df = summary_df[summary_df['Train_val_or_test']=='validation']
    folds = val_df['Fold'].to_numpy().astype(int)
    model_scores_temp = val_df['Macro average f1'].to_numpy().astype(float)
    model_scores = [0 for i in range(k)]
    for i in range(len(folds)):
        fold = folds[i]
        model_scores[fold-1] = model_scores_temp[i]
    '''
    At this point, all the required info about the classifiers is loaded. Now we will load the models and test them on the test set
    '''

    models = [None for i in range(k)]
    model_scores = [0 for i in range(k)]
    test_losses = []
    Ypred_test_weighted = np.zeros(Ytest.shape)
    test_loss_weighted = 0
    Ypred_test_unweighted = np.zeros(Ytest.shape)
    test_loss_unweighted = 0
    aggregate_model_score = 0

    for fold in range(k):
        models[fold]= get_model(model_type, dropout, regularization, (Xtest.shape[1],))
        models[fold].compile(loss=loss_function, metrics=['accuracy'], optimizer=Adam(learning_rate=learning_rate))
        models[fold].load_weights(f'{output_folder}/Weights_{outcome}_{train_datestamp}_MOD_{model_type}_DAT_{data_tag}_DROP_{dropout}_EP_{num_epochs}_LR_{learning_rate}_RG_{regularization}_VAL_{validation_method}_FLD_{fold+1}.h5')

        '''
        Calculating the loss and the predictions on the test set
        '''
        loss, acc = models[fold].evaluate(Xtest, Ytest)
        test_loss_unweighted += loss
        Ypred_test_unweighted += np.round(models[fold].predict(Xtest).reshape(-1, ))

        if model_scores[fold]> 0.7:
            test_loss_weighted += loss*model_scores[fold]
            Ypred_test_weighted += np.round(models[fold].predict(Xtest).reshape(-1, )) * model_scores[fold]
            aggregate_model_score += model_scores[fold]

    '''
    Handling the special cases if all the model scores are zero
    '''
    if aggregate_model_score == 0:
        test_loss = test_loss_unweighted / k
        Y_pred_test_raw = Ypred_test_unweighted / k
        Ypred_test = np.round(Y_pred_test_raw).astype(int)

    else:
        test_loss = test_loss_weighted / aggregate_model_score
        Y_pred_test_raw = Ypred_test_weighted / aggregate_model_score
        Ypred_test = np.round(Y_pred_test_raw).astype(int)

    test_auc = common.get_auc(Ytest, Y_pred_test_raw)
    test_roc_metrics = common.get_roc_metrics_txt(model_type, data_tag, Ytest, Y_pred_test_raw)
    roc_metrics_path = f'{output_folder}/ROC_metrics_{outcome}_CLF_trained_{train_datestamp}_Results_saved_{now_datestamp}.csv'


    print(f"Test Loss: {test_loss}")
    common.print_metrics(Ytest, Ypred_test)
    report = common.plot_the_confusion_matrix_for_test(Ytest, Ypred_test, output_folder,train_datestamp,now_datestamp,
                                              f'Test Confusion matrix with {model_type} on the FRI dataset', trainval_or_test='test')
    txt+= f'{model_type},{data_tag},{dropout},{num_epochs},{learning_rate},{regularization},{loss_function},{optimizer_name},{validation_method},test,Weighted voting,{test_loss},{report["accuracy"]},{report["1"]["recall"]},{report["0"]["recall"]},{report["1"]["precision"]},{report["0"]["precision"]},{report["1"]["f1-score"]},{report["0"]["f1-score"]},{report["macro avg"]["f1-score"]},{test_auc}\n'
    print(txt)



    with open(summary_file_path_for_test, 'a') as f:
        f.write(txt)
        txt = ''

    with open(roc_metrics_path, 'a') as f:
        f.write(test_roc_metrics)
    #return
    ens = MyEnsemble(models, model_scores, aggregate_model_score)
    cat_feat = list(range(23)) + [28, 29]
    num_feat = list(range(23, 28)) + list(range(30, 34))
    NICE_FRI = NICE(
        X_train=np.round(Xtrain).astype(int),
        predict_fn=lambda x: ens.predict(x),
        y_train=Ytrain,
        cat_feat=cat_feat,
        num_feat=num_feat,
        distance_metric='HEOM',
        num_normalization='minmax',
        optimization='proximity',
        justified_cf=True
    )

    result_df = pd.DataFrame()
    # print(Ypred_test)
    # Explain only the abnormal classes
    indices = np.where(Ypred_test == 1)[0]

    for step in range(len(indices)):
        idx = indices[step]
        print(f"Explaining instance {idx}. Serial: {step + 1}/{len(indices)}")
        tempdf = explain_nice(NICE_FRI, Xtest[idx:idx + 1], feature_names, Ypred_test[idx], ['Normal', 'Abnormal'])
        if tempdf is not None:
            result_df = pd.concat((result_df, tempdf), axis=0, ignore_index=True)
        else:
            print('No counterfactual found for instance ', idx)

    cf_filename = f'{output_folder}/NICE_{outcome}_CLF_trained_{train_datestamp}_Results_saved_{now_datestamp}.csv'
    result_df.to_csv(cf_filename, index=False)
    maxval = np.max(Xtrain, axis=0)
    minval = np.min(Xtrain, axis=0)

    df = pd.read_csv(cf_filename)
    cf_output = cf_util.find_metrics(df, minval, maxval, ens)
    cf_output_filename = f'{output_folder}/CF_metrics_{outcome}_CLF_trained_{train_datestamp}_Results_saved_{now_datestamp}.txt'
    with open(cf_output_filename, 'w') as f:
        f.write(cf_output)
