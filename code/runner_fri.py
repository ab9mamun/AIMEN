
import datamanager.fri_data.fri_processor as processor
import datamanager.fri_data.fri_data_driver as dd
from sklearn.utils import shuffle
import models.mlp as mlp
from imblearn.over_sampling import ADASYN
import numpy as np
import models.gan_for_augmentation as gan
import os
import pandas as pd
from datamanager.saveutil import save_obj, load_obj, save_text
import  models.adasyn_for_augmentation as adasyn_helper



def create_train_test_sets(basepath, outcome, output_path):
    '''
    Preparing the data. At this point, we have the data in the form of a pandas dataframe. We need to convert it to numpy arrays.
    '''
    df = dd.get_fri_df(basepath)
    features, labels, feature_names = processor.create_features_with_all(df, outcome)

    # We may need to create the output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # So, now we have the features and labels. Let's now separate the training and test sets.


    # We need the categorical columns for the GAN
    categorical_columns = processor.get_categorical_columns()
    print(f'Categorical columns: {categorical_columns}')

    df = df.dropna()
    df = df[feature_names + [outcome]]
    real_negdf = df[df[outcome] == 0]
    real_posdf = df[df[outcome] == 1]

    real_negdf = real_negdf[feature_names]
    real_posdf = real_posdf[feature_names]

    neg_features = real_negdf.values
    pos_features = real_posdf.values

    neg_features = shuffle(neg_features, random_state=0)
    pos_features = shuffle(pos_features, random_state=0)

    testsize = int(pos_features.shape[0] * 0.25)
    pos_features_test = pos_features[:testsize]
    pos_features_train = pos_features[testsize:]
    neg_features_test = neg_features[:testsize]
    neg_features_train = neg_features[testsize:]

    train_features = np.concatenate((neg_features_train, pos_features_train), axis=0)
    train_labels = np.concatenate((np.zeros(neg_features_train.shape[0]), np.ones(pos_features_train.shape[0])), axis=0)
    test_features = np.concatenate((neg_features_test, pos_features_test), axis=0)
    test_labels = np.concatenate((np.zeros(neg_features_test.shape[0]), np.ones(pos_features_test.shape[0])), axis=0)

    save_obj(neg_features_train, os.path.join(output_path, 'neg_features_train.pkl'))
    save_obj(pos_features_train, os.path.join(output_path, 'pos_features_train.pkl'))
    save_obj(neg_features_test, os.path.join(output_path, 'neg_features_test.pkl'))
    save_obj(pos_features_test, os.path.join(output_path, 'pos_features_test.pkl'))

    save_obj(train_features, os.path.join(output_path, 'train_features.pkl'))
    save_obj(train_labels, os.path.join(output_path, 'train_labels.pkl'))
    save_obj(test_features, os.path.join(output_path, 'test_features.pkl'))
    save_obj(test_labels, os.path.join(output_path, 'test_labels.pkl'))

    save_obj(feature_names, os.path.join(output_path, 'feature_names.pkl'))
    save_obj(categorical_columns, os.path.join(output_path, 'categorical_columns.pkl'))


    output_txt = f'Train test files last updated: {pd.Timestamp.now()}\n'
    save_text(os.path.join(output_path, 'information.txt'), output_txt)

def balance_adasyn(basepath, outcome, output_path):
    '''
    This function is used to augment the data using the ADASYN algorithm.
    :param basepath:
    :param outcome:
    :return:
    '''

    train_features = load_obj(os.path.join(output_path, 'train_features.pkl'))
    train_labels = load_obj(os.path.join(output_path, 'train_labels.pkl'))

    X, Y = shuffle(train_features, train_labels, random_state=0)

    ada = ADASYN(random_state=42)

    print(f"Before oversampling: {X.shape}, {Y.shape}")
    print("Training set Normal: ", sum(Y == 0), "Abnormal: ", sum(Y == 1))
    master_lower_bound = 0
    categorical_columns = load_obj(os.path.join(output_path, 'categorical_columns.pkl'))
    feature_names = load_obj(os.path.join(output_path, 'feature_names.pkl'))

    X, Y = adasyn_helper.balance(ada, X, Y, feature_names, categorical_columns, master_lower_bound)
    print(f"After oversampling: {X.shape}, {Y.shape}")
    print("Training set Normal: ", sum(Y == 0), "Abnormal: ", sum(Y == 1))

    print(f'Outcome: {outcome}, TrainSize:{X.shape}')
    save_obj(X, os.path.join(output_path, 'train_features_adasyn_balanced.pkl'))
    save_obj(Y, os.path.join(output_path, 'train_labels_adasyn_balanced.pkl'))

    save_text(os.path.join(output_path, 'information.txt'), f'ADASYN balanced data last updated: {pd.Timestamp.now()}\n')


def augment_adasyn(basepath, outcome, output_path):
    '''
    This function is used to augment the data using the ADASYN algorithm.
    :param basepath:
    :param outcome:
    :return:
    '''

    #at first balance the data. Check if the balanced data is already available or not.

    X = load_obj(os.path.join(output_path, 'train_features_adasyn_balanced.pkl'))
    Y = load_obj(os.path.join(output_path, 'train_labels_adasyn_balanced.pkl'))

    if X is None or Y is None:
        print("Balanced data does not exist. Calling the balance_adasyn function.")
        balance_adasyn(basepath, outcome, output_path)
        X = load_obj(os.path.join(output_path, 'train_features_adasyn_balanced.pkl'))
        Y = load_obj(os.path.join(output_path, 'train_labels_adasyn_balanced.pkl'))

    feature_names = load_obj(os.path.join(output_path, 'feature_names.pkl'))
    categorical_columns = load_obj(os.path.join(output_path, 'categorical_columns.pkl'))
    master_lower_bound = 0

    ada = ADASYN(random_state=42)

    neg_indices = np.where(Y == 0)[0]
    pos_indices = np.where(Y == 1)[0]

    neg_features = X[neg_indices]
    pos_features = X[pos_indices]

    #divide the negative features into 5 parts then augment each part separately. finally don't add the original negative features
    #as that would create duplicates

    neg_features_1 = neg_features[:int(neg_features.shape[0]//5)]
    neg_features_2 = neg_features[int(neg_features.shape[0]//5):int(2*neg_features.shape[0]//5)]
    neg_features_3 = neg_features[int(2*neg_features.shape[0]//5):int(3*neg_features.shape[0]//5)]
    neg_features_4 = neg_features[int(3*neg_features.shape[0]//5):int(4*neg_features.shape[0]//5)]
    neg_features_5 = neg_features[int(4*neg_features.shape[0]//5):]

    X1, Y1 = combine_and_balance_with_adasyn(neg_features_1, pos_features, ada, feature_names, categorical_columns, master_lower_bound)
    X2, Y2 = combine_and_balance_with_adasyn(neg_features_2, pos_features, ada, feature_names, categorical_columns, master_lower_bound)
    X3, Y3 = combine_and_balance_with_adasyn(neg_features_3, pos_features, ada, feature_names, categorical_columns, master_lower_bound)
    X4, Y4 = combine_and_balance_with_adasyn(neg_features_4, pos_features, ada, feature_names, categorical_columns, master_lower_bound)
    X5, Y5 = combine_and_balance_with_adasyn(neg_features_5, pos_features, ada, feature_names, categorical_columns, master_lower_bound)

    #Extract only the examples where the label is 0. As the positive examples are duplicated.
    X1_neg = X1[np.where(Y1 == 0)[0]]
    X2_neg = X2[np.where(Y2 == 0)[0]]
    X3_neg = X3[np.where(Y3 == 0)[0]]
    X4_neg = X4[np.where(Y4 == 0)[0]]
    X5_neg = X5[np.where(Y5 == 0)[0]]

    #Combine all the negative examples
    X_neg_combined = np.concatenate((X1_neg, X2_neg, X3_neg, X4_neg, X5_neg), axis=0)

    #Now augment the positive examples with the negative examples


    X_aug, Y_aug = combine_and_balance_with_adasyn(X_neg_combined, pos_features, ada, feature_names, categorical_columns, master_lower_bound)

    print(f"Before adasyn augmentation: {X.shape}, {Y.shape}")
    print("Training set Normal: ", sum(Y == 0), "Abnormal: ", sum(Y == 1))

    print(f"After adasyn augmentation: {X_aug.shape}, {Y_aug.shape}")
    print("Training set Normal: ", sum(Y_aug == 0), "Abnormal: ", sum(Y_aug == 1))

    print(f'Outcome: {outcome}, TrainSize:{X_aug.shape}')
    save_obj(X_aug, os.path.join(output_path, 'train_features_adasyn_augmented.pkl'))
    save_obj(Y_aug, os.path.join(output_path, 'train_labels_adasyn_augmented.pkl'))

    save_text(os.path.join(output_path, 'information.txt'), f'ADASYN augmented data last updated: {pd.Timestamp.now()}\n')


def combine_and_balance_with_adasyn(neg_features, pos_features, ada, feature_names, categorical_columns, master_lower_bound):
    X = np.concatenate((neg_features, pos_features), axis=0)
    Y = np.concatenate((np.zeros(neg_features.shape[0]), np.ones(pos_features.shape[0])), axis=0)
    X, Y = shuffle(X, Y, random_state=0)
    X, Y = adasyn_helper.balance(ada, X, Y, feature_names, categorical_columns, master_lower_bound)
    return X, Y


def augment_gan(basepath, outcome, output_path, sil_tag):
    """
    This function is used to train the GAN model on the FRI dataset.
    :param basepath:
    :param outcome:
    :param balance_test_set:
    :return:
    """

    if sil_tag == 'None':
        print("Silhouette tag is None. So, we will not be training the GAN model.")
        return


    neg_features_train = load_obj(os.path.join(output_path, 'neg_features_train.pkl'))
    pos_features_train = load_obj(os.path.join(output_path, 'pos_features_train.pkl'))
    feature_names = load_obj(os.path.join(output_path, 'feature_names.pkl'))
    categorical_columns = load_obj(os.path.join(output_path, 'categorical_columns.pkl'))

    neg_features_traindf = pd.DataFrame(neg_features_train, columns=feature_names)
    pos_features_traindf = pd.DataFrame(pos_features_train, columns=feature_names)

    neg_features_gendf, pos_features_gendf = gan.bidcctgan_fri(basepath, feature_names, neg_features_traindf, pos_features_traindf, categorical_columns, output_path, sil_tag)
    Xtrain = np.concatenate((neg_features_gendf.values, pos_features_gendf.values), axis=0)
    Ytrain = np.concatenate((np.zeros(neg_features_gendf.shape[0]), np.ones(pos_features_gendf.shape[0])), axis=0)

    Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=0)

    save_obj(neg_features_gendf, os.path.join(output_path, f'neg_features_gan_augmented_{sil_tag}.pkl'))
    save_obj(pos_features_gendf, os.path.join(output_path, f'pos_features_gan_augmented_{sil_tag}.pkl'))
    save_obj(Xtrain, os.path.join(output_path, f'train_features_gan_augmented_{sil_tag}.pkl'))
    save_obj(Ytrain, os.path.join(output_path, f'train_labels_gan_augmented_{sil_tag}.pkl'))
    save_text(os.path.join(output_path, 'information.txt'), f'GAN weights and augmented data for sil_tag:{sil_tag} last updated: {pd.Timestamp.now()}\n')


def train_classifier(basepath, outcome, output_path, classifier_config, data_tag):
    """
    This function is used to train the neural network model on the FRI dataset.
    :param basepath:
    :param outcome:
    :return:
    """

    Xtrain = load_obj(os.path.join(output_path, f'train_features_{data_tag}.pkl'))
    Ytrain = load_obj(os.path.join(output_path, f'train_labels_{data_tag}.pkl'))

    Xtest = load_obj(os.path.join(output_path, 'test_features.pkl'))
    Ytest = load_obj(os.path.join(output_path, 'test_labels.pkl'))
    feature_names = load_obj(os.path.join(output_path, 'feature_names.pkl'))

    convert_to_int = False

    if convert_to_int:
        Xtrain = np.round(Xtrain).astype(int)
        Xtest = np.round(Xtest).astype(int)
    else:
        Xtrain = Xtrain.astype(float)  # now let's not round them to integers
        Xtest = Xtest.astype(float)

    Ytrain = np.round(Ytrain.astype(int))
    Ytest = np.round(Ytest.astype(int))

    Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=0)

    print(f'Outcome: {outcome}, Classifier: {classifier_config["backbone"]}, Datatag: {data_tag}, TrainSize:{Xtrain.shape}, TestSize:{Xtest.shape}')
    print(f'Ytrain shape:{Ytrain.shape}, Ytest shape:{Ytest.shape}')
    #return
    mlp.train_and_test_AIMEN(Xtrain, Ytrain, Xtest, Ytest, outcome, output_path, feature_names, classifier_config, data_tag)

def test_classifier(basepath, outcome, output_path, classifier_config, data_tag):
    """
    This function is used to test the neural network model on the FRI dataset.
    :param basepath:
    :param outcome:
    :return:
    """

    Xtrain = load_obj(os.path.join(output_path, f'train_features_{data_tag}.pkl'))
    Ytrain = load_obj(os.path.join(output_path, f'train_labels_{data_tag}.pkl'))

    Xtest = load_obj(os.path.join(output_path, 'test_features.pkl'))
    Ytest = load_obj(os.path.join(output_path, 'test_labels.pkl'))
    feature_names = load_obj(os.path.join(output_path, 'feature_names.pkl'))

    convert_to_int = False
    if convert_to_int:
        Xtrain = np.round(Xtrain).astype(int)
        Xtest = np.round(Xtest).astype(int)
    else:
        Xtrain = Xtrain.astype(float)  # now let's not round them to integers
        Xtest = Xtest.astype(float)

    Ytrain = np.round(Ytrain.astype(int))
    Ytest = np.round(Ytest.astype(int))

    print(f'Outcome: {outcome}, Classifier: {classifier_config["backbone"]}, Datatag: {data_tag}, TestSize:{Xtest.shape}')
    print(f'Ytest shape:{Ytest.shape}')
    # return
    mlp.test_AIMEN(Xtrain, Ytrain, Xtest, Ytest, outcome, output_path, feature_names, classifier_config,data_tag)