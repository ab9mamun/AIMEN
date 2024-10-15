#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abdullah
"""
import numpy as np
import runner_fri
import os
import argparse

def create_config(backbone, mlp_version, dropout, regularization, num_epochs, learning_rate, validation_folds, gpu_ids, batch_size):
    config = {
        'backbone': backbone,
        'mlp_version': mlp_version,
        'dropout': dropout,
        'regularization': regularization,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'validation_folds': validation_folds,
        'gpu_ids': gpu_ids,
        'batch_size': batch_size
    }
    return config
def main(args):
    """
    This is the main control panel that executes different tasks based on the arguments provided.
    Examples of different tasks that can be done are:
    1. Augment the dataset with Adasyn
    2. Train GAN for augmentation
    3. Augment the dataset with GAN
    4. Train a model on the augmented dataset
    5. Train a model on the original dataset
    6. Predict on the test set

    :param args:
    :return:
    """
    print(args)
    args = parser.parse_args()
    dataset = args.dataset
    task = args.task
    output_folder = args.output_folder
    experiment_name = args.exp_name
    backbone = args.backbone
    mlp_version = args.mlp_version
    dropout = args.dropout
    regularization = args.regularization
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    validation_folds = args.validation_folds
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    sil_tag = args.sil_tag
    data_tag = args.data_tag
    #print(dataset, output_folder, backbone, mlp_version, dropout, num_epochs, learning_rate, validation_folds, gpu_ids, batch_size)
    #return

    np.set_printoptions(suppress=True)
    basepath = f'../data/'
    output_path = os.path.join(basepath, output_folder, experiment_name)
    print('Basepath: ', basepath)
    print('Outputpath:', output_path)


    if task == 'train_test_splits':
        runner_fri.create_train_test_sets(basepath, 'HRCP', output_path)
    elif task == 'balance_adasyn':
        runner_fri.balance_adasyn(basepath, 'HRCP', output_path)
    elif task == 'augment_adasyn':
        runner_fri.augment_adasyn(basepath, 'HRCP', output_path)
    elif task == 'train_classifier':
        classifier_config = create_config(backbone, mlp_version, dropout, regularization, num_epochs, learning_rate, validation_folds, gpu_ids, batch_size)
        runner_fri.train_classifier(basepath, 'HRCP', output_path, classifier_config, data_tag)
    elif task == 'test_classifier':
        classifier_config = create_config(backbone, mlp_version, dropout, regularization, num_epochs, learning_rate, validation_folds, gpu_ids, batch_size)
        runner_fri.test_classifier(basepath, 'HRCP', output_path, classifier_config, data_tag)

    elif task == 'augment_gan':
        runner_fri.augment_gan(basepath, 'HRCP', output_path, sil_tag)

    elif task == 'None':
        print('Please choose a task to perform')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model on the FRI dataset.')

    parser.add_argument('--dataset',
                        default='fri',
                        type=str)
    parser.add_argument('--output_folder',
                        default='output',
                        help='The folder where the output will be stored',
                        type=str)
    parser.add_argument('--exp_name',
                        default='default_exp_name',
                        help='A unique name for the experiment. If not unique, the existing experiment will be overwritten.',
                        type=str)
    parser.add_argument('--task',
                        default='None',
                        help='Choose from train_classifier, test_classifier, augment_gan, augment_adasyn, train_test_splits, balance_adasyn',
                        type=str)
    parser.add_argument('--sil_tag',
                        default='None',
                        help='Choose from nosil, negsil, possil, allsil',
                        type=str)
    parser.add_argument('--data_tag',
                        default='None',
                        help='Choose from adasyn_balanced, adasyn_augmented, gan_augmented_nosil, gan_augmented_negsil, gan_augmented_possil, gan_augmented_allsil',
                        type=str)

    parser.add_argument('--outcome',
                        default='HRCP',
                        type=str)

    parser.add_argument('--backbone',
                        default='mlp',
                        help='Options: mlp, rforest, xgboost',
                        type=str)
    parser.add_argument('--mlp_version',
                        default='1',
                        help='Options: 1, 2, 3, 4, 5',
                        type=str)
    parser.add_argument('--dropout',
                        default='False',
                        help='choose dropout from true or false',
                        type=str)
    parser.add_argument('--regularization',
                        default='False',
                        help='choose regularization from True or False',
                        type=str)
    parser.add_argument('--learning_rate',
                        default='0.01',
                        help='Choose learning rate from a valid numeric value greater than 0 and less than 1',
                        type=str)
    parser.add_argument('--validation_folds',
                        default='8',
                        help='Choose number of folds for validation between 3 and 10. Default is 8.',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='Number of epochs to train the model',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)


    main(args=parser.parse_args())
