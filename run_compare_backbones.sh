#!/bin/bash

cd code
# Split the data into train and test sets in 5 different ways
python main.py --task train_test_splits --exp_name final1
python main.py --task train_test_splits --exp_name final2
python main.py --task train_test_splits --exp_name final3
python main.py --task train_test_splits --exp_name final4
python main.py --task train_test_splits --exp_name final5

# Balance and augment the data using CTGAN for 5 experiments
python main.py --task augment_gan --sil_tag nosil --exp_name final1
python main.py --task augment_gan --sil_tag nosil --exp_name final2
python main.py --task augment_gan --sil_tag nosil --exp_name final3
python main.py --task augment_gan --sil_tag nosil --exp_name final4
python main.py --task augment_gan --sil_tag nosil --exp_name final5

# Train 5 different variants of MLP classifier on the augmented data for each experiment

# Experiment 1
python main.py --task train_classifier --backbone mlp --mlp_version 1 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final1
python main.py --task train_classifier --backbone mlp --mlp_version 2 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final1
python main.py --task train_classifier --backbone mlp --mlp_version 3 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final1
python main.py --task train_classifier --backbone mlp --mlp_version 4 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final1
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final1

# Experiment 2
python main.py --task train_classifier --backbone mlp --mlp_version 1 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final2
python main.py --task train_classifier --backbone mlp --mlp_version 2 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final2
python main.py --task train_classifier --backbone mlp --mlp_version 3 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final2
python main.py --task train_classifier --backbone mlp --mlp_version 4 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final2
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final2

# Experiment 3
python main.py --task train_classifier --backbone mlp --mlp_version 1 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final3
python main.py --task train_classifier --backbone mlp --mlp_version 2 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final3
python main.py --task train_classifier --backbone mlp --mlp_version 3 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final3
python main.py --task train_classifier --backbone mlp --mlp_version 4 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final3
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final3

# Experiment 4
python main.py --task train_classifier --backbone mlp --mlp_version 1 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final4
python main.py --task train_classifier --backbone mlp --mlp_version 2 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final4
python main.py --task train_classifier --backbone mlp --mlp_version 3 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final4
python main.py --task train_classifier --backbone mlp --mlp_version 4 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final4
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final4

# Experiment 5
python main.py --task train_classifier --backbone mlp --mlp_version 1 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final5
python main.py --task train_classifier --backbone mlp --mlp_version 2 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final5
python main.py --task train_classifier --backbone mlp --mlp_version 3 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final5
python main.py --task train_classifier --backbone mlp --mlp_version 4 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final5
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name final5

