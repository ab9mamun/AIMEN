#!/bin/bash
exp=$1

cd code
# Split data into train and test sets
python main.py --task train_test_splits --exp_name $exp

# Data variant: Balance data with ADASYN
python main.py --task balance_adasyn --exp_name $exp

# Data variant: Augment data with ADASYN
python main.py --task augment_adasyn --exp_name $exp

# Data variant: Augment data with CTGAN (No Restrictions)
python main.py --task augment_gan --sil_tag nosil --exp_name $exp

# Data variant: Augment data with CTGAN (Restriction on the negative class)
python main.py --task augment_gan --sil_tag negsil --exp_name $exp

# Data variant: Augment data with CTGAN (Restriction on both classes)
python main.py --task augment_gan --sil_tag allsil --exp_name $exp

# Data variant: Augment data with CTGAN (Restriction on the positive class)
python main.py --task augment_gan --sil_tag possil --exp_name $exp 

# All variants of the data are now ready for training the different classifiers


# Train 5 different variants of MLP classifier on the augmented data by CTGAN (No Restrictions)
# (for more in-depth comparison of the different MLP versions, see run_compare_backbones.sh)
python main.py --task train_classifier --backbone mlp --mlp_version 1 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 2 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 3 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 4 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp

# Train with MLP_v5 on the different data variants with CTGAN
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_negsil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_allsil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_possil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp

# An example of how to test an already trained classifier without retraining
python main.py --task test_classifier --backbone mlp --mlp_version 5 --data_tag gan_augmented_nosil --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp

# Train 5 different variants of MLP classifier on the augmented data by ADASYN
python main.py --task train_classifier --backbone mlp --mlp_version 1 --data_tag adasyn_augmented --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 2 --data_tag adasyn_augmented --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 3 --data_tag adasyn_augmented --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 4 --data_tag adasyn_augmented --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag adasyn_augmented --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp

# Train 5 different variants of MLP classifier on the balanced data by ADASYN
python main.py --task train_classifier --backbone mlp --mlp_version 1 --data_tag adasyn_balanced --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 2 --data_tag adasyn_balanced --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 3 --data_tag adasyn_balanced --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 4 --data_tag adasyn_balanced --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
python main.py --task train_classifier --backbone mlp --mlp_version 5 --data_tag adasyn_balanced --learning_rate 0.0001 --validation_folds 8 --num_epochs 1000 --exp_name $exp
