from ctgan import CTGAN
from sklearn.metrics import silhouette_score
import numpy as np
import os
import pandas as pd


def train_ctgan_fri(data, categorical_columns, output_path, purpose, generated_label, sil_tag, epochs=300, batch_size=500, log_frequency=True):
    """
    Train a CTGAN model on the given data and save it to the given output path.
    :param data: The data to train on.
    :param categorical_columns: The categorical columns in the data.
    :param output_path: The path to save the model to.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size to use.
    :param log_frequency: Whether to log the training frequency.
    :return: The trained CTGAN model.
    """
    ctgan_model = CTGAN(epochs=epochs, batch_size=batch_size, log_frequency=log_frequency)
    ctgan_model.fit(data, categorical_columns)
    ctgan_model.save(os.path.join(output_path, f'ctgan_model_{sil_tag}_{purpose}_generatedlabel_{generated_label}_epochs_{epochs}_batch_size_{batch_size}.pkl'))
    return ctgan_model


def generate_ctgan_samples(ctgan_model, num_samples):
    """
    Generate samples from a CTGAN model.
    :param ctgan_model: The CTGAN model to generate samples from.
    :param num_samples: The number of samples to generate.
    :return: The generated samples.
    """
    return ctgan_model.sample(num_samples)


def calculate_silhouette_score(generated, real_neg, pos_so_far, generated_label):
    """
    Compute the silhouette score of the given data and labels.
    :param X: The data.
    :param labels: The labels.
    :return: The silhouette score.
    """
    X = np.concatenate((generated.values, real_neg.values, pos_so_far.values))
    if generated_label == 1:
        labels = np.concatenate((np.ones(len(generated)), np.zeros(len(real_neg)), np.ones(len(pos_so_far))))
    elif generated_label == 0:
        labels = np.concatenate((np.zeros(len(generated)), np.zeros(len(real_neg)), np.ones(len(pos_so_far))))
    return silhouette_score(X, labels)


def bidcctgan_fri(basepath, feature_names, real_neg, real_pos,  categorical_columns, output_path, sil_tag):
    """
    This is the full algorithm for BIDCCTGAN on the FRI dataset.
    The steps are:
    Stage I. Training the CTGAN on the positive training examples.
    1. Train CTGAN on positive training examples.
    Stage II. Generating synthetic samples from the trained CTGAN.
    2. S1+ = Str+
    for i = 1 to T:
        S1+ = S1+ U CTGAN(S1+)

    :param basepath:
    :param outcome:
    :param balance_test_set:
    :return:
    """
    neg_so_far, pos_so_far = generate_until_condition(real_neg, real_pos, categorical_columns, feature_names, output_path, sil_tag,'balance', 'balance',1, 1,50)
    neg_so_far, pos_so_far = generate_until_condition(neg_so_far, pos_so_far, categorical_columns, feature_names, output_path,sil_tag, 'augment', 'augment',0, 5,  250)
    neg_so_far, pos_so_far = generate_until_condition(neg_so_far, pos_so_far, categorical_columns, feature_names, output_path, sil_tag,'balance', 'rebalance', 1, 1, 250)
    return neg_so_far, pos_so_far


def generate_until_condition(neg_init, pos_init, categorical_columns, feature_names, output_path, sil_tag, condition, purpose, generated_label, augment_factor = 1,  num_samples=1000):
    attempt = 0
    maxAttempt = 1000
    while attempt < maxAttempt:
        attempt += 1
        print(f'Training GAN. Attempt {attempt}')
        pos_so_far = pos_init
        neg_so_far = neg_init
        if generated_label == 1:
            ctgan_model = train_ctgan_fri(pos_so_far, categorical_columns, output_path, purpose, generated_label, sil_tag, epochs=300, batch_size=10,
                                          log_frequency=True)
        elif generated_label == 0:
            ctgan_model = train_ctgan_fri(neg_so_far, categorical_columns, output_path, purpose, generated_label, sil_tag, epochs=300, batch_size=10,
                                          log_frequency=True)
        T = 100

        base_silscore = calculate_silhouette_score(pd.DataFrame(columns=feature_names), neg_so_far, pos_so_far, generated_label)
        print(f'Base silhouette score: {base_silscore}')
        for i in range(1, T):
            generated = generate_ctgan_samples(ctgan_model, num_samples)
            ### here now we will make the negative values to zero.
            #generated[generated < 0] = 0 #remove negative values
            # optional (remove fractional values - not needed! can be done during training)
            #generated = pd.DataFrame(np.round(generated.values).astype(int), columns=feature_names)

            silscore = calculate_silhouette_score(generated, neg_so_far, pos_so_far, generated_label)
            print(f'{i}/{T}, Silhouette score: {silscore}')

            '''
            Applying the sillhouette score thresholding to decide whether to add the generated samples to the current set or not.
            One of the following condition must be met for the generated samples to be added to the current set.
            1. The sil_tag is 'nosil', that means silhouette score checking not required for this experiment
            2. The sil_tag is 'negsil' and the generated label is not 0, that means sillouete is applied only on negative samples but here we are generating positive samples
            3. The sil_tag is 'possil' and the generated label is not 1, that means sillouete is applied only on positive samples but here we are generating negative samples
            4. The generated silhouette score is greater than the base silhouette score or greater than 0.3
            
            '''
            if sil_tag == 'nosil' or (sil_tag == 'negsil' and generated_label != 0) or (sil_tag =='possil' and generated_label!=1) or silscore > base_silscore or silscore > 0.3:
                if generated_label == 1:
                    pos_so_far = pd.concat([pos_so_far, generated])
                    print(f'Added {len(generated)} samples to the positive set')
                elif generated_label == 0:
                    neg_so_far = pd.concat([neg_so_far, generated])
                    print(f'Added {len(generated)} samples to the negative set')
                base_silscore = silscore

            print(f'Current negative set size: {len(neg_so_far)}, positive set size: {len(pos_so_far)}')

            '''
            Checking the stoppping conditions..
            '''
            if condition == 'balance':
                if generated_label == 1 and len(pos_so_far) >= len(neg_so_far):
                    return neg_so_far, pos_so_far
                if generated_label == 0 and len(neg_so_far) >= len(pos_so_far):
                    return neg_so_far, pos_so_far

            elif condition == 'augment':
                if generated_label == 0 and len(neg_so_far) >= augment_factor * len(neg_init):
                    return neg_so_far, pos_so_far
                if generated_label == 1 and len(pos_so_far) >= augment_factor * len(pos_init):
                    return neg_so_far, pos_so_far





