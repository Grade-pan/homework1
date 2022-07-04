import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

def onehot(fasta):
    if re.search('>', fasta):
        name = re.split('\n', fasta)[0]
        sequence = re.split('\n', fasta)[1]

    seq_array = np.array(list(sequence))
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(seq_array)

    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoder_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    onehot_encoder_seq = np.asarray(onehot_encoder_seq)
    print(np.asarray(onehot_encoder_seq))
    return onehot_encoder_seq

def output_dataset(sequence_filename):
    #onehot
    outputfile = open('onehot_seq.npy', 'ab')
    with open(sequence_filename, 'r') as file:
        data = file.readlines()
        x = []
        for index, line in enumerate(data):
            #print(index)
            if index % 2 == 0:
                fasta = data[index] + data[index + 1]
                onehot_seq = pd.DataFrame(onehot(fasta))
                x.append(onehot_seq)
                print(len(x))
                #np.save(outputfile, onehot_seq,)
    file.close()
    return x

def read_dataset(seq_filename, y_filename, x):
    #x = np.load(seq_filename)
    x = np.array(x)
    y = pd.read_csv('y.csv')
    y = np.asarray(y['TPM'])
    #print(y['TPM'][0])

    print(len(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=23, test_size=0.2)
    lasso = Lasso(alpha=10)
    lasso.fit(x_train, y_train)
    print(r2_score(y_train, lasso.predict(x_train)))
    print(r2_score(y_test, lasso.predict(x_test)))

if __name__ == '__main__':
    x = output_dataset('x-2k_sequence.fa')
    read_dataset('onehot_seq.npy', 'y.csv', x)
