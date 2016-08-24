"""
Class for the syllable classification
"""

import os.path
import numpy as np
import reservoirConceptor as c
import functions as fct
import preprocessing as prep
import random


class syllableClassifier:
    def __init__(self, fname):
        """ Class that performs supervised learning on syllable data in order to perform classification.

        :param fname: Complete path to folder which includes folders for each syllable which include folders for each sample which include wave data
        """

        self.folder = fname

    def prep_data(self, n_syllables, n_train, n_test, syll_names=None, samples=None,
                  sample_rate=20000, ds_type='mean', mel_channels=12, inv_coefforder=False, winsize=20,
                  frames=64, smooth_length=5, poly_order=3, inc_der=[True, True]):

        """ Function that performs the following preprocessing steps on data in file:
        1. loading
        2. downsampling
        3. Extraction of Mel Frequency Cepstral Coefficients
        4. Extraction of shift and scale of training data
        5. Data normalization with shift and scale of training data
        6. Data smoothing
        7. Add derivatives

        :param file: complete path name for file to be loaded (string)
        :param n_syllables: number of syllables to include in preprocessing (scalar)
        :param n_train: number of training samples (scalar)
        :param n_test: number of test samples for each syllable (vector of length n_syllables)
        :param sample_rate: Desired sampling rate for the downsampling
        :param ds_type: Type of interpolation used for downsampling. Can be mean or IIR, which uses an order 8 Chebyshev type 1 filter (default = mean)
        :param syll_names the syllable names that should be used if specified
        :param samples: Order of how samples should be included in training/testing data (default = None)
        :param mel_channels: number of channels to include from the Mel freq spectrum (default = 12)
        :param winsize: size of the time window used for mfcc extraction (default = 20 ms)
        :param frames: desired number of time frames in final mfcc data (default = 64)
        :param inv_coefforder: if True, extract last n mfcc instead of first n (default = False)
        :param smooth_length: Number of sampling points to reduce mel transformed data to (default = 5)
        :param poly_order: Order of the polynomial to be used for smoothing (default = 3)
        :param inc_der: List of 2 booleans indicating whether to include first and second derivative of mfcc data (default = [True,True])

        :returns trainDataSmoothend: array of preprocessed training data
        :returns testDataSmoothend: array of preprocessed test data
        """

        self.n_train = n_train
        self.n_test = n_test
        self.n_syllables = n_syllables
        self.samples = samples

        """ Load Data """

        syllables = [files for files in os.listdir(self.folder)]
        syllables.remove('.gitignore')

        self.trainDataRaw = []
        self.testDataRaw = []
        self.skipped_syllables = []

        if syll_names is not None:
            # if syllable names are provided use those
            for i, syll in enumerate(syll_names):
                if np.sum(np.array(syllables) == syll) == 0:
                    print('Warning: Syllable ', syll, ' not found in folder.')
                    self.n_syllables -= 1
                    continue
                if not self.samples:
                    self.trainDataRaw.append(prep.load_data(self.folder + '/' + syll, self.n_train, 0))
                    self.testDataRaw.append(prep.load_data(self.folder + '/' + syll, self.n_test[i], self.n_train))
                else:
                    self.trainDataRaw.append(prep.load_data(self.folder + '/' + syll, self.n_train, 0,
                                                            sample_order=self.samples[i][0:n_train]))
                    self.testDataRaw.append(prep.load_data(self.folder + '/' + syll, self.n_test[i], self.n_train,
                                                           sample_order=self.samples[i][n_train::]))
        else:
            # sample random from the list of available syllables
            ind = random.sample(range(1, len(syllables)), n_syllables)

            for i in range(n_syllables):
                success = False
                while not success:
                    try:
                        # try to find a syllable that fullfills the condition of the n_train length
                        if not self.samples:
                            self.trainDataRaw.append(
                                prep.load_data(self.folder + '/' + syllables[ind[i]], self.n_train, 0))
                            self.testDataRaw.append(
                                prep.load_data(self.folder + '/' + syllables[ind[i]], self.n_test[i], self.n_train))
                        else:
                            self.trainDataRaw.append(
                                prep.load_data(self.folder + '/' + syllables[ind[i]], self.n_train, 0,
                                               sample_order=self.samples[i][0:n_train]))
                            self.testDataRaw.append(
                                prep.load_data(self.folder + '/' + syllables[ind[i]], self.n_test[i], self.n_train,
                                               sample_order=self.samples[i][n_train::]))
                        success = True
                    except:
                        self.skipped_syllables.append(syllables[ind[i]])
                        if i >= (len(ind) - 1): break
                        if ind[i] < ind[i + 1] and ind[i] < len(syllables):
                            ind[i] += 1
                        else:
                            break
                        pass

        """ Downsampling """

        self.trainDataDS = prep.downSample(self.trainDataRaw, sample_rate, ds_type)
        self.testDataDS = prep.downSample(self.testDataRaw, sample_rate, ds_type)

        """ MFCC extraction """

        self.trainDataMel = prep.getMEL(self.trainDataDS, mel_channels, inv_coefforder)
        self.testDataMel = prep.getMEL(self.testDataDS, mel_channels, inv_coefforder)

        """ shift and scale both datasets according to properties of training data """

        shifts, scales = prep.getShiftsAndScales(self.trainDataMel)

        trainDataNormalized = prep.normalizeData(self.trainDataMel, shifts, scales)
        testDataNormalized = prep.normalizeData(self.testDataMel, shifts, scales)

        """ Interpolate datapoints so that each sample has only (smoothLength) timesteps """

        trainDataSmoothend = prep.smoothenData(trainDataNormalized, smooth_length, poly_order, mel_channels)
        testDataSmoothend = prep.smoothenData(testDataNormalized, smooth_length, poly_order, mel_channels)

        """ Include first and second derivatives of mfcc """

        trainDataDer = prep.mfccDerivates(trainDataSmoothend, Der1=inc_der[0], Der2=inc_der[1])
        testDataDer = prep.mfccDerivates(testDataSmoothend, Der1=inc_der[0], Der2=inc_der[1])


        self.trainDataFinal = trainDataDer
        self.testDataFinal = testDataDer

    def cLearning(self, gamma_pos=25, gamma_neg=27, N=10, SR=1.2, bias_scale=1.0, inp_scale=0.2, conn=1):
        """ Function that learns positive and negative conceptors on data with the following steps:
        1. create Reservoir
        2. Feed each sample of each syllable in reservoir and collect its states
        3. Use states to compute positive conceptor
        4. Use Conceptor logic to compute negative conceptor

        :param gamma_pos: aperture of the positive conceptors
        :param gamma_neg: aperture of the negative conceptors
        :param data: list of syllables with sample data
        :param N: size of the reservoir
        :param SR: spectral radius of the reservoir
        :param bias_scale: scaling of the bias while running reservoir
        :param inp_scale: scaling of the input when fed into the reservoir
        :param conn: scaling of the amount of connectivity within the reservoir

        :returns C_pos: List of positive conceptors
        :returns C_neg: List of negative conceptors
        """

        self.res = c.Reservoir(N=N, NetSR=SR, bias_scale=bias_scale, inp_scale=inp_scale, conn=conn)
        self.C_pos = []

        for syllable in np.array(self.trainDataFinal):

            R_syll = np.zeros((syllable.shape[1] * (N + syllable.shape[2]), syllable.shape[0]))

            for i, sample in enumerate(syllable):
                self.res.run([sample], t_learn=len(sample), t_wash=0, load=False)
                states = np.concatenate((np.squeeze(self.res.TrainArgs.T), sample), axis=1)
                R_syll[:, i] = np.reshape(states, states.shape[0] * states.shape[1])

            R = np.dot(R_syll, R_syll.T) / self.n_train
            C_tmp = np.dot(R, np.linalg.inv(R + np.eye(len(R))))
            self.C_pos.append(C_tmp)

        self.C_neg = []

        for i in range(len(self.C_pos)):
            C = np.zeros_like(self.C_pos[0])
            for j in list(range(0, i)) + list(range(i + 1, len(self.C_pos))):
                C = fct.OR(C, self.C_pos[j])
            self.C_neg.append(C)

        for i in range(len(self.C_pos)):
            self.C_pos[i] = fct.phi(self.C_pos[i], gamma_pos)
            self.C_neg[i] = fct.phi(self.C_neg[i], gamma_neg)

    def cTest(self, pattern=None):
        """ Function that uses trained conceptors to recognize syllables in data by going through the following steps:
        1. Feed each sample of each syllable into reservoir and collect its states
        2. Analyize similarity of collected states and trained conceptors
        3. Choose syllable, for which similarity is highest

        :param pattern: the list of syllables that make up the pattern of the song

        :returns evidences: list of arrays of evidences with rows = trials and columns = syllables
                            for positive, negative and combined conceptors
        :returns class_perf: Mean classification performance on test data set for
                             positive, negative and combined conceptors
        """

        h_pos = []
        h_neg = []
        h_comb = []
        class_pos = []
        class_neg = []
        class_comb = []

        if pattern is not None:
            self.testData = np.array([self.testDataFinal[np.argmax(syll)] for syll in pattern])
        else:
            self.testData = self.testDataFinal

        for syll_i, syllable in enumerate(self.testData):

            for sample in syllable:

                self.res.run([sample], t_learn=sample.shape[0], t_wash=0, load=False)
                states = np.concatenate((np.squeeze(self.res.TrainArgs).T, sample), axis=1)
                z = np.reshape(states, states.shape[0] * states.shape[1])

                h_pos_tmp = np.zeros(len(self.C_pos))
                h_neg_tmp = np.zeros(len(self.C_pos))
                h_comb_tmp = np.zeros(len(self.C_pos))

                for k in range(len(self.C_pos)):
                    h_pos_tmp[k] = np.dot(np.dot(z.T, self.C_pos[k]), z)
                    h_neg_tmp[k] = np.dot(np.dot(z.T, self.C_neg[k]), z)

                h_pos_tmp = h_pos_tmp - np.min(h_pos_tmp)
                h_pos_tmp = h_pos_tmp / np.max(h_pos_tmp)
                h_neg_tmp = h_neg_tmp - np.min(h_neg_tmp)
                h_neg_tmp = 1 - h_neg_tmp / np.max(h_neg_tmp)
                h_comb_tmp = (h_pos_tmp + h_neg_tmp) / 2.0
                h_pos.append(h_pos_tmp/np.sum(h_pos_tmp))
                h_neg.append(h_neg_tmp/np.sum(h_neg_tmp))
                h_comb.append(h_comb_tmp/np.sum(h_comb_tmp))

                dec_pos = np.where(h_pos_tmp == np.max(h_pos_tmp))[0][0]
                dec_neg = np.where(h_neg_tmp == np.max(h_neg_tmp))[0][0]
                dec_comb = np.where(h_comb_tmp == np.max(h_comb_tmp))[0][0]

                classification_pos_tmp = 1 if dec_pos == syll_i else 0
                classification_neg_tmp = 1 if dec_neg == syll_i else 0
                classification_comb_tmp = 1 if dec_comb == syll_i else 0

                class_pos.append(classification_pos_tmp)
                class_neg.append(classification_neg_tmp)
                class_comb.append(classification_comb_tmp)

        h_pos = np.array(h_pos)
        h_neg = np.array(h_neg)
        h_comb = np.array(h_comb)
        class_pos = np.array(class_pos)
        class_neg = np.array(class_neg)
        class_comb = np.array(class_comb)

        self.evidences = [h_pos, h_neg, h_comb]
        self.class_perf = [np.mean(class_pos), np.mean(class_neg), np.mean(class_comb)]
