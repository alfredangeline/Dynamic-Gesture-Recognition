import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, path_videos, path_labels, path_train=None, path_validation=None, path_test=None):
        self.path_videos = path_videos
        self.path_labels = path_labels
        self.path_train  = path_train
        self.path_validation = path_validation
        self.path_test = path_test

        self.get_labels(path_labels)

        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)
        if self.path_validation:
            self.validation_df = self.load_video_labels(self.path_validation)
        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, 'input')

    def get_labels(self, path_labels):
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        #extract labels from dataframe
        self.labels = [str(label[0]) for label in self.labels_df.values]
        self.n_labels = len(self.labels)
        #create a dictionary to convert label to int and vice-versa
        self.label_to_int = dict(zip(self.labels, range(self.n_labels)))
        self.int_to_labels= dict(enumerate(self.labels))

    def load_video_labels(self, path_subset, mode='label'):
        if mode == 'input':
            names = ['video_id']
        elif mode == 'label':
            names = ['video_id', 'label']

        df = pd.read_csv(path_subset, sep=';', names=names)
        if mode == 'label':
            df = df[df.label.isin(self.labels)]
        return df

    def categorical_to_label(self):
        """ Used to convert a vector to the associated string label
        # Arguments
            vector : Vector representing the label of a video
        #Returns
            Returns a String that is the label of a video
        """
        return self.int_to_label[np.where(vector==1)[0][0]]
