from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

class ModelEvaluation():

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        # self._basic_eval()


    def confusion_matrix(self):

        cm = confusion_matrix(self.y_true, self.y_pred)

        return cm
    
    def get_accuracy(self):

        return accuracy_score(self.y_true, self.y_pred)

    def classification_report(self, target_names=None):
        if target_names is not None:
            cr = classification_report(self.y_true, self.y_pred, target_names=target_names, digits = 3)
        else:
            cr = classification_report(self.y_true, self.y_pred, digits = 3)

        return cr

    def get_precision_recall_fscore_support(self):
        precision, recall, fscore, support = precision_recall_fscore_support(self.y_true, self.y_pred)



        return np.round(precision, decimals=4),  np.round(recall, decimals=4), \
                np.round(fscore, decimals=4),  np.round(support, decimals=4)

    def average_f1score(self):
        ''' average up f1score over labels '''

        _, _, fscore, _ = self.get_precision_recall_fscore_support()

        return np.round(np.mean(fscore), decimals=4)


    def pycm_result(self):
        cm = ConfusionMatrix(actual_vector=self.y_true, predict_vector=self.y_pred)
        return cm

    def get_incorrectly_predicted(self):
        '''
        return index of incorrectly predicted
        '''
        return np.where((self.y_true == self.y_pred) == False)[0]

    def get_correctly_predicted(self):
        '''
        return index of correctly predicted
        '''
        return np.where((self.y_true == self.y_pred) == True)[0]

    def _basic_eval(self):

        cm = self.confusion_matrix()

        self.false_positive = cm.sum(axis=0) - np.diag(cm)
        self.false_negative = cm.sum(axis=1) - np.diag(cm)
        self.true_positive = np.diag(cm)
        self.true_negative = cm.sum() - (self.false_positive + self.false_negative + self.true_positive)

        # Sensitivity, hit rate, recall, or true positive rate
        self.sensitivity = self.true_positive/(self.true_positive + self.false_negative)
        # Specificity or true negative rate
        self.specificity = self.true_negative/(self.true_negative+self.false_positive)
        # Precision or positive predictive value
        self.precision = self.true_positive/(self.true_positive+self.false_positive)
        # Fall out or false positive rate
        self.false_positive_rate = self.false_positive/(self.false_positive+self.true_negative)
        # False negative rate
        self.false_negative_rate = self.false_negative/(self.true_positive+self.false_negative)

        # Overall accuracy
        self.accuracy = (self.true_positive+self.true_negative)/(self.true_positive+self.false_positive+self.false_negative+self.true_negative)
