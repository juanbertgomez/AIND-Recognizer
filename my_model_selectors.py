import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def min_scores(self, scores):
        return min(scores, key = lambda x : x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        BIC_scores = []

        for num_states in range(self.min_n_components, self.max_n_components+1):
        
            try:
               
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                d = model.n_features
                # p = = n^2 + 2*d*n - 1
                p = num_states ** 2 + 2 * d * num_states - 1
                BIC = -2.0 * logL + p * logN

                BIC_scores.append([BIC, model])

            except:
                pass

        return self.min_scores(BIC_scores)[1] if BIC_scores else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    #DIC Equation:
    #    DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))

    def m_scores(self, scores):
        return max(scores, key = lambda x : x[0])

    def dicScore(self, n):
        model = self.base_model(n)
        scores = []
        for word, (X, lengths) in self.hwords.items():
            if word != self.this_word:
                scores.append(model.score(X, lengths))
        return ([model.score(self.X, self.lengths) - np.mean(scores), model])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        dic_scores = []

        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                dic_scores.append(self.dicScore(num_states))

            except:
                pass
    
        #return dic_scores
        return self.m_scores(dic_scores)[1] if dic_scores else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def max_scores(self, scores):
        return max(scores, key = lambda x : x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        kf = KFold()
        log_likelihoods = []
        scores = []

        for num_states in range(self.min_n_components, self.max_n_components+1):
        
            try:
                # check is there is enough data to split
                if len(self.sequences)> 2:
                    for train_index, test_index in kf.split(self.sequences):

                        self.X, self.lengths = combine_sequences(train_index, self.sequences)
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)

                        model = self.base_model(num_states)
                        log_likelihood = model.score(X_test, lengths_test)
                        log_likelihoods.append(log_likelihood)

                else:
                    model = self.base_model(num_states)
                    log_likelihoods = model.score(self.X, self.lengths)

                scores.append([np.mean(log_likelihoods), model])

            except:
                pass

        return self.max_scores(scores)[1] if scores else None
