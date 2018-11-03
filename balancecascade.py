from collections import Counter

import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state, safe_indexing
from sklearn.model_selection import cross_val_predict
from sklearn.utils.deprecation import deprecated

from .base import BaseEnsembleSampler
from ..under_sampling.base import BaseUnderSampler
from ..utils import check_sampling_strategy
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
@deprecated('BalanceCascade is deprecated in 0.4 and will be removed in 0.6.')
class BalanceCascade(BaseEnsembleSampler):

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 n_max_subset=None,
                 estimator=None,
                 ratio=None):
        super(BalanceCascade, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.estimator = estimator
        self.n_max_subset = n_max_subset

    def _validate_estimator(self):
        """Private function to create the classifier"""

        if (self.estimator is not None and
                isinstance(self.estimator, ClassifierMixin) and
                hasattr(self.estimator, 'predict')):
            self.estimator_ = clone(self.estimator)
        elif self.estimator is None:
            self.estimator_ = KNeighborsClassifier()
        else:
            raise ValueError('Invalid parameter `estimator`. Got {}.'.format(
                type(self.estimator)))

    def _fit_resample(self, X, y):
        self._validate_estimator()

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, 'under-sampling')

        random_state = check_random_state(self.random_state)

        # array to know which samples are available to be taken
        samples_mask = np.ones(y.shape, dtype=bool)

        # where the different set will be stored
        idx_under = []

        n_subsets = 0
        b_subset_search = True
        while b_subset_search:
            target_stats = Counter(
                safe_indexing(y, np.flatnonzero(samples_mask)))
            # store the index of the data to under-sample
            index_under_sample = np.empty((0, ), dtype=np.int)
            # value which will be picked at each round
            index_constant = np.empty((0, ), dtype=np.int)
            for target_class in target_stats.keys():
                if target_class in self.sampling_strategy_.keys():
                    n_samples = self.sampling_strategy_[target_class]
                    # extract the data of interest for this round from the
                    # current class
                    index_class = np.flatnonzero(y == target_class)
                    index_class_interest = index_class[samples_mask[
                        y == target_class]]
                    y_class = safe_indexing(y, index_class_interest)
                    # select randomly the desired features
                    index_target_class = random_state.choice(
                        range(y_class.size), size=n_samples, replace=False)
                    index_under_sample = np.concatenate(
                        (index_under_sample,
                         index_class_interest[index_target_class]),
                        axis=0)
                else:
                    index_constant = np.concatenate(
                        (index_constant, np.flatnonzero(y == target_class)),
                        axis=0)

            # store the set created
            n_subsets += 1
            subset_indices = np.concatenate(
                (index_under_sample, index_constant), axis=0)
            idx_under.append(subset_indices)

            # fit and predict using cross validation
            X_subset = safe_indexing(X, subset_indices)
            y_subset = safe_indexing(y, subset_indices)
            pred = cross_val_predict(self.estimator_, X_subset, y_subset, cv=3)
            # extract the prediction about the targeted classes only
            pred_target = pred[:index_under_sample.size]
            index_classified = index_under_sample[pred_target == safe_indexing(
                y_subset, range(index_under_sample.size))]
            samples_mask[index_classified] = False

            # check the stopping criterion
            if self.n_max_subset is not None:
                if n_subsets == self.n_max_subset:
                    b_subset_search = False
            # check that there is enough samples for another round
            target_stats = Counter(
                safe_indexing(y, np.flatnonzero(samples_mask)))
            for target_class in self.sampling_strategy_.keys():
                if (target_stats[target_class] <
                        self.sampling_strategy_[target_class]):
                    b_subset_search = False

        X_resampled, y_resampled = [], []
        for indices in idx_under:
            X_resampled.append(safe_indexing(X, indices))
            y_resampled.append(safe_indexing(y, indices))

        if self.return_indices:
            return (np.array(X_resampled), np.array(y_resampled),
                    np.array(idx_under))
        else:
            return np.array(X_resampled), np.array(y_resampled)
