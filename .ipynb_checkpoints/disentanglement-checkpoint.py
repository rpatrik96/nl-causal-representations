# Source: https://github.com/brendel-group/cl-ica/blob/master/disentanglement_utils.py#L63

# !/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# Documentation is intended to be processed by Epydoc.

"""
Introduction
============
The Munkres module provides an implementation of the Munkres algorithm
(also called the Hungarian algorithm or the Kuhn-Munkres algorithm),
useful for solving the Assignment Problem.
Assignment Problem
==================
Let *C* be an *n*\ x\ *n* matrix representing the costs of each of *n* workers
to perform any of *n* jobs. The assignment problem is to assign jobs to
workers in a way that minimizes the total cost. Since each worker can perform
only one job and each job can be assigned to only one worker the assignments
represent an independent set of the matrix *C*.
One way to generate the optimal set is to create all permutations of
the indexes necessary to traverse the matrix so that no row and column
are used more than once. For instance, given this matrix (expressed in
Python)::
    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
You could use this code to generate the traversal indexes::
    def permute(a, results):
        if len(a) == 1:
            results.insert(len(results), a)
        else:
            for i in range(0, len(a)):
                element = a[i]
                a_copy = [a[j] for j in range(0, len(a)) if j != i]
                subresults = []
                permute(a_copy, subresults)
                for subresult in subresults:
                    result = [element] + subresult
                    results.insert(len(results), result)
    results = []
    permute(range(len(matrix)), results) # [0, 1, 2] for a 3x3 matrix
After the call to permute(), the results matrix would look like this::
    [[0, 1, 2],
     [0, 2, 1],
     [1, 0, 2],
     [1, 2, 0],
     [2, 0, 1],
     [2, 1, 0]]
You could then use that index matrix to loop over the original cost matrix
and calculate the smallest cost of the combinations::
    n = len(matrix)
    minval = sys.maxsize
    for row in range(n):
        cost = 0
        for col in range(n):
            cost += matrix[row][col]
        minval = min(cost, minval)
    print minval
While this approach works fine for small matrices, it does not scale. It
executes in O(*n*!) time: Calculating the permutations for an *n*\ x\ *n*
matrix requires *n*! operations. For a 12x12 matrix, that's 479,001,600
traversals. Even if you could manage to perform each traversal in just one
millisecond, it would still take more than 133 hours to perform the entire
traversal. A 20x20 matrix would take 2,432,902,008,176,640,000 operations. At
an optimistic millisecond per operation, that's more than 77 million years.
The Munkres algorithm runs in O(*n*\ ^3) time, rather than O(*n*!). This
package provides an implementation of that algorithm.
This version is based on
http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html.
This version was written for Python by Brian Clapper from the (Ada) algorithm
at the above web site. (The ``Algorithm::Munkres`` Perl version, in CPAN, was
clearly adapted from the same web site.)
Usage
=====
Construct a Munkres object::
    from munkres import Munkres
    m = Munkres()
Then use it to compute the lowest cost assignment from a cost matrix. Here's
a sample program::
    from munkres import Munkres, print_matrix
    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    m = Munkres()
    indexes = m.compute(matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total cost: %d' % total
Running that program produces::
    Lowest cost through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 0) -> 5
    (1, 1) -> 3
    (2, 2) -> 4
    total cost=12
The instantiated Munkres object can be used multiple times on different
matrices.
Non-square Cost Matrices
========================
The Munkres algorithm assumes that the cost matrix is square. However, it's
possible to use a rectangular matrix if you first pad it with 0 values to make
it square. This module automatically pads rectangular cost matrices to make
them square.
Notes:
- The module operates on a *copy* of the caller's matrix, so any padding will
  not be seen by the caller.
- The cost matrix must be rectangular or square. An irregular matrix will
  *not* work.
Calculating Profit, Rather than Cost
====================================
The cost matrix is just that: A cost matrix. The Munkres algorithm finds
the combination of elements (one from each row and column) that results in
the smallest cost. It's also possible to use the algorithm to maximize
profit. To do that, however, you have to convert your profit matrix to a
cost matrix. The simplest way to do that is to subtract all elements from a
large value. For example::
    from munkres import Munkres, print_matrix
    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - col]
        cost_matrix += [cost_row]
    m = Munkres()
    indexes = m.compute(cost_matrix)
    print_matrix(matrix, msg='Highest profit through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total profit=%d' % total
Running that program produces::
    Highest profit through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 1) -> 9
    (1, 0) -> 10
    (2, 2) -> 4
    total profit=23
The ``munkres`` module provides a convenience method for creating a cost
matrix from a profit matrix. Since it doesn't know whether the matrix contains
floating point numbers, decimals, or integers, you have to provide the
conversion function; but the convenience method takes care of the actual
creation of the cost matrix::
    import munkres
    cost_matrix = munkres.make_cost_matrix(matrix,
                                           lambda cost: sys.maxsize - cost)
So, the above profit-calculation program can be recast as::
    from munkres import Munkres, print_matrix, make_cost_matrix
    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    cost_matrix = make_cost_matrix(matrix, lambda cost: sys.maxsize - cost)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total profit=%d' % total
References
==========
1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   *Naval Research Logistics Quarterly*, 2:83-97, 1955.
3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   *Journal of the Society of Industrial and Applied Mathematics*,
   5(1):32-38, March, 1957.
5. http://en.wikipedia.org/wiki/Hungarian_algorithm
Copyright and License
=====================
Copyright 2008-2016 Brian M. Clapper
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__docformat__ = "restructuredtext"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import copy

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = ["Munkres", "make_cost_matrix"]

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

# Info about the module
__version__ = "1.0.8"
__author__ = "Brian Clapper, bmc@clapper.org"
__url__ = "http://software.clapper.org/munkres/"
__copyright__ = "(c) 2008 Brian M. Clapper"
__license__ = "Apache Software License"


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class Munkres:
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def __init__(self):
        """Create a new instance"""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def make_cost_matrix(profit_matrix, inversion_function):
        """
        **DEPRECATED**
        Please use the module function ``make_cost_matrix()``.
        """
        from . import munkres

        return munkres.make_cost_matrix(profit_matrix, inversion_function)

    make_cost_matrix = staticmethod(make_cost_matrix)

    def pad_matrix(self, matrix, pad_value=0):
        """
        Pad a possibly non-square matrix to make it square.
        :Parameters:
            matrix : list of lists
                matrix to pad
            pad_value : int
                value to use to pad the matrix
        :rtype: list of lists
        :return: a new, possibly padded, matrix
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                # Row too short. Pad it.
                new_row += [pad_value] * (total_rows - row_len)
            new_matrix += [new_row]

        while len(new_matrix) < total_rows:
            new_matrix += [[pad_value] * total_rows]

        return new_matrix

    def compute(self, cost_matrix):
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of (row, column) tuples
        that can be used to traverse the matrix.
        :Parameters:
            cost_matrix : list of lists
                The cost matrix. If this cost matrix is not square, it
                will be padded with zeros, via a call to ``pad_matrix()``.
                (This method does *not* modify the caller's matrix. It
                operates on a copy of the matrix.)
                **WARNING**: This code handles square and rectangular
                matrices. It does *not* handle irregular matrices.
        :rtype: list
        :return: A list of ``(row, column)`` tuples that describe the lowest
                 cost path through the matrix
        """
        self.C = self.pad_matrix(cost_matrix)
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0])
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_matrix(self.n * 2, 0)
        self.marked = self.__make_matrix(self.n, 0)

        done = False
        step = 1

        steps = {
            1: self.__step1,
            2: self.__step2,
            3: self.__step3,
            4: self.__step4,
            5: self.__step5,
            6: self.__step6,
        }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.marked[i][j] == 1:
                    results += [(i, j)]

        return results

    def __copy_matrix(self, matrix):
        """Return an exact copy of the supplied matrix"""
        return copy.deepcopy(matrix)

    def __make_matrix(self, n, val):
        """Create an *n*x*n* matrix, populating it with the specific value."""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self):
        """
        For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.
        """
        C = self.C
        n = self.n
        for i in range(n):
            minval = min(self.C[i])
            # Find the minimum value for this row and subtract that minimum
            # from every element in the row.
            for j in range(n):
                self.C[i][j] -= minval

        return 2

    def __step2(self):
        """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (
                        (self.C[i][j] == 0)
                        and (not self.col_covered[j])
                        and (not self.row_covered[i])
                ):
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True

        self.__clear_covers()
        return 3

    def __step3(self):
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == 1:
                    self.col_covered[j] = True
                    count += 1

        if count >= n:
            step = 7  # done
        else:
            step = 4

        return step

    def __step4(self):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        step = 0
        done = False
        row = -1
        col = -1
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero()
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self):
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count - 1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count - 1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self):
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
        minval = self.__find_smallest()
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i][j] += minval
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
        return 4

    def __find_smallest(self):
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxsize
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval

    def __find_a_zero(self):
        """Find the first uncovered element with value 0"""
        row = -1
        col = -1
        i = 0
        n = self.n
        done = False

        while not done:
            j = 0
            while True:
                if (
                        (self.C[i][j] == 0)
                        and (not self.row_covered[i])
                        and (not self.col_covered[j])
                ):
                    row = i
                    col = j
                    done = True
                j += 1
                if j >= n:
                    break
            i += 1
            if i >= n:
                done = True

        return (row, col)

    def __find_star_in_row(self, row):
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col):
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row):
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self, path, count):
        for i in range(count + 1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def __clear_covers(self):
        """Clear all covered matrix cells"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self):
        """Erase all prime markings"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def make_cost_matrix(profit_matrix, inversion_function):
    """
    Create a cost matrix from a profit matrix by calling
    'inversion_function' to invert each value. The inversion
    function must take one numeric argument (of any type) and return
    another numeric argument which is presumed to be the cost inverse
    of the original profit.
    This is a static method. Call it like this:
    .. python::
        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)
    For example:
    .. python::
        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxsize - x)
    :Parameters:
        profit_matrix : list of lists
            The matrix to convert from a profit to a cost matrix
        inversion_function : function
            The function to use to invert each entry in the profit matrix
    :rtype: list of lists
    :return: The converted matrix
    """
    cost_matrix = []
    for row in profit_matrix:
        cost_matrix.append([inversion_function(value) for value in row])
    return cost_matrix


def print_matrix(matrix, msg=None):
    """
    Convenience function: Displays the contents of a matrix of integers.
    :Parameters:
        matrix : list of lists
            Matrix to print
        msg : str
            Optional message to print before displaying the matrix
    """
    import math

    if msg is not None:
        print(msg)

    # Calculate the appropriate format width.
    width = 0
    for row in matrix:
        for val in row:
            width = max(width, int(math.log10(val)) + 1)

    # Make the format string
    format = "%%%dd" % width

    # Print the matrix
    for row in matrix:
        sep = "["
        for val in row:
            sys.stdout.write(sep + format % val)
            sep = ", "
        sys.stdout.write("]\n")


"""Disentanglement evaluation scores such as R2 and MCC."""

from sklearn import metrics
from sklearn import linear_model
import torch
import numpy as np
import scipy as sp
from typing import Union
from typing_extensions import Literal

__Mode = Union[
    Literal["r2"], Literal["adjusted_r2"], Literal["pearson"], Literal["spearman"]
]


def _disentanglement(z, hz, mode: __Mode = "r2", reorder=None):
    """Measure how well hz reconstructs z measured either by the Coefficient of Determination or the
    Pearson/Spearman correlation coefficient."""

    assert mode in ("r2", "adjusted_r2", "pearson", "spearman")

    if mode == "r2":
        return metrics.r2_score(z, hz), None
    elif mode == "adjusted_r2":
        r2 = metrics.r2_score(z, hz)
        # number of data samples
        n = z.shape[0]
        # number of predictors, i.e. features
        p = z.shape[1]
        adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2, None
    elif mode in ("spearman", "pearson"):
        dim = z.shape[-1]

        if mode == "spearman":
            raw_corr, pvalue = sp.stats.spearmanr(z, hz)
        else:
            raw_corr = np.corrcoef(z.T, hz.T)
        corr = raw_corr[:dim, dim:]

        if reorder:
            # effectively computes MCC
            munk = Munkres()
            indexes = munk.compute(-np.absolute(corr))

            sort_idx = np.zeros(dim)
            hz_sort = np.zeros(z.shape)
            for i in range(dim):
                sort_idx[i] = indexes[i][1]
                hz_sort[:, i] = hz[:, indexes[i][1]]

            if mode == "spearman":
                raw_corr, pvalue = sp.stats.spearmanr(z, hz_sort)
            else:
                raw_corr = np.corrcoef(z.T, hz_sort.T)

            corr = raw_corr[:dim, dim:]

        return np.diag(np.abs(corr)).mean(), corr


def linear_disentanglement(z, hz, mode: __Mode = "r2", train_test_split=False):
    """Calculate disentanglement up to linear transformations.

    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        train_test_split: Use first half to train linear model, second half to test.
            Is only relevant if there are less samples then latent dimensions.
    """

    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    # split z, hz to get train and test set for linear model
    if train_test_split:
        n_train = len(z) // 2
        z_1 = z[:n_train]
        hz_1 = hz[:n_train]
        z_2 = z[n_train:]
        hz_2 = hz[n_train:]
    else:
        z_1 = z
        hz_1 = hz
        z_2 = z
        hz_2 = hz

    model = linear_model.LinearRegression()
    model.fit(hz_1, z_1)

    hz_2 = model.predict(hz_2)

    inner_result = _disentanglement(z_2, hz_2, mode=mode, reorder=False)

    return inner_result, (z_2, hz_2)


def permutation_disentanglement(
    z,
    hz,
    mode="r2",
    rescaling=True,
    solver: Union[Literal["naive", "munkres"]] = "naive",
    sign_flips=True,
    cache_permutations=None,
):
    """Measure disentanglement up to permutations by either using the Munkres solver
    or naively trying out every possible permutation.
    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        rescaling: Rescale every individual latent to maximize the agreement
            with the ground-truth.
        solver: How to find best possible permutation. Either use Munkres algorithm
            or naively test every possible permutation.
        sign_flips: Only relevant for `naive` solver. Also include sign-flips in
            set of possible permutations to test.
        cache_permutations: Only relevant for `naive` solver. Cache permutation matrices
            to allow faster access if called multiple times.
    """

    assert solver in ("naive", "munkres")
    if mode == "r2" or mode == "adjusted_r2":
        assert solver == "naive", "R2 coefficient is only supported with naive solver"

    if cache_permutations and not hasattr(
        permutation_disentanglement, "permutation_matrices"
    ):
        permutation_disentanglement.permutation_matrices = dict()

    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    def test_transformation(T, reorder):
        # measure the r2 score for one transformation

        Thz = hz @ T
        if rescaling:
            assert z.shape == hz.shape
            # find beta_j that solve Y_ij = X_ij beta_j
            Y = z
            X = hz

            beta = np.diag((Y * X).sum(0) / (X ** 2).sum(0))

            Thz = X @ beta

        return _disentanglement(z, Thz, mode=mode, reorder=reorder), Thz

    def gen_permutations(n):
        # generate all possible permutations w/ or w/o sign flips

        def gen_permutation_single_row(basis, row, sign_flips=False):
            # generate all possible permutations w/ or w/o sign flips for one row
            # assuming the previous rows are already fixed
            basis = basis.clone()
            basis[row] = 0
            for i in range(basis.shape[-1]):
                # skip possible columns if there is already an entry in one of
                # the previous rows
                if torch.sum(torch.abs(basis[:row, i])) > 0:
                    continue
                signs = [1]
                if sign_flips:
                    signs += [-1]

                for sign in signs:
                    T = basis.clone()
                    T[row, i] = sign

                    yield T

        def gen_permutations_all_rows(basis, current_row=0, sign_flips=False):
            # get all possible permutations for all rows

            for T in gen_permutation_single_row(basis, current_row, sign_flips):
                if current_row == len(basis) - 1:
                    yield T.numpy()
                else:
                    # generate all possible permutations of all other rows
                    yield from gen_permutations_all_rows(T, current_row + 1, sign_flips)

        basis = torch.zeros((n, n))

        yield from gen_permutations_all_rows(basis, sign_flips=sign_flips)

    n = z.shape[-1]
    # use cache to speed up repeated calls to the function
    if cache_permutations and not solver == "munkres":
        key = (rescaling, n)
        if not key in permutation_disentanglement.permutation_matrices:
            permutation_disentanglement.permutation_matrices[key] = list(
                gen_permutations(n)
            )
        permutations = permutation_disentanglement.permutation_matrices[key]
    else:
        if solver == "naive":
            permutations = list(gen_permutations(n))
        elif solver == "munkres":
            permutations = [np.eye(n, dtype=z.dtype)]

    scores = []

    # go through all possible permutations and check r2 score
    for T in permutations:
        scores.append(test_transformation(T, solver == "munkres"))

    return max(scores, key=lambda x: x[0][0])