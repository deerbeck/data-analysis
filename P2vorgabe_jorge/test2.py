# -*- coding: utf-8 -*-
"""
@author: Nils Rosehr
"""

import matplotlib
matplotlib.use("Agg")
# import matplotlib.pyplot as plt

import numpy as np
import unittest
import pickle
import pprint
#import os
import sys
import re
import hashlib
import base64
import collections
import contextlib
from itertools import product
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import chi2



P_number = 2
exec('import main{}'.format(P_number))
exec('main = main{}'.format(P_number))
# exec('import lib{}'.format(P_number))
# exec('lib = lib{}'.format(P_number))
# exec('import vorgabe{}'.format(P_number))
# exec('vorgabe = vorgabe{}'.format(P_number))
main_file = 'main{}.py'.format(P_number)
test_file = 'test{}.py'.format(P_number)
vorgabe_file = 'vorgabe{}.py'.format(P_number)
data_file = 'test{}.data'.format(P_number)
lib_file = 'lib{}.py'.format(P_number)
info_file = 'info{}.md'.format(P_number)
files_tested_for_import_restrictions = [main_file]
files_tested_for_correctness = [test_file]
# files_tested_for_correctness = [test_file, vorgabe_file]
_subtest_msg_sentinel = object()


### test control:
###
### Define in main a dictionary test_control
### with test-function names as keys
### and lists of integers of sub-test indeces
### which are supposed to be run.
### This only works for tests which
### use "runner".
###
### Example:
###
### test_control = {
###     'test_Verteilungen': [1, 2],        # only subtests number 1 and 2 are performed
###     'test_ZGWS':         range(10, 20), # subtests 10-19 are run
###     'test_Bayes_Formel': []             # no subtests are run, i.e. 'test_Bayes_Formel' is switched off
###     }


try:
    if(type(main.test_control) == dict):
        test_control = main.test_control
    else:
        test_control = {}
except (NameError, AttributeError):
    test_control = {}
skipped_tests = {}


class ERROR(Exception):
    pass


class Test_Numerik(unittest.TestCase):
    
    test_counter = {}
    
    @staticmethod
    def Norm(x):
        return np.sqrt(np.sum(np.abs(x)**2))

    def assertLessEqualMultiple(self, result, true_value, multiple=2, **kwargs):
        self.assertLessEqual(result, true_value * multiple, **kwargs)
    
    def assertEqualComponent(self, result, true_value, index=0, **kwargs):
        msg = kwargs.pop('msg')
        self.assertEqual(result[index], true_value[index], msg=msg.format(result), **kwargs)
    
    def assertLessEqualConstant(self, result, true_value, constant=1, **kwargs):
        self.assertLessEqual(result, constant, **kwargs)
    
    def assertBothIterableOfSameLength(self, result, true_value):
        if not hasattr(result, '__iter__'):
            self.fail('"result" should be iterable.')
        if not hasattr(true_value, '__iter__'):
            self.fail('"true value should be iterable."')
        if not hasattr(result, '__len__'):
            self.fail('"result" does not have a length. This should not happen.')
        if not hasattr(true_value, '__len__'):
            self.fail('"True value" does not have a length. This should not happen.')
        self.assertEqual(len(result), len(true_value), msg='Lengths of "result" and "true value" are not equal.')
    
    def assertBothIterableOrBothNotIterable(self, result, true_value):
        """Check if both arguments (named result and true_value) are iterable
        of if both are not iterable. Otherwise raise assertion error.
        Return True if both arguments are iterable and False otherwise."""
        if hasattr(true_value, '__iter__'):
            self.assertBothIterableOfSameLength(result, true_value)
            return True
        else:
            if hasattr(result, '__iter__'):
                self.fail('"result" is iterable", but "true value" is not.')
            return False
        
    def assertAlmostEqualPlaces(self, result, true_value, places=7, **kwargs):
        # check if both arguments are iterable or if both are not iterable (otherwise raise assertion error)
        if type(true_value) is str:
            self.assertEqual(result, true_value)
        else:
            if self.assertBothIterableOrBothNotIterable(result, true_value):
                for result, true_value in zip(result, true_value):
                    self.assertAlmostEqualPlaces(result, true_value, places=places, **kwargs)
            else:
                self.assertAlmostEqual(result, true_value, places=places, **kwargs)        
                
    def assertAlmostEqualRelative(self, result, true_value, relative=10**-7, **kwargs):
        # check if both arguments are iterable or if both are not iterable (otherwise raise assertion error)
        if type(true_value) is str:
            self.assertEqual(result, true_value)
        else:
            if self.assertBothIterableOrBothNotIterable(result, true_value):
                for result, true_value in zip(result, true_value):
                    self.assertAlmostEqualRelative(result, true_value, relative=relative, **kwargs)
            else:
                delta = max(abs(true_value), abs(result)) * relative
                self.assertAlmostEqual(result, true_value, delta=delta, **kwargs)
        
    def assertNormAlmostZero(self, result, delta=10**-7, **kwargs):
        self.assertAlmostEqual(self.Norm(result), 0.0, delta=delta, **kwargs)
        
    def assertAlmostEqualRelativeAbs(self, result, true_value, relative=10**-7, **kwargs):
        # check if both arguments are iterable or if both are not iterable (otherwise raise assertion error)
        if self.assertBothIterableOrBothNotIterable(result, true_value):
            for result, true_value in zip(result, true_value):
                self.assertAlmostEqualRelativeAbs(result, true_value, relative=relative, **kwargs)
        else:
            delta = max(abs(true_value) + abs(result), 1) * relative
            self.assertAlmostEqual(result, true_value, delta=delta, **kwargs)
        
    def assertAlmostEqualRelativePadding(self, result, true_value, relative=10**-7):
        if not hasattr(result, '__iter__'):
            self.fail('"result" should be iterable.')
        if not hasattr(true_value, '__iter__'):
            self.fail('"true value should be iterable."')
        for l in (result, true_value):
            while len(l) > 0 and (np.isnan(l[-1]) or np.isinf(l[-1])):
                l.pop()
        while len(result) != len(true_value):
            if len(result) < len(true_value):
                result.append(result[-1])
            else:
                true_value.append(true_value[-1])
        self.assertAlmostEqualRelative(result, true_value, relative=relative)
#        for result, true_value in zip(result, true_value):
#            delta = (abs(result) + abs(true_value)) * relative
#            self.assertAlmostEqual(result, true_value, delta=delta)
        
    def assertAlmostEqualUnorderedListRelative(self, result, true_value, relative=10**-7):
        self.assertBothIterableOfSameLength(result, true_value)
        for (a, b) in ((result, true_value), (true_value, result)):
            for r in a:
                closest, dist = None, np.inf
                for t in b:
                    d = abs(r - t)
                    if d < dist:
                        closest, dist = t, d
                delta = (abs(r) + abs(closest)) * relative
                self.assertAlmostEqual(r, closest, delta=delta)
        
    def assertAlmostEqualSquareSum(self, result, true_value, delta=10**-7):
        self.assertBothIterableOfSameLength(result, true_value)
        error = sum(np.abs(np.array(result) - np.array(true_value))**2)
        self.assertLessEqual(error, delta)
        
    def assertEqualStringBeginning(self, result, true_value, length=3):
        if type(true_value) == list:
            or_value = False
            for t in true_value:
                or_value = or_value or result[:length] == t[:length]
            if not or_value:
                self.fail('Beginnings of strings are not equal.')
        else:
            self.assertEqual(result[:length], true_value[:length], msg='Beginnings of strings are not equal.')

    def runner(self, method, args, assertMethod,
               post=lambda x: x, true_value=None, marker=None, *vargs, **kwargs):
        test_name = sys._getframe().f_back.f_code.co_name  # name of calling method; this may break in the future
        if test_name in self.test_counter:
            self.test_counter[test_name] += 1
        else:
            self.test_counter[test_name] = 1
        # skip test if test_control so wishes
        if test_name in test_control and not(hasattr(test_control[test_name], '__iter__') and self.test_counter[test_name] in test_control[test_name]):
            if test_name in skipped_tests:
                skipped_tests[test_name].append(self.test_counter[test_name])
            else:
                skipped_tests[test_name] = [self.test_counter[test_name]]
            return
                        
        # test not skipped
        if marker:
            print(marker, 'Vor Aufruf, Argumente:', args)
        result = post(method(*args))
            
#        with warnings.catch_warnings():
#            warnings.filterwarnings('error')
#            try:
#                result = post(method(*args))
#            except:
#                if results.collecting:
#                    print('WARNUNG: Ersatzwert 17..17 genutzt wegen Fehler:', sys.exc_info())
#                result = int(17 * '17')

        if results.collecting:
            if true_value is not None:
                true_val = true_value
            else:
                true_val = result
            results.set(true_val)
        else:
            true_val = results.get()
        if marker:
            print(marker, 'Nach Aufruf, Ergebnisse:', result, true_val)
        assertMethod(result, true_val, *vargs, **kwargs)
        
    def setUp(self):
        results.start_new_test(self._testMethodName)
        
    def tearDown(self):
        results.finish_test()

    def subTest_orig(self, *args, **kwargs):
        return unittest.TestCase.subTest(self, *args, **kwargs)

    # Redefine subTest:
    # Code is copied from unittest.case.py python version 3.6
    # and ammended by code executed before and after block (2 lines added).
    # In three places "unittest.case." had to be inserted.
    @contextlib.contextmanager
    def subTest(self, msg=_subtest_msg_sentinel, name='', **params):
        """Return a context manager that will return the enclosed block
        of code in a subtest identified by the optional message and
        keyword parameters.  A failure in the subtest marks the test
        case as failed but resumes execution at the end of the enclosed
        block, allowing further test code to be executed.
        """
        results.start_new_subtest(name)   # executed before block
        if not self._outcome.result_supports_subtests:
            yield
            return
        parent = self._subtest
        if parent is None:
            params_map = collections.ChainMap(params)
        else:
            params_map = parent.params.new_child(params)
        self._subtest = unittest.case._SubTest(self, msg, params_map)
        try:
            with self._outcome.testPartExecutor(self._subtest, isTest=True):
                yield
            if not self._outcome.success:
                result = self._outcome.result
                if result is not None and result.failfast:
                    raise unittest.case._ShouldStop
            elif self._outcome.expectedFailure:
                # If the test is expecting a failure, we really want to
                # stop now and register the expected failure.
                raise unittest.case._ShouldStop
        finally:
            results.finish_subtest()   # executed after block
            self._subtest = parent
    
    ### set up registration of object names        
    object_names_dict_by_id = dict()
    objects_dict_by_name = dict()
        
    @classmethod
    def set_name(cls, main_object, name, use_main_object_itself=False):
        if use_main_object_itself:
            obj = main_object
        else:
            obj = getattr(main_object, name)
        cls.object_names_dict_by_id[id(obj)] = name
        assert not name in cls.objects_dict_by_name, 'The name is not unique.'
        cls.objects_dict_by_name[name] = obj
        
    @classmethod
    def get_name(cls, obj):
        assert id(obj) in cls.object_names_dict_by_id, 'Object name not stored.'
        return cls.object_names_dict_by_id[id(obj)]
    
    @classmethod
    def get_object(cls, name):
        assert name in cls.objects_dict_by_name, 'Name does not exist.'
        return cls.objects_dict_by_name[name]
    
    @classmethod
    def save(cls, eval_string, name=None):
        if not name:
            name = eval_string
        if results.collecting:
            obj = eval(eval_string)
            results.set_global(name, [obj])  # save singleton of obj as all values are lists
            return obj
        else:
            return results.get_global(name)[0]  # get singleton and retrieve element (all values must be stored as lists)
        
        

class Results:
    
    def __init__(self, data_file, collect_results=False):
        self.filename = data_file
        self.collecting = collect_results
        self.test = tuple()
        if self.collecting:
            self.data = {}   # dict of tests and subtests
            self.index = None   # indeces are not used in collecting mode: we can append
        else:
            self.data = pickle.load(open(self.filename, 'rb'))
            # we keep current indeces for all subtest, so that we can resume a subtest with the same name
            self.index = {k: 0 for k in self.data}   # set indeces for all subtests to 0
            
    def testname(self):
        return ':'.join(self.test)
            
    def __str__(self, concise=True):
        if concise:
            return str({k: self.data[k][:3] + ['...'] for k in self.data})
        else:
            return str(self.data)
    
    def set(self, result):
        """Store new result value."""
        assert self.collecting == True, 'Can only set value in collecting mode.'
        self.data[self.test].append(result)
    
    def get(self):
        """Get next result value."""
        assert self.collecting == False, 'Cannot get value in collecting mode.'
#        value = self.data[self.testname()][self.j]
#        try:
#            values = self.data[self.testname()]
#        except:
#            raise Error("Could not access (sub)-test '{}'.".format(self.testname()))
        Error_if(not self.test in self.data,
                 "Could not access (sub)-test '{}'.".format(self.testname()))
        values = self.data[self.test]
#        try:
#            value = values[self.j]
#        except:
#            raise Error("Could not get index {} of (sub)-test '{}'.".format(self.j, self.testname()),
#                        "The (sub)-test has {} values:".format(len(values)),
#                        values)
        Error_if(self.index[self.test] >= len(values),
                 "Could not get index {} of (sub)-test '{}'.".format(self.index[self.test], self.testname()),
                 "The (sub)-test has {} values:".format(len(values)),
                 values)
        value = values[self.index[self.test]]
        self.index[self.test] += 1
        return value
    
    def set_global(self, key, value):
        """Store value."""
        if self.collecting:   # do not do anything if not collecting
            assert not key.startswith('test_'), 'The key must not start with "test_".'
            self.data[(key,)] = value
    
    def get_global(self, key):
        """Get stored value."""
        assert not key.startswith('test_'), 'The key must not start with "test_".'
        return self.data[(key,)]
        
    def start_new_test(self, name):
        self.finish_test()
        if self.collecting:
            Error_if(name in self.data,
                     "Test name '{}' is not unique.".format(name))
        self.start_new_subtest(name)
            
    def start_new_subtest(self, name):
#        assert not ':' in name, "Character ':' not allowed in (sub)-test name."
        self.test += (name, ) 
        if self.collecting:
            if not self.test in self.data:
                self.data[self.test] = []
        else:
            Error_if(not self.test in self.data,
                     "(Sub)-Test name '{}' does not exist in results.".format(self.testname()))
            
    def finish_test(self):
        self.test = tuple()
            
    def finish_subtest(self):
        self.test = self.test[:-1]
        
    def store_data(self, key, value):
        if self.collecting:
            self.data[('stored', key)] = value
            
    def read_data(self, key):
        return self.data[('stored', key)]
        
    def dump(self):
        """Dump string which can be evaluated to store found results
        from collecting phase into object instance 'results'."""
        if self.collecting:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.data, f)
            with open(self.filename + '.pprint', 'w') as f:
                self.show(stream=f, concise=False)
            print()
            print("Results written to file '{}'.".format(self.filename))
            print()
            print('Results in concise form:')
            self.show()
            
    def show(self, stream=None, concise=True):
        joined_keys_dict = d = {'-'.join(k): self.data[k] for k in sorted(self.data)}
        if concise:
            shortened = {k: d[k] if len(d[k]) <= 5 else d[k][:5] + ['...'] for k in d}
            for k in shortened:
                for i, l in enumerate(shortened[k]):
                    try:
                        if len(l) > 5:
                            shortened[k][i] = l[:5] + ['...']
                    except:
                        pass
            ppstr = pprint.pformat(shortened, compact=True, depth=3)
            print(ppstr.replace("'...'", "..."), file=stream)
        else:
            pprint.pprint(joined_keys_dict, stream)

    
import_block = """
      import numpy as np
      import matplotlib.pyplot as plt
"""

def find_import(code, import_block=import_block):
    lines = [s.strip() for s in re.split(r'[\n\r]+', import_block)]   # get lines
    re_lines = [re.sub(r'\s+', r'\\s+', s) for s in lines if s]       # turn spaces into regex for arbitrary space
    for re_line in re_lines:                                          # remove those lines
        code = re.sub(re_line, '', code)
    return [m.group() for m in re.finditer(r'.*import.*', code)]      # return lines that contain import statement

def mixed_normal_cdf(M, S, P):
    assert len(M) == len(S)
    assert 0 <= len(M) - len(P) <= 1
    if len(P) < len(M):
        P = np.append(P, [0])   # last element is automatically corrected by np.random.multinomial
    def cdf(x):
        return np.sum(np.array([norm(m, s).cdf(x) for m, s in zip(M, S)]) * P)
    return cdf

data = [(-4, 4), (2, 4), (0.5, 0.5)]
mixed_normal_2_cdf = mixed_normal_cdf(*data)

data = [(-5, 0, 5), (1, 1, 1), (1/3, 1/3, 1/3)]
mixed_normal_3_cdf = mixed_normal_cdf(*data)

M = np.linspace(0, 100, 11)
S = np.linspace(2, 5, 11)
P = np.array([50, 41, 34, 29, 26, 25, 26, 29, 34, 41, 50]) / 385
data = M, S, P
mixed_normal_11_cdf = mixed_normal_cdf(*data)

def bins_probs_full(cdf, bins, a, b):
    """Return vector of probabilities according to the cumulative distribution function
    cdf of length bins+2 for the probabilities of "beeing <= a", "beeing in interval
    [a+hi, a+h(i+1) for h=(b-a)/bins and i in range(bins)" and "beeing >= b"."""
    probs = np.empty(bins+2)
    probs[0] = cdf(a)
    probs[-1] = 1 - cdf(b)
    edges = np.linspace(a, b, bins + 1)
    probs[1:-1] = [cdf(r) - cdf(l) for l, r in zip(edges[:-1], edges[1:])]
    return probs

def hist_full(X, bins, a, b):
    """Return histogram with bins-many bins equally spaced between a and b
    and too further bins left of a and right of b."""
    hist, edges = np.histogram(X, bins=bins, range=(a, b))
    return np.array([np.sum(X<a)] + list(hist) + [np.sum(X>b)])

def test_sample(X, a, b, bins, cdf, parts):
    """Return True if certain sub-samples of X with size len(X)/parts
    are distributed according to cdf."""
    
    n = round(len(X)/parts)   # size of partial samples
    hist_exp = bins_probs_full(cdf, bins, a, b) * n
    ppf99 = chi2(bins+1).ppf(0.99)   # chi2 statistic value corresponding to cdf of 0.99
    hists1 = [hist_full(X[i*n:(i+1)*n], bins, a, b) for i in range(parts)]   # partition X into parts blocks
    hists2 = [hist_full(X[i::parts], bins, a, b) for i in range(parts)]      # partition X into parts arithmetic progressions
    OK = True
    goodnesses = []
    # print(hists1)
    # calculate a chi2 statistic for each of the above partitions
    for hists in hists1, hists2:
        chi2_statistic = [np.sum((hist - hist_exp)**2/hist_exp) for hist in hists]
        # plot statistic
        # plt.hist(chi2_statistic, density=True, bins=round(parts/10), histtype='stepfilled', alpha=0.7)
        # x = np.linspace(min(chi2_statistic), max(chi2_statistic), 1000)
        # plt.plot(x, chi2(bins+1).pdf(x), 'k')   # plot relevant chi2 density function
        good_count = sum([1 for s in chi2_statistic if s <= ppf99])   # count statistics below threshold
        goodnesses.append(good_count / len(chi2_statistic))
        OK &= (goodnesses[-1]) >= 0.9               # test is OK if ALL partitions are OK
    # plt.show()
    return OK, goodnesses

distributions = [
    main.normal_10_3,
    main.uniform_100_10,
    main.mixed_normal_2,
    main.mixed_normal_3,
    main.mixed_normal_11,
    ]

loc, scale = 100-10*np.sqrt(3), 10*np.sqrt(12)

data = {
        main.normal_10_3:        (  -2,  24, 10, norm(10,  3).cdf),          # a, b, bins, cdf
        main.uniform_100_10:     (  85, 115, 10, uniform(loc, scale).cdf),   # a, b, bins, cdf
        main.mixed_normal_2:     (  -8,  12, 10, mixed_normal_2_cdf),        # a, b, bins, cdf
        main.mixed_normal_3:     (  -6,   6, 10, mixed_normal_3_cdf),        # a, b, bins, cdf
        main.mixed_normal_11:    (  -1, 102, 10, mixed_normal_11_cdf),       # a, b, bins, cdf
       }

rng = np.random.default_rng(171717)
# rng = np.random.default_rng()   # uncomment this for randomness
#quick_and_dirty = False
quick_and_dirty = True          # uncomment this for speed-up

class Test_Praktikum(Test_Numerik):
    
    try:
        true_values_data = main.true_values
    except (NameError, AttributeError):
        true_values_data = dict()

    def true_values(self, key, default='no_value_provided_by_main'):
        return self.true_values_data.get(key, default)

    def test_Bayes_Formel(self):
        print('Running "test_Bayes_Formel" ...')
        for P_B_A, P_A, P_B in product((0, 0.2, 0.4, 1), (0, 0.3, 0.5, 1), (0.1, 0.4, 0.7, 1)):
            with self.subTest(msg=f'Bayes_Formel ist nicht korrekt für Parameter {P_B_A}, {P_A}, {P_B}.'):
                self.runner(main.Bayes_Formel, (P_B_A, P_A, P_B, ), self.assertAlmostEqualRelative)

    def test_totale_Wahrscheinlichkeit(self):
        print('Running "test_totale_Wahrscheinlichkeit" ...')
        for n in range(2, 11):
            for e1, e2 in [(2, 3), (5, 6)]:
                P_Ai = np.linspace(1, n**e1, n)**e2
                P_Ai /= np.sum(P_Ai)
                for s in np.linspace(0, 1, 5):
                    for e3, e4 in [(7, 8), (12, 13)]:
                        P_B_Ai = np.linspace(1, n**e3, n)**e4
                        P_B_Ai *= s/np.sum(P_B_Ai)
                        with self.subTest(msg=f'totale_Wahrscheinlichkeit für Parameter {P_Ai}, {P_B_Ai}.'):
                            self.runner(main.totale_Wahrscheinlichkeit, (P_Ai, P_B_Ai, ), self.assertAlmostEqualRelative)

    def test_Test(self):
        print('Running "test_Test" ...')
        for Korrektheit, a_priori_Wahrscheinlichkeit in product(np.linspace(0.1, 1, 10), np.linspace(0.1, 1, 10)):
            with self.subTest(msg=f'Test ist nicht korrekt für Parameter {Korrektheit}, {a_priori_Wahrscheinlichkeit}.'):
                self.runner(main.Test, (Korrektheit, a_priori_Wahrscheinlichkeit, ), self.assertAlmostEqualRelative)

    def test_Verteilungen(self):
        print('Running "test_Verteilungen" ...')
        for distr in distributions:
            name = distr.__name__
            print(f'   Testing "{name}" ...')
            a, b, bins, cdf = data[distr]
            if quick_and_dirty:
                n = 10**6
            else:
                # n = 10**7
                n = 10**8
            parts = 100
            def test():
                X = distr(rng, n)
                return test_sample(X, a, b, bins, cdf, parts)
            with self.subTest(msg=f'Verteilung {name} nicht korrekt.', name=name):
                self.runner(test, (), self.assertEqualComponent, msg='Result of test: {}')
                
    def test_GdgZ(self):
        print('Running "test_GdgZ" ...')
        for distr in distributions:
            name = distr.__name__
            print(f'   Testing "{name}" ...')
            no_samples_max = 100
            if quick_and_dirty:
                no_runs, dist_max = 10**3, 0.06
            else:
                no_runs, dist_max = 10**4, 0.03
            def distance():
                rel_variances = [main.rel_variance_of_mean(rng, distr, no_samples, no_runs) for no_samples in range(1, no_samples_max + 1)]
                return np.sqrt(np.sum((rel_variances - 1/np.arange(1, no_samples_max+1))**2))
            with self.subTest(msg=f'GdgZ nicht korrekt für Verteilung {name}.', name=name):
                self.runner(distance, (), self.assertLessEqualConstant, constant=dist_max)
                
    def test_ZGWS(self):
        print('Running "test_ZGWS" ...')
        if quick_and_dirty:
            no_runs, relative = 10**5, 0.5
        else:
            no_runs, relative = 10**6, 0.05
        n1, n2 = 2, 10
        N = np.arange(n1, n2+1)
        a, b, bins = -2, 2, 4
        for distr in distributions:
            name = distr.__name__
            print(f'   Testing "{name}" ...')
            for n in N:
                def histogram():
                    X = main.centralized_sample(rng, distr, n, no_runs)
                    return hist_full(X, bins, a, b) / no_runs
                with self.subTest(msg=f'ZGWS ist nicht korrekt für Verteilung {name} und Anzahl {n}.', name=name):
                    self.runner(histogram, (), self.assertAlmostEqualRelative, relative=relative)
                
    def test_Dateikonsistenz(self):
        print('Running "test_Dateikonsistenz"...')
        for file in files_tested_for_correctness:
            with self.subTest(msg='Datei {} ist verändert worden oder nicht lesbar.'.format(file), name=file):
                self.runner(Hash_file, (file, ), self.assertEqual)

    def test_import_restrictions(self):
        print('Running "test_import_restrictions"...')
        import_block_indented = '\n      '.join([s.strip() for s in re.split(r'[\n\r]+', import_block) if s.strip()])
        for file in files_tested_for_import_restrictions:
            with open(file) as f:
                imports = '\n      '.join(find_import(f.read()))
                if imports:
                    Error(f"""
                          
   Die Datei "{file}" darf keine anderen import-Befehle enthalten als folgende:
   
      {import_block_indented}
   
   Folgende Befehle sind zusätzlich enthalten:
       
      {imports}
                          
""")

    def test_info(self):
        print('Running "test_info"...')
        output = '''
        
   Die Datei "{}" muss korrekt angegeben werden.

   FEHLER:

   {{}}
'''.format(info_file)
        # Info file must be readable
        try:
            with open(info_file, 'rb') as f:
                info_byte = f.read()
        except:
#            self.fail(output.format('Info-Datei "{}" konnte nicht gelesen werden.'.format(info_file)))
            Error(output.format('Info-Datei "{}" konnte nicht gelesen werden.'.format(info_file)))
        try:
            f.close()   # On some systems the above with statement does not close file
        except:
            pass
        # Info file must be utf-8
        try:
            info = str(info_byte, 'utf-8')
        except:
            Error(output.format('''Die Info-Datei ist nicht "UTF-8"-kodiert.
Editieren Sie die vorgegebene Datei in "Spyder" (oder einem
anderen Editor, der Ihnen erlaubt die Codierung festzustellen), und
überprüfen Sie dort, dass rechts unten "Encoding: UTF-8" steht.
'''))
        # Info file must be splittable into headings and data
        try:
            info_list = re.split('\s*##+\s*', info)
            info_dict = {}
            for entry in info_list:
                data = re.split('[\n\r]+', entry, maxsplit=1) + ['']
                info_dict[data[0]] = data[1]
        except:
            Error(output.format('Die Info-Datei hat falsches Format, Überschriften können nicht zugeordnet werden.'))
        # info file must contain correct headings
        for key in ('Team', 'Teilnehmer:in 1', 'Teilnehmer:in 2', 'Email 1', 'Email 2', 'Quellen', 'Bemerkungen', 'Bestätigung'):
            if not key in info_dict:
                Error(output.format('Die Überschrift "## {}" ist nicht korrekt angegeben.'.format(key)))
        if re.fullmatch('\s*Teamname\s*', info_dict['Team']):
            Error(output.format('"Teamname" kann nicht als Teamname gewählt werden, wählen Sie einen eigenen.'))
        if not re.fullmatch('\s*[^\n]+\s*', info_dict['Team']):
            Error(output.format('Das Format für Teamname ist nicht korrekt.'))
        if info_dict['Teilnehmer:in 1'].count(',') != 1:
            Error(output.format('Format "Nachname(n), Vorname(n)" für "Teilnehmer:in 1" ist nicht korrekt.'))
        if info_dict['Teilnehmer:in 2'].count(',') != 1 and not 'LEER' in info_dict['Teilnehmer:in 2']:
            Error(output.format('Format "Nachname(n), Vorname(n)" bzw. "LEER" für "Teilnehmer:in 2" ist nicht korrekt.'))
        if not re.fullmatch('\s*[^\s<>"\',;]+@[^\s<>"\',;]+\.[^\s<>"\',;]+\s*', info_dict['Email 1']) and  not 'LEER' in info_dict['Email 2']:
            Error(output.format('Email Format für "Teilnehmer:in 1" ist nicht korrekt.'))
        if not re.fullmatch('\s*[^\s<>"\',;]+@[^\s<>"\',;]+\.[^\s<>"\',;]+\s*', info_dict['Email 2']) and  not 'LEER' in info_dict['Email 2']:
            Error(output.format('Email Format für "Teilnehmer:in 2" ist nicht korrekt.'))
        if not re.fullmatch('\s*(Ich|Wir|Ich\s*.\s*wir)\s*bestätigen\s*.?\s*dass\s*wir\s*nur\s*die\s*angegebenen\s*Quellen\s*benutzt\s*haben?.?\s*', info_dict['Bestätigung'], flags=re.I+re.S):
            Error(output.format('Die Bestätigung ist nicht korrekt.'))
        # info_dict['ID'] = ', '.join([f'[{key}: {info_dict[key]}]' for key in ('Team', 'Teilnehmer:in 1', 'Teilnehmer:in 2')])
        # info_dict['hexID'] = hashlib.sha1(bytes(info_dict['ID'], encoding='utf8')).hexdigest()
        # info_dict['intID'] = int(info_dict['hexID'], 16)
        # return info_dict
            
    def test_test_control(self):
        print('Running "test_test_control"...')
        Error_if(skipped_tests, 'Dictionary "test_control" ist definiert und Tests werden ausgelassen!',
                 '\nFolgende Tests wurden übersprungen (test_Methode und übersprungene Nummern):\n',
                 '\n\n'.join([f'{t}:\n{" ".join(map(str, skipped_tests[t]))}' for t in skipped_tests]))

        
def scramble(object):
    if not type(object) == bytes:
        object = bytes(str(object), encoding='utf8')
    try:
        return str(base64.b64encode(hashlib.sha1(object).digest())[:10], encoding='ascii')
    except:
        return 'Hashwert nicht berechenbar !!!'

def scramble_float(precision=5):
    def scrmble(x):
        try:
            x = float(x)
        except:
            Error('Der angegebene Wert ist kein "float".')
        return scramble('{:.{p}e}'.format(x, p=precision))
    return scrmble

def Hash_file(file_name):
    try:
        with open(file_name, 'rb') as f:
            content = f.read()
        return scramble(content)
    except:
        return 'Datei nicht lesbar !!!'
    
def Error(*args):
    raise ERROR('\n'.join(map(str, args)))
        
def Error_if(condition, *args):
    if condition:
        Error(*args)


if __name__ == '__main__':
    
    collect_results = False
    try:
        Results(data_file, collect_results)
    except:
        print()
        print('The data file {} cannot be read.'.format(data_file))
        print('Trying to produce a new one.')
        try:
            import test_config
            if test_config.collect_results_automatically:
                collect_results = True
                print('Module test_config found.')
                print('Automatic results collection is switched on.')
        except:
            print('No module test_config.')
            print('No automatic results collection.')
        print()
    try:
        if 'collect' == sys.argv[1]:
            collect_results = True
    except:
        pass
    results = Results(data_file, collect_results)
    unittest.main(argv=[sys.argv[0]])
    results.dump()
