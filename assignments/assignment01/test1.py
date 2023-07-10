# -*- coding: utf-8 -*-
"""
@author: Nils Rosehr
"""

import matplotlib
matplotlib.use("Agg")

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


P_number = 1
exec('import main{}'.format(P_number))
exec('main = main{}'.format(P_number))
# exec('import lib{}'.format(P_number))
# exec('lib = lib{}'.format(P_number))
# exec('import vorgabe{}'.format(P_number))
# exec('vorgabe = vorgabe{}'.format(P_number))
test_file = 'test{}.py'.format(P_number)
vorgabe_file = 'vorgabe{}.py'.format(P_number)
data_file = 'test{}.data'.format(P_number)
lib_file = 'lib{}.py'.format(P_number)
info_file = 'info{}.md'.format(P_number)
files_tested_for_correctness = [test_file]
# files_tested_for_correctness = [test_file, vorgabe_file]
_subtest_msg_sentinel = object()
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

    def assertLessEqualMultiple(self, result, true_value, multiple=2):
        self.assertLessEqual(result, true_value * multiple)
    
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

    
def standardize_sign(QlY):
    # return QlY
    Q, l, Y = QlY
    # calculate diagonal matrix with eigenvalues ±1 to correct signs of Q and Y
    D = np.identity(len(Q))
    for i, col in enumerate(Q.T):
        for q in col:
            if q > 0:
                break
            if q < 0:
                D[i, i] = -1
                break
    return Q @ D, l, Y @ D


class Test_Praktikum(Test_Numerik):
    
    try:
        true_values_data = main.true_values
    except (NameError, AttributeError):
        true_values_data = dict()

    def true_values(self, key, default='no_value_provided_by_main'):
        return self.true_values_data.get(key, default)

    @staticmethod
    def random_matrix(n, m, domain='reell'):
        if domain == 'reell':
            return np.random.random(n*m).reshape(n,m)
        else:
            return np.random.random(n*m).reshape(n,m) + np.random.random(n*m).reshape(n,m) * 1j


    np.random.seed(171717)
    x1 = [17]
    x2 = [4.1, 4.5, 4.9, 5.1, 5.4, 5.7, 5.8, 5.8, 6, 67]
    x3 = np.random.randn(100)*10 + 50
    x4 = np.random.randn(10000)
    data_name_pairs_x = [(x, f'x{i+1}') for i, x in enumerate([x1, x2, x3, x4])]
    xy1 = [1, 2], [3, 4]
    xy2 = [1, 2, 5, 6], [3, 1, 5, 3]
    xy3 = np.array([[x + 5*np.random.random(), 3*x + 20 + 5*np.random.random()] for x in np.linspace(30, 50, 10)]).T
    xy4 = np.array([[x + 50*np.random.random(), -3*x + 20 + 50*np.random.random()] for x in np.linspace(30, 50, 1000)]).T
    data_name_pairs_xy = [(x, f'x{i+1}', y, f'y{i+1}') for i, (x, y) in enumerate([xy1, xy2, xy3, xy4])]
    X1 = np.array([1, 2, 5, 6,     3, 1, 5, 3]).reshape(2, -1).T
    X2 = np.array([1, 2, 3, 4, 5,  2, 1, 5, 4, 3]).reshape((2, -1)).T
    X3 = np.vstack([np.random.randn(10, 2)*5 + s for s in ((50, 50), (70, 30))])
    X4 = np.vstack([np.random.randn(10000, 20) + s for s in [np.random.randn(20)*100 + 500 for _ in range(5)]])
    data_name_pairs_X = [(X, f'X{i+1}') for i, X in enumerate([X1, X2, X3, X4])]
    

    def test_mittel(self):
        for a, name in self.data_name_pairs_x:
            with self.subTest(msg=f'Mittelwert ist nicht korrekt für x={name}.', x=a, name=name):
                self.runner(main.mittel, (a, ), self.assertAlmostEqualRelative)

    def test_quantil(self):
        for a, name in self.data_name_pairs_x:
            for p in [0, 0.123, 0.789, 0.99, 1]:
                with self.subTest(msg=f'Quantil ist nicht korrekt für x={name}, p={p}.', x=a, name=name+'_' + str(p)):
                    self.runner(main.quantil, (a, p), self.assertAlmostEqualRelative)

    def test_median(self):
        for a, name in self.data_name_pairs_x:
            with self.subTest(msg=f'Median ist nicht korrekt für x={name}.', x=a, name=name):
                self.runner(main.median, (a, ), self.assertAlmostEqualRelative)
                
    def test_var(self):
        for a, name in self.data_name_pairs_x:
            with self.subTest(msg=f'Varianz ist nicht korrekt für x={name}.', x=a, name=name):
                self.runner(main.var, (a, ), self.assertAlmostEqualRelative)
                
    def test_regress(self):
        for a, name1, b, name2 in self.data_name_pairs_xy:
            with self.subTest(msg=f'Varianz ist nicht korrekt für x={name1}, y={name2}.', x=a, y=b, name=name1 + '_' + name2):
                self.runner(main.regress, (a, b), self.assertAlmostEqualRelative)

    def test_pca(self):
        for a, name in self.data_name_pairs_X:
            with self.subTest(msg=f'PCA ist nicht korrekt für X={name}.', X=a, name=name):
                self.runner(main.pca, (a, ), self.assertAlmostEqualPlaces, post=standardize_sign)
            
    def test_Dateikonsistenz(self):
        for file in files_tested_for_correctness:
            with self.subTest(msg='Datei {} ist verändert worden oder nicht lesbar.'.format(file), name=file):
                self.runner(Hash_file, (file, ), self.assertEqual)

    def test_info(self):
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
