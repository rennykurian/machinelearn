
Python¶
Python is a high-level, dynamically typed multiparadigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable. As an example, here is an implementation of the classic quicksort algorithm in Python:

In [110]:
def quicksort(eip):
    if len(eip) <= 1:
        return eip
    mlbr = eip[len(eip) // 2]
    left_eip = [x for x in eip if x < mlbr]
    middle_eip = [x for x in eip if x == mlbr]
    right_eip = [x for x in eip if x > mlbr]
    return quicksort(left_eip) + middle_eip + quicksort(right_eip)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"
[1, 1, 2, 3, 6, 8, 10]
Python versions
There are currently two different supported versions of Python, 2.7 and 3.5. Somewhat confusingly, Python 3.0 introduced many backwards-incompatible changes to the language, so code written for 2.7 may not work under 3.5 and vice versa. For this class all code will use Python 3.5.

You can check your Python version at the command line by running python --version.

Basic data types
Like most languages, Python has a number of basic types including integers, floats, booleans, and strings. These data types behave in ways that are familiar from other programming languages.

Numbers: Integers and floats work as you would expect from other language

In [111]:
eip = 3
print(type(eip)) # Prints "<class 'int'>"
print(eip)       # Prints "3"
print(eip + 1)   # Addition; prints "4"
print(eip - 1)   # Subtraction; prints "2"
print(eip * 2)   # Multiplication; prints "6"
print(eip ** 2)  # Eeipponentiation; prints "9"
eip += 1
print(eip)  # Prints "4"
eip *= 2
print(eip)  # Prints "8"
mlbr = 2.5
print(type(mlbr)) # Prints "<class 'float'>"
print(mlbr, mlbr + 1, mlbr * 2, mlbr ** 2) # Prints "2.5 3.5 5.0 6.25"
<class 'int'>
3
4
2
6
9
4
8
<class 'float'>
2.5 3.5 5.0 6.25
Note that unlike many languages, Python does not have unary increment (x++) or decrement (x--) operators.

Python also has built-in types for complex numbers; you can find all of the details in the documentation.

Booleans: Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (&&, ||, etc.):

In [112]:
eip_in = True
eip_out = False
print(type(eip_in)) # Prineip_ins "<class 'bool'>"
print(eip_in and eip_out) # Logical AND; prineip_ins "EIP_OUTalse"
print(eip_in or eip_out)  # Logical OR; prineip_ins "EIP_INrue"
print(not eip_in)   # Logical NOEIP_IN; prineip_ins "EIP_OUTalse"
print(eip_in != eip_out)  # Logical XOR; prineip_ins "EIP_INrue"
<class 'bool'>
False
True
False
True
Strings: Python has great support for strings:

In [113]:
mlbr_in = 'hello'    # String literals can use single quotes
mlbr_out = "world"    # or double quotes; it does not matter.
print(mlbr_in)       # Prints "hello"
print(len(mlbr_out))  # String length; prints "5"
hw = mlbr_in + ' ' + mlbr_out  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (mlbr_in, mlbr_out, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
hello
5
hello world
hello world 12
String objects have a bunch of useful methods; for example:

In [114]:
eip = "hello"
print(eip.capitalize())  # Capitalize a string; prints "Hello"
print(eip.upper())       # Convert a string to uppercase; prints "HELLO"
print(eip.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(eip.center(7))     # Center a string, padding with spaces; prints " hello "
print(eip.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
Hello
HELLO
  hello
 hello 
he(ell)(ell)o
world
You can find a list of all string methods in the documentation

Containers
Python includes several built-in container types: lists, dictionaries, sets, and tuples.

Lists A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:

In [115]:
eip_list = [3, 1, 2]    # Create a list
print(eip_list, eip_list[2])  # Prints "[3, 1, 2] 2"
print(eip_list[-1])     # Negative indices count from the end of the list; prints "2"
eip_list[2] = 'foo'     # Lists can contain elements of different types
print(eip_list)         # Prints "[3, 1, 'foo']"
eip_list.append('bar')  # Add a new element to the end of the list
print(eip_list)         # Prints "[3, 1, 'foo', 'bar']"
mlbr = eip_list.pop()      # Remove and return the last element of the list
print(mlbr, eip_list)      # Prints "bar [3, 1, 'foo']"
[3, 1, 2] 2
2
[3, 1, 'foo']
[3, 1, 'foo', 'bar']
bar [3, 1, 'foo']
As usual, you can find all the gory details about lists in the documentation

Slicing: In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:

In [116]:
mlbr_list = list(range(5))     # range is a built-in function that creates a list of integers
print(mlbr_list)               # Prints "[0, 1, 2, 3, 4]"
print(mlbr_list[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(mlbr_list[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(mlbr_list[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(mlbr_list[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(mlbr_list[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
mlbr_list[2:4] = [8, 9]        # Assign a new sublist to a slice
print(mlbr_list)               # Prints "[0, 1, 8, 9, 4]"
[0, 1, 2, 3, 4]
[2, 3]
[2, 3, 4]
[0, 1]
[0, 1, 2, 3, 4]
[0, 1, 2, 3]
[0, 1, 8, 9, 4]
We will see slicing again in the context of numpy arrays.
Loops: You can loop over the elements of a list like this:

In [117]:
eip_list = ['cat', 'dog', 'monkey']
for mlbr_out in eip_list:
    print(mlbr_out)
# Prints "cat", "dog", "monkey", each on its own line.
cat
dog
monkey
If you want access to the index of each element within the body of a loop, use the built-in enumerate function:

In [118]:
mlbr_list = ['cat', 'dog', 'monkey']
for eip, mlbr in enumerate(mlbr_list):
    print('#%d: %s' % (eip + 1, mlbr))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line: When programming, frequently we want to transform one type of data into another. As a simple example, consider the following code that computes square numbers:
#1: cat
#2: dog
#3: monkey
List comprehensions: When programming, frequently we want to transform one type of data into another. As a simple example, consider the following code that computes square numbers:

In [119]:
eip_list = [0, 1, 2, 3, 4]
mlbr_list = []
for eip in eip_list:
    mlbr_list.append(eip ** 2)
print(mlbr_list)   # Prints [0, 1, 4, 9, 16]
[0, 1, 4, 9, 16]
You can make this code simpler using a list comprehension:

In [120]:
eip_list = [0, 1, 2, 3, 4]
mlbr_list = [eip ** 2 for eip in eip_list]
print(mlbr_list)   # Prints [0, 1, 4, 9, 16]
[0, 1, 4, 9, 16]
List comprehensions can also contain conditions:

In [121]:
eip_list = [0, 1, 2, 3, 4]
even_mlbr_list = [eip ** 2 for eip in eip_list if eip % 2 == 0]
print(even_mlbr_list)  # Prints "[0, 4, 16]"
[0, 4, 16]
Dictionaries
A dictionary stores (key, value) pairs, similar to a Map in Java or an object in Javascript. You can use it like this:

In [122]:
eip_dict = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(eip_dict['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in eip_dict)     # Check if a dictionary has a given key; prints "True"
eip_dict['fish'] = 'wet'     # Set an entry in a dictionary
print(eip_dict['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(eip_dict.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(eip_dict.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del eip_dict['fish']         # Remove an element from a dictionary
print(eip_dict.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
cute
True
wet
N/A
wet
N/A
You can find all you need to know about dictionaries in the documentation

Loops: It is easy to iterate over the keys in a dictionary:

In [123]:
eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for mlbr in eip_dict:
    mlbr_out = eip_dict[mlbr]
    print('A %s has %d legs' % (mlbr, mlbr_out))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
A person has 2 legs
A cat has 4 legs
A spider has 8 legs
If you want access to keys and their corresponding values, use the items method:

In [124]:
eip_dict = {'person': 2, 'cat': 4, 'spider': 8}
for eip, mlbr in eip_dict.items():
    print('A %s has %d legs' % (eip, mlbr))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
A person has 2 legs
A cat has 4 legs
A spider has 8 legs
Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. For example:

In [125]:
eip_list = [0, 1, 2, 3, 4]
mlbr = {eip: eip ** 2 for eip in eip_list if eip % 2 == 0}
print(mlbr)  # Prints "{0: 0, 2: 4, 4: 16}"
{0: 0, 2: 4, 4: 16}
Sets
A set is an unordered collection of distinct elements. As a simple example, consider the following:

In [126]:
eip_dict = {'cat', 'dog'}
print('cat' in eip_dict)   # Check if an element is in a set; prints "True"
print('fish' in eip_dict)  # prints "False"
eip_dict.add('fish')       # Add an element to a set
print('fish' in eip_dict)  # Prints "True"
print(len(eip_dict))       # Number of elements in a set; prints "3"
eip_dict.add('cat')        # Adding an element that is already in the set does nothing
print(len(eip_dict))       # Prints "3"
eip_dict.remove('cat')     # Remove an element from a set
print(len(eip_dict))       # Prints "2"
True
False
True
3
3
2
As usual, everything you want to know about sets can be found in the documentation.

Loops: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:

In [127]:
mlbr = {'cat', 'dog', 'fish'}
for eip_in, eip_out in enumerate(mlbr):
    print('#%d: %s' % (eip_in + 1, eip_out))
# Prints "#1: fish", "#2: dog", "#3: cat"
#1: dog
#2: cat
#3: fish
Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:

In [128]:
from math import sqrt
mlbr = {int(sqrt(eip)) for eip in range(30)}
print(mlbr)  # Prints "{0, 1, 2, 3, 4, 5}"
{0, 1, 2, 3, 4, 5}
Tuples
A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:

In [129]:
eip_dict = {(eip, eip + 1): eip for eip in range(10)}  # Create a dictionary with tuple keys
mlbr = (5, 6)        # Create a tuple
print(type(mlbr))    # Prints "<class 'tuple'>"
print(eip_dict[mlbr])       # Prints "5"
print(eip_dict[(1, 2)])  # Prints "1"
<class 'tuple'>
5
1
The documentation has more information about tuples.

Functions
Python functions are defined using the def keyword. For example:

In [130]:
def sign(eip):
    if eip > 0:
        return 'positive'
    elif eip < 0:
        return 'negative'
    else:
        return 'zero'

for mlbr in [-1, 0, 1]:
    print(sign(mlbr))
# Prints "negative", "zero", "positive"
negative
zero
positive
We will often define functions to take optional keyword arguments, like this:

In [131]:
def mlbr(eip_in, mlbr_in=False):
    if mlbr_in:
        print('HELLO, %s!' % eip_in.upper())
    else:
        print('Hello, %s' % eip_in)

mlbr('Bob') # Prints "Hello, Bob"
mlbr('Fred', mlbr_in=True)  # Prints "HELLO, FRED!"
Hello, Bob
HELLO, FRED!
There is a lot more information about Python functions in the documentation

Classes
The syntax for defining classes in Python is straightforward:

In [132]:
class eip(object):

    # Constructor
    def __init__(self, eip_in):
        self.eip_in = eip_in  # Create an instance variable

    # Instance method
    def greet(self, mlbr_in=False):
        if mlbr_in:
            print('HELLO, %s!' % self.eip_in.upper())
        else:
            print('Hello, %s' % self.eip_in)

mlbr = eip('Fred')  # Construct an instance of the Greeter class
mlbr.greet()            # Call an instance method; prints "Hello, Fred"
mlbr.greet(mlbr_in=True)   # Call an instance method; prints "HELLO, FRED!"
Hello, Fred
HELLO, FRED!
You can read a lot more about Python classes in the documentation.

Numpy
Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. If you are already familiar with MATLAB, you might find this tutorial useful to get started with Numpy.

Arrays
A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

In [133]:
import numpy as np

eip = np.array([1, 2, 3])   # Create a rank 1 array
print(type(eip))            # Prints "<class 'numpy.ndarray'>"
print(eip.shape)            # Prints "(3,)"
print(eip[0], eip[1], eip[2])   # Prints "1 2 3"
eip[0] = 5                  # Change an element of the array
print(eip)                  # Prints "[5, 2, 3]"

mlbr = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(mlbr.shape)                     # Prints "(2, 3)"
print(mlbr[0, 0], mlbr[0, 1], mlbr[1, 0])   # Prints "1 2 4"
<class 'numpy.ndarray'>
(3,)
1 2 3
[5 2 3]
(2, 3)
1 2 4
Numpy also provides many functions to create arrays:

In [134]:
import numpy as np

eip = np.zeros((2,2))   # Create an array of all zeros
print(eip)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

mlbr = np.ones((1,2))    # Create an array of all ones
print(mlbr)              # Prints "[[ 1.  1.]]"

eip_in = np.full((2,2), 7)  # Create a constant array
print(eip_in)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

mlbr_in = np.eye(2)         # Create a 2x2 identity matrix
print(mlbr_in)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

eip_out = np.random.random((2,2))  # Create an array filled with random values
print(eip_out)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
[[ 0.  0.]
 [ 0.  0.]]
[[ 1.  1.]]
[[7 7]
 [7 7]]
[[ 1.  0.]
 [ 0.  1.]]
[[ 0.35332528  0.11853759]
 [ 0.94482776  0.69502299]]
You can read about other methods of array creation in the documentation.

Array indexing
Numpy offers several ways to index into arrays.

Slicing:
Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

In [135]:
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
mlbr = eip[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(eip[0, 1])   # Prints "2"
mlbr[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(eip[0, 1])   # Prints "77"
2
77
You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the original array. Note that this is quite different from the way that MATLAB handles array slicing:

In [ ]:
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
eip_in = eip[1, :]    # Rank 1 view of the second row of a
eip_out = eip[1:2, :]  # Rank 2 view of the second row of a
print(eip_in, eip_in.shape)  # Prints "[5 6 7 8] (4,)"
print(eip_out, eip_out.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
mlbr_in = eip[:, 1]
mlbr_out = eip[:, 1:2]
print(mlbr_in, mlbr_in.shape)  # Prints "[ 2  6 10] (3,)"
print(mlbr_out, mlbr_out.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
Integer array indexing_: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:

In [ ]:
import numpy as np

eip = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(eip[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([eip[0, 0], eip[1, 1], eip[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(eip[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([eip[0, 1], eip[0, 1]]))  # Prints "[2 2]"
One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:

In [ ]:
import numpy as np

# Create a new array from which we will select elements
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(eip)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
mlbr = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(eip[np.arange(4), mlbr])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
eip[np.arange(4), mlbr] += 10

print(eip)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Here is an example:

In [ ]:
import numpy as np

eip = np.array([[1,2], [3, 4], [5, 6]])

mlbr = (eip > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(mlbr)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(eip[mlbr])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(eip[eip > 2])     # Prints "[3 4 5 6]"
For brevity we have left out a lot of details about numpy array indexing; if you want to know more you should read the documentation.

Datatypes
Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. Here is an example:

In [ ]:
import numpy as np

eip = np.array([1, 2])   # Let numpy choose the datatype
print(eip.dtype)         # Prints "int64"

mlbr = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(mlbr.dtype)             # Prints "float64"

eip_in = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(eip_in.dtype)                         # Prints "int64"
You can read all about numpy datatypes in the documentation.

Array math
Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:

In [ ]:
import numpy as np

eip = np.array([[1,2],[3,4]], dtype=np.float64)
mlbr = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(eip + mlbr)
print(np.add(eip, mlbr))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(eip - mlbr)
print(np.subtract(eip, mlbr))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(eip * mlbr)
print(np.multiply(eip, mlbr))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(eip / mlbr)
print(np.divide(eip, mlbr))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(eip))
Note that unlike MATLAB, * is elementwise multiplication, not matrix multiplication. We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:

In [ ]:
import numpy as np

eip_in = np.array([[1,2],[3,4]])
mlbr_in = np.array([[5,6],[7,8]])

eip_out = np.array([9,10])
mlbr_out = np.array([11, 12])

# Inner product of vectors; both produce 219
print(eip_out.dot(mlbr_out))
print(np.dot(eip_out, mlbr_out))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(eip_in.dot(eip_out))
print(np.dot(eip_in, eip_out))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(eip_in.dot(mlbr_in))
print(np.dot(eip_in, mlbr_in))
Numpy provides many useful functions for performing computations on arrays; one of the most useful is sum:

In [ ]:
import numpy as np

eip = np.array([[1,2],[3,4]])

print(np.sum(eip))  # Compute sum of all elements; prints "10"
print(np.sum(eip, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(eip, axis=1))  # Compute sum of each row; prints "[3 7]"
You can find the full list of mathematical functions provided by numpy in the documentation.

Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:

In [ ]:
import numpy as np

mlbr = np.array([[1,2], [3,4]])
print(mlbr)    # Prints "[[1 2]
            #          [3 4]]"
print(mlbr.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
eip = np.array([1,2,3])
print(eip)    # Prints "[1 2 3]"
print(eip.T)  # Prints "[1 2 3]"
Numpy provides many more functions for manipulating arrays; you can see the full list in the documentation

Broadcasting
Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.

For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:

In [ ]:
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mlbr = np.array([1, 0, 1])
eip_out = np.empty_like(eip)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for eip_in in range(4):
    eip_out[eip_in, :] = eip[eip_in, :] + mlbr

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(eip_out)
This works; however when the matrix x is very large, computing an explicit loop in Python could be slow. Note that adding the vector v to each row of the matrix x is equivalent to forming a matrix vv by stacking multiple copies of v vertically, then performing elementwise summation of x and vv. We could implement this approach like this:

In [ ]:
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:

In [ ]:
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mlbr = np.array([1, 0, 1])
eip_out = eip + mlbr  # Add v to each row of x using broadcasting
print(eip_out)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
The line y = x + v works even though x has shape (4, 3) and v has shape (3,) due to broadcasting; this line works as if v actually had shape (4, 3), where each row was a copy of v, and the sum was performed elementwise.

Broadcasting two arrays together follows these rules:

If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension. The arrays can be broadcast together if they are compatible in all dimensions. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension If this explanation does not make sense, try reading the explanation from the documentation or this explanation.

Functions that support broadcasting are known as universal functions. You can find the list of all universal functions in the documentation.

Here are some applications of broadcasting:

In [ ]:
import numpy as np

# Compute outer product of vectors
eip = np.array([1,2,3])  # v has shape (3,)
mlbr = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(eip, (3, 1)) * mlbr)

# Add a vector to each row of a matrix
eip_in = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(eip_in + eip)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((eip_in.T + mlbr).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(eip_in + np.reshape(mlbr, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(eip_in * 2)
Broadcasting typically makes your code more concise and faster, so you should strive to use it where possible

Numpy Documentation
This brief overview has touched on many of the important things that you need to know about numpy, but is far from complete. Check out the numpy reference to find out much more about numpy.

SciPy
Numpy provides a high-performance multidimensional array and basic tools to compute with and manipulate these arrays.SciPy builds on this, and provides a large number of functions that operate on numpy arrays and are useful for different types of scientific and engineering applications.

The best way to get familiar with SciPy is to browse the documentation. We will highlight some parts of SciPy that you might find useful for this class.

Image operations
SciPy provides some basic functions to work with images. For example, it has functions to read images from disk into numpy arrays, to write numpy arrays to disk as images, and to resize images. Here is a simple example that showcases these functions:

In [8]:
from scipy.misc import imread, imsave, imresize
from urllib.request import urlopen

# Read an JPEG image into a numpy array
url = 'https://raw.githubusercontent.com/machinelearningblr/machinelearningblr.github.io/2c0aa0c2b7f3531190ed52e9eafbb303b7e8649a/tutorials/CS231n-Materials/assets/cat.jpg'
with urlopen(url) as file:
    img = imread(file,mode='RGB')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('cat_tinted.jpg', img_tinted)
uint8 (400, 248, 3)
Left: The original image. Right: The tinted and resized image

<img src="C:/Users/manjunath.dmurthy/Desktop/MLBLR/img/cat.png", width=30, height=200>

MATLAB files
The functions scipy.io.loadmat and scipy.io.savemat allow you to read and write MATLAB files. You can read about them in the documentation.

Distance between points
SciPy defines some useful functions for computing distances between sets of points.

The function scipy.spatial.distance.pdist computes the distance between all pairs of points in a given set:

In [ ]:
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)
You can read all the details about this function in the documentation.

A similar function (scipy.spatial.distance.cdist) computes the distance between all pairs across two sets of points; you can read about it in the documentation.

Matplotlib
Matplotlib is a plotting library. In this section give a brief introduction to the matplotlib.pyplot module, which provides a plotting system similar to that of MATLAB.

Plotting
The most important function in matplotlib is plot, which allows you to plot 2D data. Here is a simple example:

In [12]:
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.

With just a little bit of extra work we can easily plot mult:iple lines at once, and add a title, legend, and axis labels

In [13]:
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
eip = np.arange(0, 3 * np.pi, 0.1)
eip_out = np.sin(eip)
mlbr_out = np.cos(eip)

# Plot the points using matplotlib
plt.plot(eip, eip_out)
plt.plot(eip, mlbr_out)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

You can read much more about the plot function in the documentation.

Subplots
You can plot different things in the same figure using the subplot function. Here is an example:

In [15]:
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

You can read much more about the subplot function in the documentation.

Images
You can use the imshow function to show images. Here is an example:

In [ ]:
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from urllib.request import urlopen

# Read an JPEG image into a numpy array
url = 'https://raw.githubusercontent.com/machinelearningblr/machinelearningblr.github.io/2c0aa0c2b7f3531190ed52e9eafbb303b7e8649a/tutorials/CS231n-Materials/assets/cat.jpg'
with urlopen(url) as file:
    img = imread(file,mode='RGB')

img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
