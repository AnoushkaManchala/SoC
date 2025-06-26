#Q1
import numpy as np
#Q2
print(np.__version__)
#Q3
np.zeros(10)
#Q4
a = np.zeros(10)
a.size * a.itemsize 
#Q5
python -c "import numpy; help(numpy.add)"
#Q6
z = np.zeros(10)
z[4] = 1
#Q7
np.arange(10, 50)
#Q8
np.arange(10, 50)[::-1]
#Q9
np.arange(9).reshape(3, 3)
#Q10
np.nonzero([1, 2, 0, 0, 4, 0])  # (array([0, 1, 4]),)
#Q11
np.eye(3)
#Q12
np.random.random((3, 3, 3))
#Q13
np.random.random((3, 3, 3))
#Q14
Z = np.random.random((10, 10))
Z.min(), Z.max()
#Q15
Z = np.random.random(30)
Z.mean()
#Q16
Z = np.ones((5, 5))
Z[1:-1, 1:-1] = 0
#Q17
{
  "0 * np.nan": np.nan,
  "np.nan == np.nan": False,
  "np.inf > np.nan": False,
  "np.nan - np.nan": np.nan,
  "np.nan in set([np.nan])": True,
  "0.3 == 3 * 0.1": False
}
#Q18
Z = np.zeros((5, 5))
for i in range(1, 5):
    Z[i, i - 1] = i
#Q19
Z = np.zeros((8, 8))
Z[1::2, ::2] = 1
Z[::2, 1::2] = 1
#Q20
np.unravel_index(100, (6, 7, 8))  # → (1, 5, 4)
#Q21
np.tile([[0, 1], [1, 0]], (4, 4))
#Q22
Z = np.random.random((5, 5))
Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
#Q23
color = np.dtype([("r", np.ubyte), ("g", np.ubyte), ("b", np.ubyte), ("a", np.ubyte)])
#Q24
A = np.ones((5, 3))
B = np.ones((3, 2))
C = np.dot(A, B)
#Q25
Z = np.arange(11)
Z[(3 <= Z) & (Z <= 8)] *= -1
#Q26
sum(range(5), -1)  # → 9 (Python built-in sum starts at -1)
np.sum(range(5), -1)  # → 10 (Numpy treats -1 as axis, not initial)
#Q27
Z = np.arange(5)
# Legal:
Z ** Z
2 << Z >> 2
1j * Z
Z / 1 / 1
# Illegal:
Z <- Z  # invalid
Z < Z > Z  # chained comparison doesn't work this way in NumPy
#Q28
np.array(0) / np.array(0)  # → nan (div by 0)
np.array(0) // np.array(0)  # → 0 (raises warning)
np.array([np.nan]).astype(int).astype(float)  # → weird large negative float
#Q29
Z = np.random.uniform(-10, 10, 10)
Z_rounded = np.copysign(np.ceil(np.abs(Z)), Z)
#Q30
A = np.array([1, 2, 3, 4, 5])
B = np.array([4, 5, 6, 7, 8])
np.intersect1d(A, B)  # → [4 5]
#Q31
import warnings
np.seterr(all='ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    np.ones(1) / 0
#Q32
np.sqrt(-1) == np.emath.sqrt(-1)  # → False
#Q33
today = np.datetime64('today', 'D')
yesterday = today - np.timedelta64(1, 'D')
tomorrow = today + np.timedelta64(1, 'D')
#Q34
july_2016 = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
#Q35
A = np.ones(3)
B = np.ones(3) * 2
A += B        # A becomes [3, 3, 3]
A *= -A / 2   # In-place result: [-4.5, -4.5, -4.5]
#Q36
Z = np.random.uniform(0, 10, 5)
Z.astype(int)
np.floor(Z)
np.trunc(Z)
np.modf(Z)[1]
#Q37
Z = np.zeros((5, 5))
for i in range(5):
    Z[i] = i
#Q38
  def generate():
    for x in range(10):
        yield x
np.fromiter(generate(), dtype=int)
#Q39
Z = np.linspace(0, 1, 12)[1:-1]
#Q40
Z = np.random.random(10)
Z.sort()
#Q41
np.add.reduce(np.arange(10))
#Q42
np.array_equal(A, B)
#Q43
Z = np.zeros(10)
Z.flags.writeable = False
#Q44
Z = np.random.random((10, 2))
X, Y = Z[:, 0], Z[:, 1]
R = np.sqrt(X**2 + Y**2)
T = np.arctan2(Y, X)
#Q45
Z = np.random.random(10)
Z[Z.argmax()] = 0
#Q46
import numpy as np
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
X, Y = np.meshgrid(x, y)

structured = np.zeros(X.shape, dtype=[('x', float), ('y', float)])
structured['x'], structured['y'] = X, Y
print(structured)
#Q47
X = np.arange(8)
Y = X + 0.5

diff = X[:, None] - Y[None, :]
C = 1.0 / diff
print(C)
#Q48
types = [np.int8, np.int16, np.int32, np.int64,
         np.uint8, np.uint16, np.uint32, np.uint64,
         np.float16, np.float32, np.float64]

for dtype in types:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    print(f"{dtype.__name__}: min = {info.min}, max = {info.max}")
#Q49
  np.set_printoptions(threshold=np.inf)  # Show all elements
arr = np.random.rand(100, 100)
print(arr)
#Q50
vec = np.linspace(0, 100, 1000)
target = 42.3

idx = np.abs(vec - target).argmin()
print(vec[idx])
#Q51
structured = np.zeros(10, dtype=[('position', [('x', float), ('y', float)]),
                                 ('color', [('r', float), ('g', float), ('b', float)])])
print(structured)
#Q52
points = np.random.rand(100, 2)

diff = points[:, None, :] - points[None, :, :]
distances = np.sqrt(np.sum(diff**2, axis=-1))
print(distances)
#Q53
arr = np.arange(10, dtype=np.float32)

arr = arr.astype(np.int32)
print(arr)
#Q54
from io import StringIO

txt = StringIO("1, 2, 3, 4, 5\n6,  ,  , 7, 8\n ,  , 9,10,11")
arr = np.genfromtxt(txt, delimiter=",")
print(arr)
#Q55
arr = np.array([[10, 20], [30, 40]])
for idx, val in np.ndenumerate(arr):
    print(f"Index: {idx}, Value: {val}")
#Q56
  x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
D = np.sqrt(X**2 + Y**2)

sigma = 0.4
gaussian = np.exp(- (D**2) / (2 * sigma**2))
print(gaussian)
#Q57
n, m, p = 5, 5, 5
arr = np.zeros((n, m), dtype=int)
idx = np.random.choice(n * m, p, replace=False)
arr[np.unravel_index(idx, (n, m))] = 1
print(arr)
#Q58
mat = np.random.rand(5, 10)
mean = mat.mean(axis=1, keepdims=True)
centered = mat - mean
print(centered)
#Q59
arr = np.random.randint(0, 100, (5, 5))
n = 2  # Sort by 3rd column
sorted_arr = arr[arr[:, n].argsort()]
print(sorted_arr)
#Q60
mat = np.random.randint(0, 2, (5, 10))

null_cols = np.all(mat == 0, axis=0)
print(null_cols)
