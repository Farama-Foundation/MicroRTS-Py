import jpype
from jpype.imports import registerDomain
from jpype.types import *
import numpy as np
from timeit import timeit

"""
Overall idea is to pre-allocate memory shared between JVM and NumPy.
The approch is the following:
* allocate direct byte buffer on JVM side (once)
* map to a NumPy array using np.asarray (once)
* put all observations into flat repr in the buffer (easy for row-based non-jagged array)
* use read only NumPy array view for observations

Still not sure what is the best course of actions for passing actions into env.

```
In [36]: %timeit perf_java_buffer_fill(ByteBuffer, ArrayPerf, repeat=500)
387 ms ± 39.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [37]: %timeit perf_java_array_fill(ByteBuffer, ArrayPerf, repeat=500)
730 ms ± 219 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Order of operations:

```
In [30]: # allocate direct buffer on JVM side for 20 integers (20*4 bytes)
In [31]: jb = ByteBuffer.allocateDirect(80).asIntBuffer()

In [32]: # map this to NumPy array, note that order="C" is default
In [33]: x = np.asarray(jb, order="C")

In [34]: x
Out[34]:
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int32)

In [35]: # reshape gives us view on to the same memory. proof:
In [36]: y = x.reshape(2,2,5)

In [37]: y
Out[37]:
array([[[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],
       [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]], dtype=int32)

In [38]: jb.put(0, 1); jb.put(1, 10)
Out[38]: <java buffer 'java.nio.DirectIntBufferS'>

In [39]: x
Out[39]:
array([ 1, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0], dtype=int32)

In [40]: y
Out[40]:
array([[[ 1, 10,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],
       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]]], dtype=int32)
```


"""

def start_jvm():
    registerDomain("ts", alias="tests")
    jpype.startJVM(classpath=["./perf.jar"], convertStrings=False)

def perf_ndarray(ArrayPerf):
    x = np.random.randint(0, 16, (24,256,78), dtype=np.int32)
    assert x.sum() == ArrayPerf.sumOfArray(JArray.of(x))

def perf_flat_array(ArrayPerf):
    x = np.random.randint(0, 16, (24*256*78,), dtype=np.int32)
    assert x.sum() == ArrayPerf.sumOfFlatArray(JArray.of(x))

def perf_direct_buffer(ArrayPerf):
    x = np.random.randint(0, 16, (24*256*78,), dtype=np.int32)
    buffer = jpype.nio.convertToDirectBuffer(x)
    assert x.sum() == ArrayPerf.sumOfBuffer(buffer.asIntBuffer())

def perf_direct_buffer_shape(ArrayPerf):
    x = np.random.randint(0, 16, (24,256,78), dtype=np.int32)
    buffer = jpype.nio.convertToDirectBuffer(x.reshape(-1))
    assert x.sum() == ArrayPerf.sumOfBuffer(buffer.asIntBuffer())

def perf_java_buffer_fill(ByteBuffer, ArrayPerf, repeat=10):
    nbytes, ntimes = 1916928, 479232
    jb = ByteBuffer.allocateDirect(nbytes).asIntBuffer()
    x = np.asarray(jb)
    for _ in range(repeat):
        ArrayPerf.fillBuffer(jb, ntimes)

def perf_java_array_fill(ByteBuffer, ArrayPerf, repeat=10):
    nbytes, ntimes = 1916928, 479232
    for _ in range(repeat):
        np.array(ArrayPerf.fillArray(ntimes))


if __name__ == "__main__":
    start_jvm()

    from ts import ArrayPerf, ArrayHolder
    from java.nio import ByteBuffer
    
    print("perf_ndarray:")
    timeit(lambda: perf_ndarray(ArrayPerf))

    print("perf_flat_array:")
    timeit(lambda: perf_flat_array(ArrayPerf))

    print("perf_direct_buffer:")
    timeit(lambda: perf_direct_buffer(ArrayPerf))

    print("perf_direct_buffer_shape:")
    timeit(lambda: perf_direct_buffer(ArrayPerf))