#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:07:51 2020

@author: costa
"""

# Import module
import jpype

# Enable Java imports
from jpype.imports import registerDomain
import jpype.imports
# registerDomain("ai")
# registerDomain("gui")
# registerDomain("rts")
# registerDomain("tests")
registerDomain("ts", alias="tests")
# registerDomain("ts", alias="tests.sockets")

# Pull in types
from jpype.types import *

jpype.addClassPath("/home/costa/Documents/work/go/src/github.com/vwxyzjn/microrts/microrts.jar")

# Launch the JVM
jpype.startJVM()

import java.lang
import numpy as np



from java.lang import System
print(System.getProperty("java.class.path"))

from ts import JNIClient
jc = JNIClient()


def convert3DJarrayToNumpy(jArray):
    # get shape
    arr_shape = (len(jArray),)
    temp_array = jArray[0]
    while hasattr(temp_array, '__len__'):
        arr_shape += (len(temp_array),)
        temp_array = temp_array[0]
    arr_type = type(temp_array)
    # transfer data
    resultArray = np.empty(arr_shape, dtype=arr_type)
    for ix in range(arr_shape[0]):
        for i,cols in enumerate(jArray[ix][:]):
            resultArray[ix][i,:] = cols[:]
    return resultArray
