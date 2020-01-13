# Import module
import numpy as np
import jpype
from jpype.imports import registerDomain
import jpype.imports
registerDomain("ts", alias="tests")
from jpype.types import *
jpype.addClassPath("/home/costa/Documents/work/go/src/github.com/vwxyzjn/microrts/microrts.jar")

# Launch the JVM
jpype.startJVM()
from java.lang import System
print(System.getProperty("java.class.path"))
from ts import JNIClient
jc = JNIClient("/home/costa/Documents/work/go/src/github.com/vwxyzjn/microrts/")
