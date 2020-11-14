# Import module
import numpy as np
import jpype
from jpype.imports import registerDomain
import jpype.imports
registerDomain("ts", alias="tests")
registerDomain("ai")
from jpype.types import *
jpype.addClassPath("/home/costa/Documents/work/go/src/github.com/vwxyzjn/microrts/microrts.jar")

# Launch the JVM
jpype.startJVM()
from java.lang import System
print(System.getProperty("java.class.path"))
from ts import JNIClient
from ai.rewardfunction import ResourceGatherRewardFunction
rf = ResourceGatherRewardFunction()
jc = JNIClient(rf, "/home/costa/Documents/work/go/src/github.com/vwxyzjn/microrts/", "maps/4x4/baseTwoWorkers4x4.xml")
