from jpype import JClass
import jpype
import numpy as np
from utils import *

import os

#set up jvm path to use Java
jvmPath = jpype.getDefaultJVMPath()

# we to specify the classpath used by the JVM
classpath=os.getcwd()
jpype.startJVM(jvmPath,'-Djava.class.path=%s' % classpath)

JavaTools = JClass('pottslab/JavaTools')
PLImage = JClass('pottslab/PLImage')

def minL2Potts2DADMM(f, gamma, **kwargs):
    
    # obtain dimensions of image f
    m = f.shape[0]
    n = f.shape[1]


    # fill in inputs
    if 'muInit' not in kwargs:
        kwargs['muInit'] = gamma*1e-2
    
    if 'muStep' not in kwargs:
        kwargs['muStep'] = 2
    
    if 'tol' not in kwargs:
        kwargs['tol'] = 1e-10
    
    if 'isotropic' not in kwargs:
        kwargs['isotropic'] = True
    
    if 'verbose' not in kwargs:
        kwargs['verbose'] = False
    
    if 'weights' not in kwargs:
        kwargs['weights'] = np.ones([m,n])
    
    if 'multiThreading' not in kwargs:
        kwargs['multiThreading'] = True
    
    if 'quantization' not in kwargs:
        kwargs['quantization'] = True
    
    if 'useADMM' not in kwargs:
        kwargs['useADMM'] = True
    
    # check inputs
    assert kwargs['muStep'] > 1, 'Variable muStep must be > 1.'
    assert np.sum(kwargs['weights']<0.) == 0, 'Weights must be >= 0.'
    assert kwargs['tol'] > 0, 'Stopping tolerance must be > 0.'
    assert kwargs['muInit'] > 0, 'muInit must be > 0.'


    # convert image to PLImage
    if len(f.shape) == 3:
        f1 = PLImage(jpype.JArray(float,3)(f))
    else:
        f1 = PLImage(f)
    
    # isotropic vs anisotropic discretization
    if kwargs['isotropic']:
        omega = np.asarray([np.sqrt(2.0)-1, 1.0-np.sqrt(2.0)/2.0])
        result = JavaTools.minL2PottsADMM8(f1, gamma, kwargs['weights'], kwargs['muInit'], kwargs['muStep'], kwargs['tol'], kwargs['verbose'], kwargs['multiThreading'], kwargs['useADMM'], omega)
    else:
        result = JavaTools.minL2PottsADMM4(f1, gamma, kwargs['weights'], kwargs['muInit'], kwargs['muStep'], kwargs['tol'], kwargs['verbose'], kwargs['multiThreading'], kwargs['useADMM'])

    # convert to array
    r = result.toDouble()
    
    r = np.asarray(r)
    
    # reshape
    if len(f.shape) == 3:
        r= np.reshape(r, [m,n,3], order = 'F')
    else:
        r = np.reshape(r, [m,n], order = 'F')

    # remove all small remaining variations in result
    if kwargs['quantization']:
        r = np.round(r*255)/255
    
    return r
    

