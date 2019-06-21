'''
Common function for CNN
'''
import numpy as np
import traceback
def text(v):
    (fn,ln,fn,text) = traceback.extract_stack()[-2]
    begin = text.find('text(')+len('text(')
    end = text.find(')',begin)
    print("\n{0}".format(text[begin:end])," shape:",np.shape(v)," type:",type(v))
#    print(v)

class transFun:
    def __init__(self,name = "sigmod"):
        self.funName = name

    def active(self,z):
        if self.funName == "sigmod":
            return self.sigmod(z)
        elif self.funName == "reLU":
            return self.reLU(z)

    def diff(self,z):
        if self.funName == "sigmod":
            return self.diffSigmod(z)
        elif self.funName == "reLU":
            return self.diffReLU(z)

    def diffOut(self,fz):
        if self.funName == "sigmod":
            return self.diffOutSigmod(fz)
        elif self.funName == "reLU":
            return self.diffOutReLU(fz)
    
    def sigmod(self,z):
        return 1/(1+np.exp(-z))
    
    #f'(z) = f(z)(1-f(z))
    def diffSigmod(self,z):
        return np.multiply(sigmod(z),(1-sigmod(z)))
    
    def diffOutSigmod(self,f):
        return np.multiply(f,(1-f))
    
    #ReLU
    #Rectified Linear Unit
    def reLU(self,z):
        rea = [e if e>=0 else 0 for e in z]
        return np.array(rea).reshape(z.shape[0],1)
    
    def diffReLU(self,z):
        red = [1 if e>=0 else 0 for e in z]
        return np.array(red).reshape(z.shape[0],1)
    
    def diffOutReLU(self,f):
        redo = [1 if e>=0 else 0 for e in f]
        return np.array(redo).reshape(f.shape[0],1)

