#coding:utf8

from perceptron import Perceptron

def activator(x):
    return x

class LinearUnit(Perceptron):
    def __init__(self,input_num,activator):
        Perceptron.__init__(self,input_num,activator)

    def __str__(self):
        return "weights   :%s   bias   :%f "%(self.weights,self.bias)

if __name__=="__main__":
    input_vecs=[[1],[2],[3],[4],[5]]
    labels=[10,20,30,40,50]

    one=LinearUnit(1,activator)

    one.train(input_vecs,labels,10000,0.01)

    print(one)

    print(one.predict([3.5]))
    print(one.predict([6]))
    print(one.predict([0]))
