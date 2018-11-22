#coding:utf8
'''
实现感知机
训练y=wx+b
'''

class Perceptron(object):
    def __init__(self,input_num,activator):
        '''
        初始化，需要参数个数以及激活函数
        '''
        self.activator=activator

        self.weights=[0.0 for i in range(input_num)]
        self.bias = 0.0

    def predict(self,input_vec):
        '''
        return current perceptron's result
        '''
        #temp=map(lambda(a,b):a*b,zip(self.weights,input_vec))
        #print("weights ",self.weights,"vec ",input_vec)
        temp=[i[0]*i[1] for i in zip(self.weights,input_vec)]
        res=self.bias
        for i in temp:
            res+=i
        return self.activator(res)
    
    def update_weights(self,input_vec,output,label,rate):
        '''
        update weights
        '''
        delta=label-output
        #self.weights=map(lambda item:item[1]+rate*delta*item[0],
        #        zip(input_vec,self.weights))
        self.weights=[i[1]+rate*delta*i[0] for i in zip(input_vec,self.weights)]
        self.bias+=rate*delta

    def one_iteration(self,input_vecs,labels,rate):
        '''
        one iteration means train all input once
        '''
        for input_vec,label in zip(input_vecs,labels):
            output=self.predict(input_vec)
            self.update_weights(input_vec,output,label,rate)

    def train(self,input_vecs,labels,iter_num,rate):
        for i in range(iter_num):
            self.one_iteration(input_vecs,labels,rate)


def activator(arg):
    '''
    activator function
    '''
    #return 1 if arg > 0 else 0
    if arg > 0:
        return 1
    else :
        return 0


#if __name__=="__main__":
def test():
    input_vecs=[[1,1],[1,0],[0,0],[0,1]]
    labels=[1,0,0,0]
    one=Perceptron(2,activator)
    one.train(input_vecs,labels,100,0.1)

    print("predict 1 1 ",one.predict([1,1]),activator(one.predict([1,1])))
    print("predict 1 0 ",one.predict([1,0]),activator(one.predict([1,0])))
    print("predict 0 1 ",one.predict([0,1]),activator(one.predict([0,1])))
    print("predict 0 0 ",one.predict([0,0]),activator(one.predict([0,0])))
    print("weights ",one.weights,"bias ",one.bias)
if __name__ =="__main__":
    test()
