from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import glob
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
import csv


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def rangeMax(dataset, pos, nRange):
	for i in range(pos, pos+nRange):
		if(i==pos or vMax<float(dataset[i])):
			vMax=float(dataset[i]) 
	return vMax 

def rangeMin(dataset, pos, nRange):
	for i in range(pos, pos+nRange):
		if(i==pos or vMin>float(dataset[i])):
			vMin=float(dataset[i])
	return vMin 

def load_data(nRange=30):
    train_set_x=[]
    train_set_y=[]
    valid_set_x=[]
    valid_set_y=[]
    test_set_x=[]
    test_set_y=[]
    path=os.getcwd()+'/sourcedata/'
    os.chdir(path)
    for filename in glob.glob('*.csv'):
        data=[]
        data_set=[]
        onefile_set=[]
        ifile  = open(path+filename, "rt") 
        reader = csv.reader(ifile)
        for row in reader:
            data.append(row)
        for i in range(2, len(data)):
        	#sort data on date ascend and only keep 5 cols
        	#del data[len(data)-i+1][5]
        	#del data[len(data)-i+1][4]
        	#del data[len(data)-i+1][3]
        	del data[len(data)-i+1][0]
        	data_set.append(data[len(data)-i+1])

        #get change vs previous day 
        for i in range(0, len(data_set)-1):
            data_set[i][0]=float(data_set[i+1][0])-float(data_set[i][0])
            data_set[i][1]=float(data_set[i+1][1])-float(data_set[i][1])            
            data_set[i][2]=float(data_set[i+1][2])-float(data_set[i][2])            
            data_set[i][3]=float(data_set[i+1][3])-float(data_set[i][3])            
            data_set[i][4]=float(data_set[i+1][4])-float(data_set[i][4])            
        #get avg change vs previous day
        avgP=[]
        avgV=[]
        avga=[]
        avgb=[]
        avgc=[]
        for i in range(0, len(data_set)-1-(nRange-1)):
            for j in range(i,i+nRange):
                if(i==j):
                    avgP.append(abs(float(data_set[j][0])))
                    avgV.append(abs(float(data_set[j][1])))
                    avga.append(abs(float(data_set[j][2])))
                    avgb.append(abs(float(data_set[j][3])))
                    avgc.append(abs(float(data_set[j][4])))
                else:
                    avgP[i]+=abs(float(data_set[j][0]))
                    avgV[i]+=abs(float(data_set[j][1]))
                    avga[i]+=abs(float(data_set[j][2]))
                    avgb[i]+=abs(float(data_set[j][3]))
                    avgc[i]+=abs(float(data_set[j][4]))
            avgP[i]/=nRange
            avgV[i]/=nRange
        #set each day's relative variation
        for i in range(0, len(avgP)-(nRange-1)):
            data_set[i][0]=float(data_set[i][0])/avgP[i]
            data_set[i][1]=float(data_set[i][1])/avgV[i]
            data_set[i][2]=float(data_set[i][2])/avga[i]
            data_set[i][3]=float(data_set[i][3])/avgb[i]
            data_set[i][4]=float(data_set[i][4])/avgc[i]
            onefile_set.append(data_set[i])
        #set up data sets
        n=len(train_set_x)
        for i in range(0, len(onefile_set)-nRange-1):
						#set_X has nRange days of data each row
		        for j in range(i,i+nRange):
		            if(i==j):
		                train_set_x.append(onefile_set[j])
		            else:
		                train_set_x[n+i]+=onefile_set[j]
						#set_y has 1 data each row
		        if(float(onefile_set[i+nRange][0])>=0.3):
		            train_set_y.append(3)
		        elif(float(onefile_set[i+nRange][0])<=-0.3):
		            train_set_y.append(1)
		        else:
		            train_set_y.append(2)
		        if(i>=1800):
				        valid_set_x.append(train_set_x[n+i])
				        valid_set_y.append(train_set_y[n+i])
				        test_set_x.append(train_set_x[n+i])
				        test_set_y.append(train_set_y[n+i])
        print(len(train_set_x))

    train_set_x=numpy.array(train_set_x)
    train_set_y=numpy.array(train_set_y)
    valid_set_x=numpy.array(valid_set_x)
    valid_set_y=numpy.array(valid_set_y)
    test_set_x=numpy.array(test_set_x)
    test_set_y=numpy.array(test_set_y)
    train_set=(train_set_x, train_set_y)
    valid_set=(valid_set_x, valid_set_y)
    test_set=(test_set_x, test_set_y)

    print('... load data!') 
    print(len(train_set_x)) 
    print(len(valid_set_x)) 
    print(len(train_set_x[0])) 
    print(train_set_y[:30]) 

    def shared_dataset(data_xy, borrow=True):
      data_x, data_y = data_xy
      shared_x = theano.shared(numpy.asarray(data_x,
                                             dtype=theano.config.floatX),
                               borrow=borrow)
      shared_y = theano.shared(numpy.asarray(data_y,
                                             dtype=theano.config.floatX),
                               borrow=borrow)
      return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           batch_size=600):
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=30 * 5, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open(os.getcwd()+'/../best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def predict():
    # load the saved model
    classifier = pickle.load(open('best_model.pkl', 'rb'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    print ("... predict now!")
    predicted_values = predict_model(test_set_x[:30])
    print("Predicted values for selected examples in test set:")
    print(predicted_values)
    print (test_set_y.eval()[:30])
    print (train_set_y.eval()[1800:1830])
    match_rate(predict_model(test_set_x[:60]), test_set_y.eval()[:60])
    match_rate(predict_model(test_set_x[0:628]), test_set_y.eval()[0:628])
    match_rate(predict_model(test_set_x[629:1257]), test_set_y.eval()[629:1257])
    match_rate(predict_model(test_set_x[1258:1886]), test_set_y.eval()[1258:1886])
    match_rate(predict_model(test_set_x[1887:2515]), test_set_y.eval()[1887:2515])
    match_rate(predict_model(test_set_x[2516:3144]), test_set_y.eval()[2516:3144])
    match_rate(predict_model(test_set_x[3145:3773]), test_set_y.eval()[3145:3773])
    match_rate(predict_model(test_set_x[3774:4402]), test_set_y.eval()[3774:4402])
    match_rate(predict_model(test_set_x[4403:5031]), test_set_y.eval()[4403:5031])
    match_rate(predict_model(test_set_x[5032:5660]), test_set_y.eval()[5032:5660])
    match_rate(predict_model(test_set_x[5661:6289]), test_set_y.eval()[5661:6289])
def match_rate(answer, result):
		n=0
		for i in range(0, len(answer)):
			if(answer[i]==result[i]):
				n+=1
		print(n/len(answer)*100)

if __name__ == '__main__':
    sgd_optimization_mnist()
    predict()


