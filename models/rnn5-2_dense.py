import os, sys
import argparse
import math
import random
import numpy as np
import pandas as pd
import time

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader, ArrayDataset
from mxnet.gluon.nn import HybridSequential, Dense
from mxnet.initializer import Xavier
from mxnet.metric import RMSE

parser = parser = argparse.ArgumentParser(description='Launch distributed mxnet models on vagrant')

parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-e', '--epochs', required=True, type=int)
parser.add_argument('-b', '--batch-size', required=True, type=int)
parser.add_argument('-s', '--sequence-length', required=True, type=int)
parser.add_argument('-t', '--test-split', required=True, type=float)

args, unknown = parser.parse_known_args()

name = os.path.basename(sys.argv[0]).replace('.py', '')
dir = os.path.dirname(args.dataset)

# hyper parameters
epochs = args.epochs
batch_size = args.batch_size
sequence_length = args.sequence_length
test_size = args.test_split

# kvstore
use_kv = True
kv = None

if use_kv:
	kv = mx.kvstore.create('dist_sync')

# load dataframe
lines = sum(1 for line in open(args.dataset))

df_len = lines
df_skip = 0

if use_kv:
	df_len = lines // kv.num_workers
	df_skip = (df_len * kv.rank) - 1

df = pd.read_csv(args.dataset, header = None, skiprows=df_skip, nrows = df_len)

# extract targets
y =  df[df.columns[-1]].to_numpy()

# data preperation
data_x = df.drop(df.columns[-1], axis=1).to_numpy()
data_y = np.array(y)

def split_to_sequences(x, y, n_prev=10):
	docX, docY = [], []
	for i in range(len(x) - n_prev):
		docX.append(x[i:i + n_prev])
		docY.append(y[i + n_prev])

	return np.array(docX).astype('float32'), np.array(docY).astype('float32')

data_x, data_y = split_to_sequences(data_x, data_y, n_prev=sequence_length)
ntr = int(len(data_x) * (1 - test_size))

# dataloader
train_dataset = ArrayDataset(data_x[:ntr], data_y[:ntr])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ArrayDataset(data_x[ntr:], data_y[ntr:])
test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model
ctx = mx.cpu()

net = gluon.nn.Sequential()
with net.name_scope():
	net.add(gluon.rnn.RNN(5, 1, layout='NTC'))
	net.add(gluon.nn.Dense(2))

net.initialize(mx.init.Xavier(), ctx=ctx)

# metrics
train_metrics = mx.metric.CompositeEvalMetric()

train_metric_accuracy = mx.metric.Accuracy()
train_metrics.add(train_metric_accuracy)

train_metric_f1 = mx.metric.F1()
train_metrics.add(train_metric_f1)

train_metric_perplexity = mx.metric.Perplexity(ignore_label=None)
train_metrics.add(train_metric_perplexity)

test_metrics = mx.metric.CompositeEvalMetric()

test_metric_accuracy = mx.metric.Accuracy()
test_metrics.add(test_metric_accuracy)

test_metric_f1 = mx.metric.F1()
test_metrics.add(test_metric_f1)

test_metricc_perplexity = mx.metric.Perplexity(ignore_label=None)
test_metrics.add(test_metricc_perplexity)

# training
l_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam')

if use_kv:
	trainer = gluon.Trainer(net.collect_params(), 'adam', kvstore=kv, update_on_kvstore=True)

for epoch in range(epochs):

	train_metrics.reset()
	tic = time.time()

	for i, (x, y) in enumerate(train_dataloader):
		with autograd.record():
			x = x.as_in_context(ctx)
			y = y.as_in_context(ctx)

			output = net(x)
			loss = l_cross_entropy(output, y)
			loss.backward()

		trainer.step(x.shape[0])

		train_metrics.update(y, output)

	toc = time.time() - tic

	if use_kv:
		print('[%s:%s] training completed in %ss with a accuracy of %s.' % (kv.rank, epoch, toc, train_metric_accuracy.get()[1]))
	else:
		print('[%s] training completed in %ss with a accuracy of %s.' % (epoch, toc, train_metric_accuracy.get()[1]))

	test_metrics.reset()
	tic = time.time()

	for i, (x, y) in enumerate(test_dataloader):
		# move to GPU if needed
		x = x.as_in_context(ctx)
		y = y.as_in_context(ctx)

		output = net(x)
		test_metrics.update(y, output)

	toc = time.time() - tic

	if use_kv:
		print('[%s:%s] testing completed in %ss with a accuracy of %s.' % (kv.rank, epoch, toc, test_metric_accuracy.get()[1]))
	else:
		print('[%s] testing completed in %ss with a accuracy of %s.' % (epoch, toc, test_metric_accuracy.get()[1]))

# checkpoint
acc = test_metric_accuracy.get()[1]
checkpoint = os.path.join(dir, '%s_%d_acc_%.4f.params' % (name, epochs, acc))
net.save_parameters(checkpoint)
