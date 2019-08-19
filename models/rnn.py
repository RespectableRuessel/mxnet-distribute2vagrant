import os, sys
import time
import argparse
import logging

import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon.data import DataLoader, ArrayDataset

# --- cli args
parser = argparse.ArgumentParser(description='Train a distributed rnn model.')

parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-c', '--chunksize', type=int, default=10000)
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-s', '--sequence-length', type=int, default=12)
parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
parser.add_argument('-m', '--momentum', type=float, default=0.9)
parser.add_argument('-u', '--rnn-units', type=int, default=5)
parser.add_argument('-t', '--test-split', type=float, default=0.3)
parser.add_argument('-k', '--kvstore', type=str)
parser.add_argument('-S', '--statistics', action='store_true')

args, unknown = parser.parse_known_args()

# --- name
name = os.path.basename(sys.argv[0]).replace('.py', '')
dir = os.path.dirname(args.dataset)

# --- logging
log_interval = 24
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('%s.log' % (name))
logger = logging.getLogger()
logger.addHandler(fh)
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logging.debug('\n%s', '-' * 100)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)


# --- metrics
def get_metrics():
	metrics = mx.metric.CompositeEvalMetric()

	accuracy = mx.metric.Accuracy()
	metrics.add(accuracy)

	f1 = mx.metric.F1()
	metrics.add(f1)

	perplexity = mx.metric.Perplexity(ignore_label=None)
	metrics.add(perplexity)

	return metrics, accuracy, f1, perplexity

# --- checkpoint
def save_checkpoint(net, epoch, accuracy):
	fname = os.path.join(dir, '%s_acc_%.4f.params' % (name, accuracy))
	net.save_parameters(fname)
	logger.info('epoch %d\t\tsaving checkpoint with accuracy %.4f' % (epoch, accuracy))

# --- measurements
def track_statistics(statistics, epoch, batch, loss, metrics):
	metric, val = metrics.get()
	statistics.append([epoch, batch, nd.mean(loss).asscalar(), val[0], val[1], val[2]])

def save_statistics(statistics):
	fname = os.path.join(dir, '%s_metrics.csv' % (name))

	if not os.path.isfile(fname):
		with open(fname, 'w') as f:
			f.write('epoch,batch,loss,accuracy,f1,perplexity\n')

	with open(fname, 'a') as f:
		for stat in statistics:
			f.write('%s,%s,%s,%s,%s,%s\n' %
				(stat[0], stat[1], stat[2], stat[3], stat[4], stat[5]))

	statistics.clear()

# --- test
def test(ctx, test_data, metrics):
	metrics.reset()

	for i, (x, y) in enumerate(test_data):
		# move to GPU if needed
		x = x.as_in_context(ctx)
		y = y.as_in_context(ctx)

		pred = net(x)
		metrics.update(y, pred)

	return metrics.get()

# --- train
def train(ctx, net, train_data, test_data, metrics, epochs, batch_size, learning_rate, momentum, kv=None, track_stats=False):
	trainer = gluon.Trainer(net.collect_params(), 'sgd',
		optimizer_params={
			'learning_rate': learning_rate,
			'momentum': momentum,
			'multi_precision': True
		},
		kvstore=kv,
		update_on_kvstore=True if not kv is None else None)

	loss = gluon.loss.SoftmaxCrossEntropyLoss()

	statistics = []
	total_time = 0
	start_time = time.time()
	num_epochs = 0
	best_acc = 0

	for epoch in range(epochs):
		metrics.reset()
		tic = time.time()
		btic = time.time()

		for i, (x, y) in enumerate(train_data):
			with mx.autograd.record():
				x = x.as_in_context(ctx)
				y = y.as_in_context(ctx)

				output = net(x)
				L = loss(output, y)
				L.backward()

			trainer.step(x.shape[0])
			metrics.update(y, output)

			if log_interval and not (i+1)%log_interval:
				metric, val = metrics.get()
				logger.info('epoch %d batch %d\tspeed=%8.f samples/sec\t%s=%f, %s=%f' %
					(epoch, i, batch_size/(time.time() + 0.000001 - btic), metric[0], val[0], metric[1], val[1]))

			if track_stats and i > 0:
				track_statistics(statistics, epoch, i, L, metrics)

			btic = time.time()

		toc = time.time() - tic

		if num_epochs > 0:
			total_time = total_time + toc
		num_epochs = num_epochs + 1

		if track_stats:
			save_statistics(statistics)

		metric, val = metrics.get()
		logger.info('epoch %d\t\ttraining: %s=%f, %s=%f, %s=%f' %
			(epoch, metric[0], val[0], metric[1], val[1], metric[2], val[2]))
		logger.info('epoch %d\t\ttime cost: %fs' % (epoch, toc))

		metric, val = test(ctx, test_data, metrics)
		logger.info('epoch %d\t\tvalidation: %s=%f, %s=%f' %
			(epoch, metric[0], val[0], metric[1], val[1]))

		if val[0] > best_acc:
			best_acc = val[0]
			save_checkpoint(net, epoch, best_acc)

	if num_epochs > 1:
		logger.info('average epoch time: %fs' % (total_time / float(num_epochs - 1)))

	logger.info('training completed in %.3fs with a update time of %.3fs' % (time.time() - start_time, total_time))

# --- model
logger.info('starting new classification task:, %s', args)

ctx = mx.cpu()

net = gluon.nn.Sequential()
with net.name_scope():
	net.add(gluon.rnn.RNN(args.rnn_units, 1, layout='NTC', activation='tanh'))
	net.add(gluon.nn.Dense(2))

net.initialize(mx.init.Xavier(), ctx=ctx)

# --- kvstore
kv = None

if args.kvstore:
	kv = mx.kvstore.create(args.kvstore)

# --- dataframe
lines = sum(1 for line in open(args.dataset))
df_len = lines
df_skip = 0

if not kv is None:
	df_len = lines // kv.num_workers
	df_skip = (df_len * kv.rank) - 1

df_time = time.time()

logger.info('reading %s lines of data with a chunksize of %s' % (lines, args.chunksize))

tfr = pd.read_csv(args.dataset, header=None, chunksize=args.chunksize, iterator=True, skiprows=df_skip, nrows=df_len)
df = pd.concat(tfr)
#df = pd.read_csv(args.dataset, header=None, skiprows=df_skip, nrows=df_len)

logger.info('loaded data from %s starting at %s with %s rows in %.3fs' % (args.dataset, df_skip + 1, df_len, time.time() - df_time))

# --- data preperation
labels = df[df.columns[-1]].to_numpy()
data_x = df.drop(df.columns[-1], axis=1).to_numpy()
data_y = np.array(labels)

def split_to_sequences(x, y, n_prev=10):
	docX, docY = [], []
	for i in range(len(x) - n_prev):
		docX.append(x[i:i + n_prev])
		docY.append(y[i + n_prev])

	return np.array(docX).astype('float32'), np.array(docY).astype('float32')

data_x, data_y = split_to_sequences(data_x, data_y, n_prev=args.sequence_length)
ntr = int(len(data_x) * (1 - args.test_split))

# --- dataloader
train_dataset = ArrayDataset(data_x[:ntr], data_y[:ntr])
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, last_batch='discard', shuffle=True)

test_dataset = ArrayDataset(data_x[ntr:], data_y[ntr:])
test_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, last_batch='discard', shuffle=True)

logger.info('datasets: train=%d samples, validation=%d samples' % (len(train_dataset), len(test_dataset)))

# --- run
metrics, acc, f1, perp = get_metrics()
train(ctx, net, train_dataloader, test_dataloader, metrics,
	args.epochs, args.batch_size, args.learning_rate, args.momentum, kv, args.statistics and (kv is None or kv.rank == 0))
