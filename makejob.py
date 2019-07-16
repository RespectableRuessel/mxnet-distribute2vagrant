import os, sys
import argparse
import json

parser = parser = argparse.ArgumentParser(description='Launch distributed mxnet models on vagrant')

parser.add_argument('-c', '--copy', type=str)
parser.add_argument('-p', '--pretty', action='store_true')
parser.add_argument('-n', '--name', required=True, type=str)
parser.add_argument('-S', '--script', required=True, type=str)
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-b', '--batch-size', type=int)
parser.add_argument('-s', '--sequence-length', type=int)

args, unknown = parser.parse_known_args()

layout = {
	'script': args.script,
	'dataset': args.dataset,
	'epochs': 0,
	'batch-size': 0,
	'sequence-length': 0
}

if args.copy:
	if args.copy == args.name:
		parser.error('--name must not match --copy.')

	copy = args.copy

	if not str.endswith(copy, '.json'):
		copy = '%s.json' % (copy)

	with open(copy) as f:
		layout = json.load(f)

if args.epochs:
	layout['epochs'] = args.epochs

if args.batch_size:
	layout['batch-size'] = args.batch_size

if args.sequence_length:
	layout['sequence-length'] = args.sequence_length

job = args.name

if not str.endswith(job, '.json'):
	job = '%s.json' % (job)

with open(job, 'w') as f:
	if args.pretty:
		f.write(json.dumps(layout, indent=4, sort_keys=True))
	else:
		f.write(json.dumps(layout))
