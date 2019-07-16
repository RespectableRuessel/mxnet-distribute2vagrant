import os, sys
import argparse
import re
import json
import random
import time

def install():
	if not os.path.isdir('incubator-mxnet'):
		os.system('git clone --recursive https://github.com/apache/incubator-mxnet')

def deploy(hosts):
	no_keycheck = '-o "StrictHostKeyChecking no"'
	default_locale = 'export LC_ALL=C;'
	pipe_sudo = 'echo "vagrant" | sudo -S'

	with open(hosts) as f:
		for host in f:
			cmd = '%s apt-get -y install python3-pip' % (pipe_sudo)
			os.system('%s ssh %s "%s" "%s"' % (default_locale, no_keycheck, host, cmd))

			cmd = 'pip3 --upgrade'
			os.system('%s ssh %s "%s" "%s"' % (default_locale, no_keycheck, host, cmd))

			cmd = 'pip install mxnet pandas'
			os.system('%s ssh %s "%s" "%s"' % (default_locale, no_keycheck, host, cmd))

def launch(hosts, job, nodes):
	if not os.path.isfile('incubator-mxnet/tools/launch.py'):
		print('error: install launcher first.')
		exit(1)

	if not str.endswith(job, '.json'):
		job = '%s.json' % (job)

	if not os.path.isfile(job):
		print('error: job not found.')
		exit(1)

	launch_args = '-H %s --launcher ssh' % (hosts)
	launch_cmd = 'python3'

	if len(nodes) > 1:
		launch_args = '%s -n %s -s %s' % (launch_args, nodes[0], nodes[1])
	else:
		launch_args = '%s -n %s' % (launch_args, nodes[0])

	def make_argument(info, desc):
		return '--%s %s' % (info, desc[info])

	def parse_job(job):
		line = ''

		with open(job) as f:
			desc = json.load(f)
			line = '%s' % (desc.pop('script'))
			for key in desc:
				line = '%s %s' % (line, make_argument(key, desc))

		return line

	launch_cmd = '%s %s' % (launch_cmd, parse_job(job))

	#print('python3 incubator-mxnet/tools/launch.py %s %s' % (launch_args, launch_cmd))

	tic = time.time()

	os.system('python3 incubator-mxnet/tools/launch.py %s %s' % (launch_args, launch_cmd))

	toc = time.time() - tic

	print('distributed training completed in %ss.' % (toc))

def pick(hosts, count):
	all = []

	with open(hosts) as f:
		all = f.readlines()
		random.shuffle(all)
		all = all[0:count]

	with open(hosts, 'w') as f:
		for h in all:
			f.write('%s' % (h))

	return hosts

def ansible2hostfile(hosts):
	file = '%s%s' % (hosts, '_out')
	open(file, 'w').close()

	def is_commented(line, substring):
		comment = line.find('#')

		if comment is -1:
			return False

		return comment < line.find(substring)

	def parse(line):
		ip = re.findall(r'[0-9]+(?:\.[0-9]+){3}', line)

		if len(ip) < 1 or is_commented(line, ip[0]):
			return ''

		return ip[0]

	def write(line):
		if line is not '':
			out.write('%s\n' % (line))

	with open(file, 'a') as out:
		with open(hosts) as f:
			is_worker = True

			for line in f:
				if 'master' in line:
					is_worker = False

				if 'worker' in line:
					is_worker = True

				if is_worker:
					write(parse(line))
	return file

parser = parser = argparse.ArgumentParser(description='Launch distributed mxnet models on vagrant')

parser.add_argument('-i', '--install', action='store_true',
					help='installs all master machine requirements')
parser.add_argument('-H', '--hosts', type=str,
					help='the hostfile of slave machines')
parser.add_argument('-p', '--pick', type=int,
					help='picks randomly a certain number of hosts.')
parser.add_argument('-a', '--ansible', action='store_true',
					help='converts an ansible to a plain host file containing \
					only worker nodes')
parser.add_argument('-d', '--deploy', action='store_true',
					help='prepares a worker node with necessary software \
					and modules')
parser.add_argument('-l', '--launch', type=str,
					help='the job that gets executed via the mxnet \
					tools/launch.py script')
parser.add_argument('-n', '--launch-nodes', nargs='+', type=int,
					help='number of worker and (optional) server nodes to be launched.')

args, unknown = parser.parse_known_args()

if args.install:
	install()

if not args.hosts:
	exit(1)

hosts = args.hosts

if not os.path.isfile(hosts):
	print('error: hostfile not found.')
	exit(1)

if args.ansible and args.hosts is None:
	parser.error('--ansible requires --hosts.')

if args.ansible:
	hosts = ansible2hostfile(hosts)

if args.pick and args.hosts is None:
	parser.error('--pick requires --hosts.')

if args.pick:
	hosts = pick(hosts, args.pick)

if args.deploy and args.hosts is None:
	parser.error('--deploy requires --hosts.')

if args.deploy:
	deploy(hosts)

if args.launch and args.launch_nodes is None:
	parser.error('--launch requires --launch-nodes.')

if args.launch:
	launch(hosts, args.launch, args.launch_nodes)
