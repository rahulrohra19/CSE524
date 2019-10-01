import tensorflow as tf
import json
import sys
from tensorflow.python.client import timeline
from protobuf_to_dict import protobuf_to_dict
from google.protobuf.json_format import MessageToJson
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import graph_io
import os
import time


with tf.device('/gpu:0'):
    a = tf.random_normal([20000, 50000])
    b = tf.random_normal([50000, 10000])
    res = tf.matmul(a, b)

config_proto = tf.ConfigProto(graph_options=tf.GraphOptions(build_cost_model=1))
config_proto.intra_op_parallelism_threads = 1
config_proto.inter_op_parallelism_threads = 1
config_proto.graph_options.optimizer_options.opt_level = -1
config_proto.graph_options.rewrite_options.constant_folding = (rewriter_config_pb2.RewriterConfig.OFF)
config_proto.graph_options.rewrite_options.arithmetic_optimization = (rewriter_config_pb2.RewriterConfig.OFF)
config_proto.graph_options.rewrite_options.dependency_optimization = (rewriter_config_pb2.RewriterConfig.OFF)
config_proto.graph_options.rewrite_options.layout_optimizer = (rewriter_config_pb2.RewriterConfig.OFF)


sess = tf.Session(config=config_proto)
sess.run(tf.global_variables_initializer())

tot_time = 0
for i in range(10):
    print(i)
    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
    st = time.time()
    W_ = sess.run(res, options=run_options, run_metadata=run_metadata)

    if i != 0:
        jsonObj = MessageToJson(run_metadata)
        with open('logs/metadata/matmul_%d.json' % (i), 'w') as outfile:
            json.dump(jsonObj, outfile)

        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
	trace_file = open('logs/timeline/matmul_%d_.ctf.json' % (i), 'w')
        trace_file.write(trace.generate_chrome_trace_format())
        tot_time += time.time() -st
print('total time taken : ', tot_time)
