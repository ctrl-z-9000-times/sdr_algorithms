#!/usr/bin/python3
# Written by David McDougall, 2018
"""

The time has come to micro-manage the parameters. I want to start cleaning up
and fixing parameters which have some-what converged onto a value, like the
grid-cell spaceing has.  Then the PSO can work on the important stuff.

"""

import argparse
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from sdr import SDR
from grid_cell_encoder import GridCellEncoder
from spatial_pooler import StableSpatialPooler
from temporal_memory import TemporalMemory
from nupic.algorithms.sdr_classifier import SDRClassifier
from synapses import debug as synapses_debug


class Environment(object):
  """
  Environment is a 2D square, in first quadrant with corner at origin.
  """
  def __init__(self, size):
    self.size      = size
    self.speed     = 1.
    self.reset()

  def reset(self):
    quarter = self.size / 4
    self.position  = (
      quarter + 2 * quarter * random.random(),
      quarter + 2 * quarter * random.random(),)
    self.course    = []
    self.angle     = random.uniform(0, 2 * math.pi)
    self.collision = False

  def in_bounds(self, position):
    x, y = position
    x_in = x >= 0 and x < self.size
    y_in = y >= 0 and y < self.size
    return x_in and y_in

  def move(self, angle):
    self.angle = angle
    vx   = self.speed * np.cos(angle)
    vy   = self.speed * np.sin(angle)
    x, y = self.position
    new_position = (x + vx, y + vy)

    if not self.in_bounds(new_position):
      self.collision = True
      x, y = new_position
      x = max(0, min(self.size, x))
      y = max(0, min(self.size, y))
      new_position = (x, y)
    else:
      self.collision = False

    self.position = new_position
    self.course.append(self.position)

  def plot_course(self, show=True):
    plt.figure("Path")
    plt.ylim([0, self.size])
    plt.xlim([0, self.size])
    x, y = zip(*self.course)
    plt.plot(x, y, 'k-')
    if show:
      plt.show()


symbol_size = 10
patterns = [
  # 6 Hexagonal directions
  lambda angle, age: 0,
  lambda angle, age: 60,
  lambda angle, age: 120,
  lambda angle, age: 180,
  lambda angle, age: 240,
  lambda angle, age: 300,

  # Circles.  Note: this should read arctan(diameter / speed)
  lambda angle, age: angle + (180 - (180/math.pi) * 2 * math.atan(symbol_size / 1)),
  lambda angle, age: angle - (180 - (180/math.pi) * 2 * math.atan(symbol_size / 1)),
]
if False:
  patterns.extend([
    # Triangles
    lambda angle, age: angle + 120 if age % symbol_size == 0 else angle,
    lambda angle, age: angle - 120 if age % symbol_size == 0 else angle,

    # Squares
    lambda angle, age: angle + 90 if age % symbol_size == 0 else angle,
    lambda angle, age: angle - 90 if age % symbol_size == 0 else angle,

    # Pentagram
    lambda angle, age: angle + (180 - 36) if age % symbol_size == 0 else angle,
    lambda angle, age: angle - (180 - 36) if age % symbol_size == 0 else angle,

    # Hexagons
    lambda angle, age: angle + 60 if age % (symbol_size/2) == 0 else angle,
    lambda angle, age: angle - 60 if age % (symbol_size/2) == 0 else angle,
  ])

default_parameters = {
          'grid_cells': {'module_periods': (6.,
                                            8.5,
                                            12.,
                                            17.),
                         'n': 200,
                         'sparsity': 0.3},
          'motion': {'boosting_alpha': 0.0001,
                     'mini_columns': 1851,
                     'permanence_dec': 0.0006537325136937361,
                     'permanence_inc': 0.0035348050360959125,
                     'permanence_thresh': 0.20650900722951646,
                     'potential_pool': int(2400 * .95),
                     'segments': 2,
                     'init_dist': (0.15, 0.15),
                     'sparsity': 0.01689568144866546,
                     'stability_rate': 0.047058850549730885},
          'trajectory': {'add_synapses': 43,
                         'cells_per_column': 12,
                         'init_dist': (0.2719112306140346,
                                       0.17374506720195643),
                         'learning_threshold': 16,
                         'mispredict_dec': 0.0010616217883329827,
                         'permanence_dec': 0.0027755630357514647,
                         'permanence_inc': 0.011754088836345277,
                         'permanence_thresh': 0.5076141319852034,
                         'predictive_threshold': 20,
                         'synapses_per_segment': 92,}}

def main(parameters=default_parameters, argv=None, verbose=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('--episode_length', type=int, default   = 100,)
  parser.add_argument('--train_episodes', type=int, default   = 100 * len(patterns),)
  parser.add_argument('--test_episodes',  type=int, default   = 20 * len(patterns),)
  parser.add_argument('--environment_size', type=int, default = 40,)
  parser.add_argument('--move_env',     action='store_true')
  parser.add_argument('--show_pattern', action='store_true')
  args = parser.parse_args(args = argv)

  # PARAMETER OVERRIDES!
  parameters['grid_cells'] = default_parameters['grid_cells']

  if verbose:
    import pprint
    print("Parameters = ", end='')
    pprint.pprint(parameters)
    print("Episode Length", args.episode_length)

  env = Environment(size = args.environment_size)
  gc  = GridCellEncoder(**parameters['grid_cells'])

  trajectory = TemporalMemory(
      column_sdr           = gc.grid_cells,
      context_sdr          = None,
      anomaly_alpha        = 1./1000,
      predicted_boost      = 1,
      segments_per_cell    = 20,
      **parameters['trajectory'])
  trajectory_sdrc = SDRClassifier(steps=[0])

  motion = StableSpatialPooler(
      input_sdr           = SDR(trajectory.active),
      **parameters['motion'])
  motion_sdrc = SDRClassifier(steps=[0])

  def reset():
    env.reset()
    gc.reset()
    trajectory.reset()
    motion.reset()

  env_offset = np.zeros(2)
  def compute(learn=True):
    gc_sdr = gc.encode(env.position + env_offset)

    trajectory.compute(
      column_sdr  = gc_sdr,
      learn       = learn,)

    motion.compute(
      input_sdr           = trajectory.active,
      input_learning_sdr  = trajectory.learning,
      learn               = learn,)

  # Train
  if verbose:
    print("Training for %d episodes ..."%args.train_episodes)
    start_time = time.time()
  for session in range(args.train_episodes):
    reset()
    pattern = random.randrange(len(patterns))
    pattern_func = patterns[pattern]
    for step in range(args.episode_length):
      angle = pattern_func(env.angle * 180 / math.pi, motion.age) * math.pi / 180
      env.move(angle)
      if env.collision:
        reset()
        continue
      compute()
      trajectory_sdrc.compute(trajectory.age, trajectory.learning.flat_index,
          classification={"bucketIdx": pattern, "actValue": pattern},
          learn=True, infer=False)
      motion_sdrc.compute(motion.age, motion.columns.flat_index,
          classification={"bucketIdx": pattern, "actValue": pattern},
          learn=True, infer=False)
      if verbose and motion.age % 10000 == 0:
        print("Cycle %d"%motion.age)
    if args.show_pattern:
      env.plot_course()

  if verbose:
    train_time = time.time() - start_time
    start_time = time.time()
    print("Elapsed time (training): %d seconds."%int(round(train_time)))
    print("")
    print("Trajectory", trajectory.statistics())
    print("Motion", motion.statistics())
    print("")

  # Test
  if verbose:
    print("Testing for %d episodes ..."%args.test_episodes)
  if args.move_env:
    env_offset = np.array([9 * env.size, 9 * env.size])
    if verbose:
      print("Moved to new environment.")
  trajectory_accuracy  = 0
  motion_accuracy      = 0
  sample_size          = 0
  trajectory_confusion = np.zeros((len(patterns), len(patterns)))
  motion_confusion     = np.zeros((len(patterns), len(patterns)))
  for episode in range(args.test_episodes):
    reset()
    pattern = random.randrange(len(patterns))
    pattern_func = patterns[pattern]
    for step in range(args.episode_length):
      angle = pattern_func(env.angle * 180 / math.pi, motion.age) * math.pi / 180
      env.move(angle)
      if env.collision:
        reset()
        continue
      compute(learn=True)
      trajectory_inference = trajectory_sdrc.infer(trajectory.learning.flat_index, None)[0]
      if pattern == np.argmax(trajectory_inference):
        trajectory_accuracy += 1
      trajectory_confusion[pattern][np.argmax(trajectory_inference)] += 1
      motion_inference = motion_sdrc.infer(motion.columns.flat_index, None)[0]
      if pattern == np.argmax(motion_inference):
        motion_accuracy += 1
      motion_confusion[pattern][np.argmax(motion_inference)] += 1
      sample_size += 1
  trajectory_accuracy /= sample_size
  motion_accuracy     /= sample_size
  if verbose:
    print("Trajectory Accuracy %g, %d catagories."%(trajectory_accuracy, len(patterns)))
    print("Motion Accuracy     %g"%motion_accuracy)

  # Display Confusion Matixes
  if verbose:
    conf_matrices = (trajectory_confusion, motion_confusion,)
    conf_titles   = ('Trajectory', 'Motion',)
    #
    plt.figure("Pattern Recognition Confusion")
    for subplot_idx, matrix_title in enumerate(zip(conf_matrices, conf_titles)):
        matrix, title = matrix_title
        plt.subplot(1, len(conf_matrices), subplot_idx + 1)
        plt.title(title + " Confusion")
        matrix_sum = np.sum(matrix, axis=1)
        matrix_sum[matrix_sum == 0] = 1
        matrix = (matrix.T / matrix_sum).T
        plt.imshow(matrix, interpolation='nearest')
        plt.xlabel('Prediction')
        plt.ylabel('Label')

  if synapses_debug:
    gc.synapses.check_data_integrity()
    trajectory.synapses.check_data_integrity()
    motion.synapses.check_data_integrity()
    print("Synapse data structure integrity is OK.")

  if verbose:
    test_time = time.time() - start_time
    print("Elapsed time (testing): %d seconds."%int(round(test_time)))
    plt.show()
  return motion_accuracy

if __name__ == '__main__':
    main()
