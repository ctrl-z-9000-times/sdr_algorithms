#!/usr/bin/python3
"""
Swarming meta-parameter search
Written by David McDougall, 2018

To use this module, structure experiments as follows:
    ExperimentModule is a python3 module containing the model to be optimized as
    well as code to evaluate model performance.

    ExperimentModule.default_parameters = {}
    This global dictionary contains all of the parameters to modify.
    Parameters must be one of the following types: dict, tuple, float, int.
    Parameters can be nested in multiple levels of dictionaries and tuples.

    ExperimentModule.main(parameters=default_parameters, argv=None, verbose=True)
    Returns (float) performance of parameters, to be maximized.  Debug is set to
    False when running particle swarm optimization.

Usage:
$ swarming.py [swarming arguments] ExperimentModule.py [experiment arguments]
"""

# TODO: Deal with global constants: particle_strength, global_strength, velocity_strength
#       Maybe make them into CLI Arguments?

particle_strength   =  .25
global_strength     =  .50
velocity_strength   =  .95
assert(velocity_strength + particle_strength / 2 + global_strength / 2 >= 1)

import argparse
import sys
import os
import random
import pprint
import time
import multiprocessing
import resource
import signal

def parameter_types(default_parameters):
    """
    Convert a set of parameters into the data types used to represent them.
    Returned result has the same structure as the parameters.
    """
    # Recurse through the parameter data structure.
    if isinstance(default_parameters, dict):
        return {key: parameter_types(value)
            for key, value in default_parameters.items()}
    elif isinstance(default_parameters, tuple):
        return tuple(parameter_types(value) for value in default_parameters)
    # Determine data type of each entry in parameter data structure.
    elif isinstance(default_parameters, float):
        return float
    elif isinstance(default_parameters, int):
        return int
    raise TypeError('Unaccepted type in swarming parameters: type "%s".'%(type(default_parameters).__name__))

def typecast_parameters(parameters, parameter_types):
    # Recurse through the parameter data structure.
    if isinstance(parameter_types, dict):
        return {key: typecast_parameters(parameters[key], parameter_types[key])
            for key in parameter_types.keys()}
    elif isinstance(parameter_types, tuple):
        return tuple(typecast_parameters(*args)
            for args in zip(parameters, parameter_types))
    # Type cast values.
    elif parameter_types == float:
        return float(parameters)
    elif parameter_types == int:
        return int(round(parameters))

def initial_parameters(default_parameters):
    # Recurse through the parameter data structure.
    if isinstance(default_parameters, dict):
        return {key: initial_parameters(value)
            for key, value in default_parameters.items()}
    elif isinstance(default_parameters, tuple):
        return tuple(initial_parameters(value) for value in default_parameters)
    # Calculate good initial values.
    elif isinstance(default_parameters, float):
        return default_parameters * 1.25 ** (random.random()*2-1)
    elif isinstance(default_parameters, int):
        if abs(default_parameters) < 10:
            return default_parameters + random.choice([-1, 0, +1])
        else:
            initial_value_float = initial_parameters(float(default_parameters))
            return int(round(initial_value_float))

def initial_velocity(default_parameters):
    # Recurse through the parameter data structure.
    if isinstance(default_parameters, dict):
        return {key: initial_velocity(value)
            for key, value in default_parameters.items()}
    elif isinstance(default_parameters, tuple):
        return tuple(initial_velocity(value) for value in default_parameters)
    # Calculate good initial velocities.
    elif isinstance(default_parameters, float):
        max_percent_change = 10
        uniform = 2 * random.random() - 1
        return default_parameters * uniform * (max_percent_change / 100.)
    elif isinstance(default_parameters, int):
        if abs(default_parameters) < 10:
            uniform = 2 * random.random() - 1
            return uniform
        else:
            return initial_velocity(float(default_parameters))

def initialize_particle_swarm(default_parameters, num_particles):
    swarm_data = {}
    for particle in range(num_particles):
        if particle in [0, 1, 2]:
            # Evaluate the default parameters a few times, before branching out
            # to the mre experimential stuff.  Several evals are needed since
            # these defaults will have their random velocity applied.
            value = default_parameters
        else:
            value = initial_parameters(default_parameters)
        swarm_data[particle] = {
            'value':      value,
            'velocity':   initial_velocity(default_parameters),
            'best':       value,
            'best_score': None,
        }
    swarm_data['best']       = random.choice(list(swarm_data.values()))['best']
    swarm_data['best_score'] = None
    swarm_data['evals']      = 0
    return swarm_data

def update_particle_position(position, velocity):
    # Recurse through the parameter data structure.
    if isinstance(position, dict):
        return {key: update_particle_position(value, velocity[key])
            for key, value in position.items()}
    elif isinstance(position, tuple):
        return tuple(update_particle_position(value, velocity[index])
            for index, value in enumerate(position))
    else:
        return position + velocity

def update_particle_velocity(postition, velocity, particle_best, global_best):
    # Recurse through the parameter data structure.
    if isinstance(postition, dict):
        return {key: update_particle_velocity(
                        postition[key], 
                        velocity[key], 
                        particle_best[key],
                        global_best[key])
            for key in postition.keys()}
    elif isinstance(postition, tuple):
        return tuple(update_particle_velocity(
                        postition[index], 
                        velocity[index], 
                        particle_best[index],
                        global_best[index])
            for index, value in enumerate(postition))
    else:
        # Update velocity.
        particle_bias = (particle_best - postition) * particle_strength * random.random()
        global_bias   = (global_best - postition)   * global_strength   * random.random()
        return velocity * velocity_strength + particle_bias + global_bias

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--best', action='store_true',
        help='Evaluate the best set of parameters on file, with verbose=True.')
    arg_parser.add_argument('-n', '--processes',  type=int, default=os.cpu_count(),)
    arg_parser.add_argument('-p', '--particles',  type=int, default=10,
        help='Size of swarm, number of particles to use.')
    arg_parser.add_argument('--time_limit',  type=float, default=None,
        help='Hours, time limit for parameter score evaluations.',)
    arg_parser.add_argument('--memory_limit',  type=float, default=None,
        help=('Gigabytes, RAM memory limit for parameter score evaluations.'
            'Default is (20 - 1.5   ) / N'),)
    arg_parser.add_argument('--clear_scores', action='store_true',
        help=('Remove all scores from the particle swarm so that the '
              'experiment can be safely altered.'))
    arg_parser.add_argument('experiment', nargs=argparse.REMAINDER,
        help='Name of experiment module followed by its command line arguments.')
    args = arg_parser.parse_args()
    assert(args.particles >= args.processes)
    if args.memory_limit is not None:
        memory_limit = args.memory_limit * 1e9
    else:
        memory_limit = int((20e9 - 1.5e9) / args.processes)
        print("Memory Limit %g GB per instance."%(memory_limit / 1e9))

    # Load the experiment module.
    experiment_file = args.experiment[0]
    experiment_path, experiment_module = os.path.split(experiment_file)
    experiment_module, dot_py = os.path.splitext(experiment_module)
    assert(dot_py == '.py')
    sys.path.append(experiment_path)
    exec('import %s; '%experiment_module)
    # Load the parameter structure & types and their default values.
    default_parameters = eval('%s.default_parameters'%experiment_module)
    parameter_structure = parameter_types(default_parameters)

    def evaluate_parameters(parameters, verbose):
        # This would have been a more useful function to have, in retrospect.
        1/0
        return parameters, score

    def timeout_callback(signum, frame):
        raise ValueError("Time limit exceded.")

    def evaluate_particle(particle_data):
        # Setup time limits
        if args.time_limit is not None:
            signal.signal(signal.SIGALRM, timeout_callback)
            time_limit = max(1, int(round(args.time_limit * 60 * 60)))
            signal.alarm(time_limit)
        # Setup memory limits
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))

        parameters = typecast_parameters(particle_data['value'], parameter_structure)
        eval_str = ('%s.main(parameters=%s, argv=[%s], verbose=False)'%(
                    experiment_module,
                    repr(parameters),
                    ', '.join("'%s'"%arg for arg in args.experiment[1:]),))
        score = eval(eval_str)
        if args.time_limit is not None:
            signal.alarm(0) # Disable time limit.
        return score

    # Setup the particle swarm.
    swarm_path = os.path.join(experiment_path, experiment_module) + '.pso'
    try:
        # Load an existing particle swarm.
        with open(swarm_path, 'r') as swarm_file:
            swarm_raw = swarm_file.read()
        try:
            swarm_data = eval(swarm_raw)
        except SyntaxError:
            while True:
                print("Corrupted particle swarm data file.  [B]ackup, [O]verwrite, or [EXIT]?")
                choice = input().upper()
                if choice == 'B':
                    backup_path = swarm_path + ".backup"
                    os.rename(swarm_path, backup_path)
                    print("BACKUP PATH: %s"%backup_path)
                    swarm_data = initialize_particle_swarm(default_parameters, args.particles)
                    break
                elif choice == 'O':
                    swarm_data = initialize_particle_swarm(default_parameters, args.particles)
                    break
                elif choice in 'EXIT':
                    print("EXIT")
                    sys.exit()
                else:
                    print('Invalid input "%s".'%choice)
    except FileNotFoundError:
        # Initialize a new particle swarm.
        swarm_data = initialize_particle_swarm(default_parameters, args.particles)

    if args.particles != sum(isinstance(key, int) for key in swarm_data):
        print("Warning: argument 'particles' does not match number of particles stored on file.")

    if args.best:
        print("Evaluating best parameters.")
        print("Score:", swarm_data['best_score'])
        pprint.pprint(swarm_data['best'])
        eval_str = ('%s.main(parameters=%s, argv=[%s], verbose=True)'%(
                    experiment_module,
                    repr(swarm_data['best']),
                    ', '.join("'%s'"%arg for arg in args.experiment[1:]),))
        score = eval(eval_str)
        print("Score:", score)
        sys.exit()

    if args.clear_scores:
        print("Removing Scores from Particle Swarm File %s."%swarm_path)
        swarm_data['best_score'] = None
        for entry in swarm_data:
            if isinstance(entry, int):
                swarm_data[entry]['best_score'] = None
        with open(swarm_path, 'w') as swarm_file:
            pprint.pprint(swarm_data, stream = swarm_file)
        sys.exit()

    # Run the particle swarm optimization.
    pool = multiprocessing.Pool(args.processes, maxtasksperchild=1)
    async_results = []
    while True:
        for particle_number in range(args.particles):
            particle_data = swarm_data[particle_number]

            # Update the particles velocity.
            particle_data['velocity'] = update_particle_velocity(
                particle_data['value'],
                particle_data['velocity'],
                particle_data['best'],
                swarm_data['best'],)

            # Update the particles postition.
            particle_data['value'] = update_particle_position(
                particle_data['value'],
                particle_data['velocity'])

            # Evaluate the particle.
            promise = pool.apply_async(evaluate_particle, (particle_data,))
            async_results.append((promise, particle_number))

            # Wait for a job to complete before continuing.
            while len(async_results) >= args.processes:
                # Check each job for completion or failure.
                for index, async_data in enumerate(async_results):
                    promise, particle_number = async_data
                    if promise.ready():
                        async_results.pop(index)
                        break
                else:
                    # No jobs done, wait and recheck.
                    time.sleep(10)
                    continue

                particle_data = swarm_data[particle_number]
                try:
                    score = promise.get()
                except (ValueError, MemoryError, ZeroDivisionError, AssertionError) as err:
                    print("")
                    print("Particle Number %d"%particle_number)
                    pprint.pprint(particle_data['value'])
                    print("%s:"%(type(err).__name__), err)
                    print("")
                    # Replace this particle.
                    particle_data['velocity'] = initial_velocity(default_parameters)
                    if particle_data['best_score'] is not None:
                        particle_data['value'] = particle_data['best']
                    elif swarm_data['best_score'] is not None:
                        particle_data['value'] = swarm_data['best']
                    else:
                        particle_data['value'] = initial_parameters(default_parameters)
                    continue
                except Exception:
                    print("")
                    pprint.pprint(particle_data['value'])
                    raise

                # Update best scoring particles.
                if particle_data['best_score'] is None or score > particle_data['best_score']:
                    particle_data['best']       = particle_data['value']
                    particle_data['best_score'] = score
                    print("New particle (%d) best score %g"%(particle_number, particle_data['best_score']))
                if swarm_data['best_score'] is None or score > swarm_data['best_score']:
                    swarm_data['best']       = typecast_parameters(particle_data['best'], parameter_structure)
                    swarm_data['best_score'] = particle_data['best_score']
                    swarm_data['best_particle'] = particle_number
                    print("New global best score %g"%swarm_data['best_score'])

                # Save the swarm to file.
                swarm_data['evals'] += 1
                with open(swarm_path, 'w') as swarm_file:
                    print('# ' + ' '.join(sys.argv), file=swarm_file)
                    pprint.pprint(swarm_data, stream = swarm_file)

        print("%d particles movements completed."%(swarm_data['evals']))
