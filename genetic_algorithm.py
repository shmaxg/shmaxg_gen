import numpy
import concurrent.futures
from typing import Callable, Tuple, Optional

def fitness_function_wrapper(fitness_function, x, proc_id):
    return fitness_function(x), proc_id

def initialize_solutions(num_solutions: int, solution_dimension: int) -> numpy.ndarray:
    return 2 * numpy.random.random((num_solutions, solution_dimension)).astype('float32') - 1

def evaluate_solutions(fitness_function: Callable, solutions: numpy.ndarray) -> numpy.ndarray:

    nsolutions = len(solutions)

    rewards = numpy.zeros((nsolutions,))

    for i in range(nsolutions):
        rewards[i] = fitness_function(solutions[i].copy())

    return rewards

def evaluate_solutions_parallel(fitness_function: Callable, solutions: numpy.ndarray, max_workers: int, process_or_thread: str) -> numpy.ndarray:

    results, exec_list = [], []

    nsolutions = len(solutions)

    rewards = numpy.zeros((nsolutions,))

    if process_or_thread == 'process':

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

            for i in range(nsolutions):
                exec_list.append(executor.submit(fitness_function_wrapper, fitness_function, solutions[i].copy(), i))

            for j, f in enumerate(concurrent.futures.as_completed(exec_list)):
                results.append(f.result())

    elif process_or_thread == 'thread':

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            for i in range(nsolutions):
                exec_list.append(executor.submit(fitness_function_wrapper, fitness_function, solutions[i].copy(), i))

            for j, f in enumerate(concurrent.futures.as_completed(exec_list)):
                results.append(f.result())

    else:

        raise Exception('processORthread should be "process" or "thread".')

    for result in results:
        rewards[result[1]] = result[0]

    return rewards

def return_children(solutions: numpy.ndarray, sorted_parent_indexes: numpy.ndarray, mutation_parameter: float, mutation_decay_parameter: float, no_generation: int, num_solutions: int, solution_dimension: int) -> numpy.ndarray:

    children_solutions = numpy.zeros((num_solutions, solution_dimension))
    for i in range(num_solutions):
        selected_solution_index = sorted_parent_indexes[numpy.random.randint(len(sorted_parent_indexes))]
        children_solutions[i] = mutate(solutions[selected_solution_index].copy(), mutation_parameter, mutation_decay_parameter, no_generation)

    return children_solutions

def selection1(solutions: numpy.ndarray, children_solutions: numpy.ndarray, rewards: numpy.ndarray, child_rewards: numpy.ndarray, num_solutions: int) -> Tuple[numpy.ndarray, numpy.ndarray]:

    all_solutions = numpy.concatenate((
        solutions, children_solutions
    ), axis=0)

    all_rewards = numpy.concatenate((
        rewards, child_rewards
    ), axis=0)

    sorted_indexes = numpy.argsort(all_rewards)[::-1][:num_solutions]

    new_solutions = all_solutions[sorted_indexes]
    new_rewards = all_rewards[sorted_indexes]

    return new_solutions, new_rewards

def selection2(solutions: numpy.ndarray, children_solutions: numpy.ndarray, rewards: numpy.ndarray, child_rewards: numpy.ndarray, num_solutions: int) -> Tuple[numpy.ndarray, numpy.ndarray]:

    sorted_parent_indexes = numpy.argsort(rewards)[::-1]
    sorted_child_indexes = numpy.argsort(child_rewards)[::-1]

    new_solutions = numpy.concatenate((
        solutions[sorted_parent_indexes[0]].reshape((1, -1)),
        children_solutions[sorted_child_indexes[:num_solutions-1]]
    ), axis=0)

    new_rewards = numpy.concatenate((
        rewards[sorted_parent_indexes[0]].reshape((1,)),
        child_rewards[sorted_child_indexes[:num_solutions-1]]
    ), axis=0)

    return new_solutions, new_rewards

def mutate(solution: numpy.ndarray, mutation_parameter: float, mutation_decay_parameter: float, no_generation: int) -> numpy.ndarray:

    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        mu = mutation_parameter * (2 * numpy.random.random() - 1) * mutation_decay_parameter ** no_generation
        mutated_solution[i] += mu * abs(mutated_solution[i])

    return mutated_solution

def print_best_value(generation: int, rewards: numpy.ndarray, sorted_parent_indexes: numpy.ndarray) -> None:
    print(f"Generation {generation}: best fitness {rewards[sorted_parent_indexes[0]]}")

def save_best(solution: numpy.ndarray) -> None:
    numpy.save('genetic_classic_model_parameters', solution)

def optimize(fitness_function: Callable,
             num_generations: int,
             num_parents_mating: int,
             num_solutions: int,
             solution_dimension: int,
             mutation_parameter: float,
             mutation_decay_parameter: float,
             initial_generation: Optional[numpy.ndarray] = None,
             parallel: bool = False,
             process_or_thread: str = 'process',
             max_workers: Optional[int] = None) -> numpy.ndarray:

    if initial_generation is not None:
        if initial_generation.shape[0] != num_solutions:
            raise Exception('initial_generation.shape[0] should equals num_solutions.')
        if initial_generation.shape[1] != solution_dimension:
            raise Exception('initial_generation.shape[1] should equals solution_dimension.')
        solutions = initial_generation.copy()
    else:
        solutions = initialize_solutions(num_solutions, solution_dimension)

    if parallel:
        rewards = evaluate_solutions_parallel(fitness_function, solutions, max_workers, process_or_thread)
    else:
        rewards = evaluate_solutions(fitness_function, solutions)

    sorted_parent_indexes = numpy.argsort(rewards)[::-1][:num_parents_mating]

    for generation in range(num_generations):

        children_solutions = return_children(solutions, sorted_parent_indexes, mutation_parameter, mutation_decay_parameter, generation, num_solutions, solution_dimension)

        if parallel:
            child_rewards = evaluate_solutions_parallel(fitness_function, children_solutions, max_workers, process_or_thread)
        else:
            child_rewards = evaluate_solutions(fitness_function, children_solutions)

        solutions, rewards = selection2(solutions, children_solutions, rewards, child_rewards, num_solutions)

        sorted_parent_indexes = numpy.argsort(rewards)[::-1][:num_parents_mating]

        print_best_value(generation, rewards, sorted_parent_indexes)

        save_best(solutions[sorted_parent_indexes[0]])

    return solutions[sorted_parent_indexes[0]]
