import numpy
from genetic_algorithm import optimize

# Function to MAXIMIZE
def objective(x):
    return -numpy.linalg.norm(x)**2


if __name__ == '__main__':

    sol = optimize(objective,
                   num_generations=1000,             # number of global iterations
                   num_parents_mating=10,            # how many of the best will be mutated
                   num_solutions=10,                 # the overall quanity of solutions
                   solution_dimension=10,            # dimension of the solution vector
                   mutation_parameter=0.1,           # x_i += |x_i| * mutation_parameter * (2U-1) * mutation_decay_parameter ** no_generetion
                   mutation_decay_parameter=0.999,
                   initial_generation=None,
                   parallel=True,
                   process_or_thread='process',      # if thread, be careful with shared memory when calculating objective
                   max_workers=8)                    # maximum number of processes or threads to execute objective

    print(sol)

