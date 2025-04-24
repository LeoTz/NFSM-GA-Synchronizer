import math
import random
import time
import tracemalloc
import numpy as np
import argparse
import os
import json
import sys
from typing import List, Dict, Tuple, Set, Optional, Union, FrozenSet

# Assuming these are imported from external files
from Node import Node
from Nfsm import Nfsm

class NFSMGeneticAlgorithm:
    """
    A class implementing a genetic algorithm for NFSM synchronization.
    
    This algorithm attempts to find an input sequence that brings all states of 
    an NFSM to the same final state (synchronization).
    """
    
    def __init__(self, 
                 population_size: int = 60, 
                 max_iterations: int = 30, 
                 hill_climb_percent: float = 0.1, 
                 min_chrom_size: float = 0.5, 
                 max_chrom_size: float = 2.0,
                 timeout_seconds: float = None):
        """
        Initialize the genetic algorithm with configuration parameters.
        
        Args:
            population_size: Number of individuals in the population
            max_iterations: Maximum number of iterations without improvement
            hill_climb_percent: Percentage of population to apply hill climbing
            min_chrom_size: Minimum chromosome size multiplier (relative to number of states)
            max_chrom_size: Maximum chromosome size multiplier (relative to number of states)
            timeout_seconds: Optional timeout in seconds
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.hill_climb_percent = hill_climb_percent
        self.min_chrom_size = min_chrom_size
        self.max_chrom_size = max_chrom_size
        self.timeout_seconds = timeout_seconds
        self.next_state_cache: Dict[Tuple[FrozenSet[int], int], Set[int]] = {}
        
    @staticmethod
    def read_nfsms_from_file(textFile: str) -> List[Nfsm]:
        """
        Read FSMs from a text file and return a list of NFSM objects.
        
        Each FSM block in the file follows a structured format:
        - Header line with FSM ID, number of states, inputs, outputs, etc.
        - Transition list (current_state next_state input_symbol)
        - Initial state
        
        Args:
            textFile: Path to the file containing NFSM definitions
            
        Returns:
            A list of Nfsm objects
        """
        nfsm_list = []

        with open(textFile, 'r') as file:
            fsm_header = file.readline()

            while fsm_header:
                fsm_header_parts = fsm_header.split()
                fsm_id = fsm_header_parts[0]  # Extract FSM ID as a string
                numOfStates, numInputs, numOutputs, saturation, numTransitions, L = map(int, fsm_header_parts[1:])

                nfsm_counter = Nfsm(fsm_id, numOfStates, numTransitions, numInputs, numOutputs, saturation)

                # Read transitions
                for i in range(numTransitions):
                    line = file.readline()
                    parts = line.split()
                    state, next_state = map(int, parts[:2])
                    input_val = int(parts[2])
                    nfsm_counter.setStateFunction(state, input_val, next_state)

                nfsm_counter.setInitialState(int(file.readline()))
                nfsm_list.append(nfsm_counter)
                
                # Read the next fsm header
                fsm_header = file.readline()

        return nfsm_list
    
    def precompute_valid_inputs(self, nfsm: Nfsm) -> Dict[int, Set[int]]:
        """
        Precompute and return valid input transitions for every state in the NFSM.
        
        Returns:
            A dictionary mapping state indices to sets of valid input indices
        """
        valid_inputs = {state: set() for state in range(nfsm.getNoStates())}

        for state in range(nfsm.getNoStates()):
            for input_index in range(nfsm.getNoInputs()):
                next_states = nfsm.returnNextStateValue(state, input_index)

                if next_states != -1 and next_states:
                    valid_inputs[state].add(input_index)

        return valid_inputs
    
    def initialize_population(self, nfsm: Nfsm) -> List[List[int]]:
        """
        Initialize a population of random valid input sequences for the NFSM.
        
        Returns:
            A list of chromosomes (input sequences)
        """
        valid_inputs = self.precompute_valid_inputs(nfsm)
        population = []
        max_attempts = 2 * self.population_size
        max_at = 0
        numStates = nfsm.getNoStates()

        while len(population) < self.population_size and max_at < max_attempts:
            sequence = []
            current_states = set(range(nfsm.getNoStates()))
            low = int(self.min_chrom_size * numStates)
            high = int(self.max_chrom_size * numStates)

            # Generate a valid sequence
            for _ in range(random.randint(low, high)):
                valid_choices = set.intersection(
                    *[valid_inputs[state] for state in current_states]
                )

                if valid_choices:
                    input_index = random.choice(list(valid_choices))
                    sequence.append(input_index)

                    # Determine next states
                    next_states = set()
                    for state in current_states:
                        next_state_result = nfsm.returnNextStateValue(state, input_index)
                        if next_state_result != -1:
                            next_states.update(next_state_result)

                    if not next_states:
                        break
                    current_states = next_states
                else:
                    break

            if sequence and current_states:
                population.append(sequence)

            max_at += 1
        
        return population
    
    def fitness_function(self, chromosome: List[int], nfsm: Nfsm) -> float:
        """
        Calculate fitness value for a chromosome.
        
        Higher fitness indicates better synchronization with fewer inputs.
        
        Args:
            chromosome: A sequence of input symbols
            nfsm: The NFSM to synchronize
            
        Returns:
            Fitness value (higher is better)
        """
        current_node = Node(set(range(nfsm.getNoStates())))

        for input_index in chromosome:
            key = (frozenset(current_node.getSet()), input_index)

            # Use cached transitions if available
            if key in self.next_state_cache:
                next_states = self.next_state_cache[key]
            else:
                next_states = set()
                for state in current_node.getSet():
                    possible_next_states = nfsm.returnNextStateValue(state, input_index)

                    # If any transition is invalid, return fitness = 0
                    if possible_next_states == -1:
                        return 0

                    next_states.update(possible_next_states)

                # Save to cache
                self.next_state_cache[key] = next_states

            current_node = Node(next_states)

        unique_states = set(current_node.getSet())

        # Fitness favors fewer final states and shorter sequences
        fitness_value = (10**5 / (len(unique_states) + 1)) * (1 / math.log2(len(chromosome) + 4))

        return fitness_value
    
    def proportional_selection(self, population: List[List[int]], fitness_scores: List[float]) -> Tuple[List[int], List[int]]:
        """
        Select two parents using proportional (roulette wheel) selection.
        
        Args:
            population: List of chromosomes
            fitness_scores: Corresponding fitness values
            
        Returns:
            Two selected parent chromosomes
        """
        parent_a_idx, parent_b_idx = np.random.choice(
            a=np.arange(len(population)),
            size=2,
            p=fitness_scores / np.sum(fitness_scores),
            replace=False
        )

        return population[parent_a_idx], population[parent_b_idx]
    
    def trim_sequence(self, chromosome: List[int], nfsm: Nfsm) -> List[int]:
        """
        Trim a sequence at the first invalid input.
        
        Args:
            chromosome: The input sequence to trim
            nfsm: The NFSM
            
        Returns:
            A valid prefix of the input sequence
        """
        current_node = Node(set(range(nfsm.getNoStates())))
        trimmed_chromosome = []

        for input_index in chromosome:
            next_states = set()

            for state in current_node.getSet():
                possible_next_states = nfsm.returnNextStateValue(state, input_index)

                if possible_next_states == -1 or not possible_next_states:
                    return trimmed_chromosome

                next_states.update(possible_next_states)

            trimmed_chromosome.append(input_index)

            if not next_states:
                break

            current_node = Node(next_states)

        return trimmed_chromosome
    
    def crossover(self, parent_a: List[int], parent_b: List[int], nfsm: Nfsm) -> Tuple[List[int], List[int]]:
        """
        Perform single-point crossover between two parents.
        
        Args:
            parent_a: First parent chromosome
            parent_b: Second parent chromosome
            nfsm: The NFSM
            
        Returns:
            Two child chromosomes
        """
        min_len = min(len(parent_a), len(parent_b))

        if min_len < 2:
            return parent_a, parent_b

        crossover_point = np.random.randint(min_len // 2, min_len)

        child_1 = parent_a[:crossover_point] + parent_b[crossover_point:]
        child_2 = parent_b[:crossover_point] + parent_a[crossover_point:]

        if self.fitness_function(child_1, nfsm) == 0:
            child_1 = self.trim_sequence(child_1, nfsm)
        if self.fitness_function(child_2, nfsm) == 0:
            child_2 = self.trim_sequence(child_2, nfsm)

        return child_1, child_2
    
    def hill_climber(self, nfsm: Nfsm, population: List[List[int]], fitness_values: List[float]) -> Tuple[List[List[int]], List[float]]:
        """
        Apply hill climbing to the top percentage of the population.
        
        Args:
            nfsm: The NFSM
            population: List of chromosomes
            fitness_values: Corresponding fitness values
            
        Returns:
            Updated population and fitness values
        """
        combined = list(zip(population, fitness_values))
        combined.sort(key=lambda x: x[1], reverse=True)

        num_to_select = max(1, int(len(combined) * self.hill_climb_percent))
        top_indices = range(num_to_select)

        for i in top_indices:
            individual, _ = combined[i]
            new_individual = self.modify_individual(individual, nfsm)
            new_fitness = self.fitness_function(new_individual, nfsm)
            combined[i] = (new_individual, new_fitness)

        new_population, new_fitness_values = zip(*combined)
        return list(new_population), list(new_fitness_values)
    
    def modify_individual(self, individual: List[int], nfsm: Nfsm) -> List[int]:
        """
        Modify an individual by adding inputs that reduce the state set.
        
        Args:
            individual: Chromosome to modify
            nfsm: The NFSM
            
        Returns:
            Modified chromosome
        """
        current_node = Node(set(range(nfsm.getNoStates())))

        # Replay the original individual to get the current state set
        for input_index in individual:
            key = (frozenset(current_node.getSet()), input_index)
            if key in self.next_state_cache:
                next_states = self.next_state_cache[key]
            else:
                next_states = set()
                for state in current_node.getSet():
                    possible_next_states = nfsm.returnNextStateValue(state, input_index)
                    if possible_next_states == -1:
                        return individual
                    next_states.update(possible_next_states)
                self.next_state_cache[key] = next_states
            current_node = Node(next_states)

        final_states = set(current_node.getSet())
        new_individual = individual.copy()

        while True:
            best_input = None
            best_next_states = None
            best_reduction = len(final_states)

            # Try each input to see if any reduce the state set
            for input_index in range(nfsm.getNoInputs()):
                key = (frozenset(final_states), input_index)
                if key in self.next_state_cache:
                    next_states = self.next_state_cache[key]
                else:
                    next_states = set()
                    valid = True
                    for state in final_states:
                        possible_next_states = nfsm.returnNextStateValue(state, input_index)
                        if possible_next_states == -1:
                            valid = False
                            break
                        next_states.update(possible_next_states)
                    if not valid:
                        continue
                    self.next_state_cache[key] = next_states

                if len(next_states) < best_reduction:
                    best_reduction = len(next_states)
                    best_input = input_index
                    best_next_states = next_states

            if best_input is None:
                break

            # Apply the best input
            new_individual.append(best_input)
            final_states = best_next_states

        return new_individual
    
    def verify_solution(self, nfsm: Nfsm, chromosome: List[int]) -> bool:
        """
        Verify if a chromosome is a valid synchronizing sequence.
        
        Args:
            nfsm: The NFSM
            chromosome: The sequence to verify
            
        Returns:
            True if the sequence synchronizes the NFSM, False otherwise
        """
        current_node = Node(set(range(nfsm.getNoStates())))

        for input_index in chromosome:
            key = (frozenset(current_node.getSet()), input_index)

            if key in self.next_state_cache:
                next_states = self.next_state_cache[key]
            else:
                next_states = set()
                for state in current_node.getSet():
                    possible_next_states = nfsm.returnNextStateValue(state, input_index)

                    if possible_next_states == -1:
                        return False

                    next_states.update(possible_next_states)

                self.next_state_cache[key] = next_states

            current_node = Node(next_states)

        # A solution is valid if we end up in a single unique state
        return len(current_node.getSet()) == 1
    
    def run(self, nfsm: Nfsm) -> Tuple[Optional[List[int]], Optional[int], Optional[float], float]:
        """
        Run the genetic algorithm to find a synchronizing sequence.
        
        Args:
            nfsm: The NFSM to synchronize
            
        Returns:
            Tuple of (best solution, solution length, elapsed time, peak memory usage in MB)
        """
        # Start timing and memory tracking
        start_time = time.time()
        tracemalloc.start()
        
        # Reset the cache
        self.next_state_cache = {}

        # Initialize the population
        population = self.initialize_population(nfsm)
        if not population:
            return None, None, None, 0.0

        # Evaluate initial fitness
        fitness_values = [self.fitness_function(chromosome, nfsm) for chromosome in population]

        # Track best solution
        best_index = np.argmax(fitness_values)
        best_solution = population[best_index]
        best_fitness = fitness_values[best_index]

        # Counter for iterations without improvement
        count = 0

        # Main evolutionary loop
        while count < self.max_iterations:
            # Check for timeout
            if self.timeout_seconds is not None and (time.time() - start_time) > self.timeout_seconds:
                break
                
            new_population = []
            new_fitness_values = []

            # Create new individuals through selection and crossover
            for _ in range(self.population_size // 2):
                parent_a, parent_b = self.proportional_selection(population, fitness_values)
                child_1, child_2 = self.crossover(parent_a, parent_b, nfsm)
                
                new_population.append(child_1)
                new_population.append(child_2)
                
                new_fitness_values.append(self.fitness_function(child_1, nfsm))
                new_fitness_values.append(self.fitness_function(child_2, nfsm))

            # Apply hill climbing
            population, fitness_values = self.hill_climber(nfsm, new_population, new_fitness_values)

            # Update best solution if improved
            max_index = np.argmax(fitness_values)
            if fitness_values[max_index] > best_fitness:
                best_solution = population[max_index]
                best_fitness = fitness_values[max_index]
                count = 0
            else:
                count += 1

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)  # Convert bytes to megabytes

        # Verify and return the best solution
        if self.verify_solution(nfsm, best_solution):
            return best_solution, len(best_solution), elapsed_time, peak_mb
        else:
            return None, None, None, 0.0

import argparse
import os
import json
import sys
import datetime

# Example usage
if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Run Genetic Algorithm for NFSM Synchronization')
    parser.add_argument('input_file', type=str, help='Path to the NFSM file')
    parser.add_argument('-o', '--output_file', type=str, default='results.json', 
                        help='Path to the output file (default: results.json)')
    parser.add_argument('-p', '--population_size', type=int, default=60,
                        help='Population size (default: 60)')
    parser.add_argument('-i', '--max_iterations', type=int, default=30,
                        help='Maximum number of iterations without improvement (default: 30)')
    parser.add_argument('-c', '--hill_climb_percent', type=float, default=0.1,
                        help='Percentage of population to apply hill climbing (0 to 1, default: 0.1)')
    parser.add_argument('--min_chrom_size', type=float, default=0.5,
                        help='Minimum chromosome size multiplier (0 to 1, default: 0.5)')
    parser.add_argument('--max_chrom_size', type=float, default=2.0,
                        help='Maximum chromosome size multiplier (>= min_chrom_size, default: 2.0)')
    parser.add_argument('-t', '--timeout', type=int, default=60,
                        help='Timeout in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # Validate the input file
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    # Validate percentage values
    if not 0 <= args.hill_climb_percent <= 1:
        print(f"Error: hill_climb_percent must be between 0 and 1, got {args.hill_climb_percent}")
        sys.exit(1)
        
    if not 0 <= args.min_chrom_size <= 1:
        print(f"Error: min_chrom_size must be between 0 and 1, got {args.min_chrom_size}")
        sys.exit(1)
        
    if args.max_chrom_size < args.min_chrom_size:
        print(f"Error: max_chrom_size ({args.max_chrom_size}) must be greater than or equal to min_chrom_size ({args.min_chrom_size})")
        sys.exit(1)
    
    try:
        # Load NFSMs from file
        nfsms = NFSMGeneticAlgorithm.read_nfsms_from_file(args.input_file)
        
        if not nfsms:
            print("No NFSMs found in the input file.")
            sys.exit(1)
            
        # Create algorithm instance with parameters from command line
        ga = NFSMGeneticAlgorithm(
            population_size=args.population_size,
            max_iterations=args.max_iterations,
            hill_climb_percent=args.hill_climb_percent,
            min_chrom_size=args.min_chrom_size,
            max_chrom_size=args.max_chrom_size,
            timeout_seconds=args.timeout
        )
        
        # Initialize results dictionary
        results = {
            "execution_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": args.input_file,
            "algorithm_parameters": {
                "population_size": args.population_size,
                "max_iterations": args.max_iterations,
                "hill_climb_percent": args.hill_climb_percent,
                "min_chrom_size": args.min_chrom_size,
                "max_chrom_size": args.max_chrom_size,
                "timeout_seconds": args.timeout
            },
            "machines": []
        }
        
        # Process each NFSM
        for idx, nfsm in enumerate(nfsms):
            nfsm_id = nfsm.getId()
            print(f"Processing NFSM {idx+1}/{len(nfsms)}: {nfsm_id}")
            
            solution, length, elapsed_time, memory = ga.run(nfsm)
            
            # Gather machine attributes
            machine_attributes = {
                "id": nfsm_id,
                "num_states": nfsm.getNoStates(),
                "num_inputs": nfsm.getNoInputs(),
                "num_outputs": nfsm.getNoOutputs(),
                "saturation": nfsm.getSaturation() if hasattr(nfsm, "getSaturation") else None,
                "num_transitions": nfsm.getNoTransitions(),
                "initial_state": nfsm.getInitialState()
            }
            
            # Store results
            machine_result = {
                "machine": nfsm_id,
                "machine_attributes": machine_attributes,
                "execution_time_seconds": elapsed_time if elapsed_time is not None else None,
                "memory_usage_mb": memory if memory is not None else 0.0,
                "success": solution is not None
            }
            
            if solution:
                machine_result["rs_length"] = length
                machine_result["resetting_sequence"] = solution
                print(f"Found resetting sequence of length {length} in {elapsed_time:.2f} seconds")
                print(f"Sequence: {solution}")
                print(f"Peak memory usage: {memory:.2f} MB")
            else:
                machine_result["rs_length"] = None
                machine_result["resetting_sequence"] = None
                print("No resetting sequence found")
            
            results["machines"].append(machine_result)
            print("-" * 50)
        
        # Save results to output file
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {args.output_file}")
        
        # Also generate a summary to the console
        print("\nSUMMARY:")
        print("-" * 50)
        print(f"{'Machine':<10} {'States':<8} {'Inputs':<8} {'RS Length':<10} {'Time (s)':<10} {'Memory (MB)':<12}")
        print("-" * 50)
        
        for machine in results["machines"]:
            attrs = machine["machine_attributes"]
            rs_length = machine["rs_length"] if machine["rs_length"] is not None else "N/A"
            time_val = f"{machine['execution_time_seconds']:.2f}" if machine['execution_time_seconds'] is not None else "N/A"
            print(f"{machine['machine']:<10} {attrs['num_states']:<8} {attrs['num_inputs']:<8} {rs_length:<10} {time_val:<10} {machine['memory_usage_mb']:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)