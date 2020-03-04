
# Set up library imports.
import random
from collections import Counter
from itertools import chain
from bitstring import *

########################################################
'''
    - This module assumes you first read the Jupyter notebook. 
    - You are free to add other members functions in class GeneticAlgorithm
      as long as you do not modify the code already written. If you have justified
      reasons for making modifications in code, come talk to us. 
    - Our implementation uses recursive solutions and some flavor of 
      functional programming (maps/lambdas); You're not required to do so.
      Just Write clean code. 
'''
########################################################

class GeneticAlgorithm(object):

    def __init__(self, POPULATION_SIZE, CHROMOSOME_LENGTH, verbose):
        self.wall_bit_string_raw = "01010101011001101101010111011001100101010101100101010101"
        self.wall_bit_string = ConstBitStream(bin = self.wall_bit_string_raw)
        self.population_size = POPULATION_SIZE
        self.chromosome_length = CHROMOSOME_LENGTH # this is the length of self.wall_bit_string
        self.terminate = False
        self.verbose = verbose # In verbose mode, fitness of each individual is shown. 

    def run_genetic_alg(self):
        '''  
        The pseudo you saw in slides of Genetic Algorithm is implemented here. 
        Here, You'll get a flavor of functional 
        programming in Python- Those who attempted ungraded optional tasks in tutorial
        have seen something similar there as well. 
        Those with experience in functional programming (Haskell etc)
        should have no trouble understanding the code below. Otherwise, take our word that
        this is more or less similar to the generic pseudocode in Jupyter Notebook.

        '''
        "You may not make any changes to this function."

        # Creation of Population
        solutions = self.generate_candidate_sols(self.population_size) # arg passed for recursive implementation.

        # Evaluation of individuals
        parents = self.evaluate_candidates(solutions)

        while(not self.terminate):
            # Make pairs
            pairs_of_parents = self.select_parents(parents)

            # Recombination of pairs.
            recombinded_parents = list(chain(*map(lambda pair: \
                self.recombine_pairs_of_parents(pair[0], pair[1]), \
                    pairs_of_parents))) 

            # Mutation of each individual
            mutated_offspring = list(map(lambda offspring: \
                self.mutate_offspring(offspring), recombinded_parents))

            # Evaluation of individuals
            parents = self.evaluate_candidates(mutated_offspring) # new parents (offspring)
            if self.verbose and not self.terminate:
                self.print_fitness_of_each_indiviudal(parents)

######################################################################
###### These two functions print fitness of each individual ##########

# *** "Warning" ***: In this function, if an individual with 100% fitness is discovered, algorithm stops. 
# You should implement a stopping condition elsewhere. This codition, for example,
# won't stop your algorithm if mode is not verbose.
    def print_fitness_of_one_individual(self, _candidate_sol):
        _WallBitString = self.wall_bit_string
        _WallBitString.pos = 0
        _candidate_sol.pos = 0
        
        matching_bit_pairs = 0
        try:
            if not self.terminate:
                while (_WallBitString.read(2).bin == _candidate_sol.read(2).bin):
                    matching_bit_pairs = matching_bit_pairs + 1
                print('Individual Fitness: ', round((matching_bit_pairs)/28*100, 2), '%')
        except: # When all bits matched.
            return

    def print_fitness_of_each_indiviudal(self, parents):
        if parents:
            for _parent in parents:
                self.print_fitness_of_one_individual(_parent)

###### These two functions print fitness of each individual ##########
######################################################################

    def select_parents(self, parents):
        '''
        args: parents (list) => list of bitstrings (ConstbitStream)
        returns: pairs of parents (tuple) => consecutive pairs.
        '''

        # **** Start of Your Code **** #
        if len(parents) % 2 != 0: parents = parents[:-2]
        return list(map(lambda x,y: (x,y), parents[::2], parents[1::2]))
        # **** End of Your Code **** #


    # A helper function that you may find useful for `generate_candidate_sols()`
    def random_num(self):
        random.seed()
        return random.randrange(2**14) ## for fitting in 14 bits.

    def generate_candidate_sols(self, n): 
        '''
        args: n (int) => Number of cadidates solutions to generate. 
        retruns: (list of n random 56 bit ConstBitStreams) 
                 In other words, a list of individuals: Population.

        Each cadidates solution is a 56 bit string (ConstBitStreams object). 

        One clean way is to first get four 14 bit random strings then concatenate
        them to get the desired 56 bit candidate. Repeat this for n candidates.
        '''

        # **** Start of Your Code **** #
        population = []
        for _ in range(0, n):
            sample = ConstBitStream(uint = self.random_num(), length = 14)
            for _ in range(0, 3):
                sample += ConstBitStream(uint = self.random_num(), length = 14)
            population.append(sample)
        return population
        # **** End of Your Code **** # 

    def recombine_pairs_of_parents(self, p1, p2):
        """
        args: p1, and p2  (ConstBitStream)
        returns: p1, and p2 (ConstBitStream)

        split at .6-.9 of 56 bits (CHROMOSOME_LENGTH). i.e. between 31-50 bits
        """
        # **** Start of Your Code **** #
        split_at = int(random.randint(60,90) / 100 * self.chromosome_length)
        b1 = ConstBitStream(bin=p1.bin[:split_at] + p2.bin[split_at:])
        b2 = ConstBitStream(bin=p2.bin[:split_at] + p1.bin[split_at:])
        return   b1, b2
        # **** End of Your Code **** #

    def mutate_offspring(self, p):
        ''' 
            args: individual (ConstBitStream)
            returns: individual (ConstBitStream)
        '''
        # **** Start of Your Code **** #
        mutation_rate = 0.00765
        mutate = lambda x: "1" if x == "0" else "0"  
        return ConstBitStream(bin="".join([mutate(x) if random.random() < mutation_rate else x for x in p.bin]))        
        # **** End of Your Code **** #

    def num_matches(self, candidate):
        num_matches = 0
        self.wall_bit_string.pos = 0
        if candidate.bin == self.wall_bit_string_raw:
            print("Found Best Invidual:", candidate.bin)
            self.terminate = True
            return self.chromosome_length // 2
        while (candidate.read("bin:2") == self.wall_bit_string.read("bin:2")):
            num_matches = num_matches + 1
            if self.wall_bit_string.pos >= self.chromosome_length or candidate.pos >= self.chromosome_length:
                break
        candidate.pos = 0
        self.wall_bit_string.pos = 0
        return num_matches
    
    def evaluate_candidates(self, candidates): 
        '''
        args: candidate solutions (list) => each element is a bitstring (ConstBitStream)
        
        returns: parents (list of ConstBitStream) => each element is a bitstring (ConstBitStream) 
                    but elements are not unique. Fittest candidates will have multiple copies.
                    Size of 'parents' must be equal to population size.  
        '''
        # **** Start of Your Code **** #
        max_num_matches = self.chromosome_length // 2
        # find_match = lambda x1,x2,y1,y2: 1 if x1+x2==y1+y2 else 0
        # num_matches = lambda candidate: list(map(find_match, candidate.bin[::2], candidate.bin[1::2], self.wall_bit_string_raw[::2], self.wall_bit_string_raw[1::2])).count(1)
        
        fitness = [self.num_matches(candidate) / float(max_num_matches) for candidate in candidates]
        return random.choices(candidates, fitness, k=len(candidates))
        # **** End of Your Code **** # 