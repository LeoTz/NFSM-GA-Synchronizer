import sys
import math

class Node:

    def __init__(self, setOfStates: set = set()) -> None:
        self.stateSet = list(setOfStates)  # Initialize an empty list to store the states in the set
        # Used for iterator
        self.index = 0  # Initialize the iterator index

    def __len__(self):
        return len(self.stateSet)  # Return the number of states in the set

    def __contains__(self, item):
        return item in self.stateSet  # Check if the item is in the state set

    @staticmethod
    def exists(V, v) -> bool:
        # Check if the state 'v' exists in the list of states 'V'
        if len(V) == 0:
            return False

        for element in V:
            if element == v:
                return True

        return False

    def addSet(self, s):
        # Add a state 's' to the state set
        self.stateSet.append(s)

    def elemAt(self, a):
        # Return the state at the specified index 'a'
        return self.stateSet[a]

    def getSet(self):
        # Return the state set of the node
        return self.stateSet

    def __iter__(self):
        # Initialize the iterator for the state set
        self.index = 0
        return self

    def sort(self):
        # Sort the state set in ascending order using a simple sorting algorithm
        not_in_order = True  # Flag to check if the set is sorted
        curr = 0  # Current index for sorting

        while not_in_order:
            next_index = curr + 1  # Get the next index

            if next_index < len(self.stateSet):
                if self.stateSet[curr] > self.stateSet[next_index]:
                    # Swap the elements if they are in the wrong order
                    temp = self.stateSet[curr]
                    self.stateSet[curr] = self.stateSet[next_index]
                    self.stateSet[next_index] = temp
                    curr = 0  # Reset the index to check from the beginning
                else:
                    curr += 1  # Move to the next index
            else:
                not_in_order = False  # Exit the loop when the set is sorted