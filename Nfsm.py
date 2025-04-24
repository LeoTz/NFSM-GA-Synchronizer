class Nfsm:

    def __init__(self, id: str, states: int, transitions: int, inputs: int, outputs: int, saturation: int):
        # Initialize the nfsm with its ID, number of states, transitions, and inputs
        self.id = id  # Unique identifier for the nfsm
        self.stateCount = states  # Number of states in the nfsm
        self.transitions = transitions  # Number of transitions in the nfsm
        self.inputCount = inputs  # Number of inputs in the nfsm
        self.outputCount = outputs  # Number of outputs in the nfsm
        self.saturation = saturation  # Saturation of the nfsm
        # Create a transition table to store the transitions for each state and input
        self.transition_table = [[[] for _ in range(inputs)] for _ in range(states)]
        #  print(self.transition_table)  # Print the initial transition table for debugging

    def setInitialState(self, initialState: int):
        # Set the initial state of the nfsm
        self.initialState = initialState

    def getInitialState(self):
        # Return the initial state of the nfsm
        return self.initialState

    def getNoInputs(self):
        # Return the number of inputs in the nfsm
        return self.inputCount

    def getNoTransitions(self):
        # Return the number of transitions in the nfsm
        return self.transitions

    def getNoStates(self):
        # Return the number of states in the nfsm
        return self.stateCount

    def getId(self):
        # Return the ID of the nfsm
        return self.id

    def setStateFunction(self, s: int, i: int, des: int):
        # Set the destination state for a given state 's' and input 'i'
        # Append the destination state to the transition table
        self.transition_table[s-1][i].append(des)

    def returnNextStateValue(self, s: int, i: int) -> list:
        # Return the list of possible next states for a given state 's' and input 'i'
        # If there is no destination state for the given state and action, return -1
        if len(self.transition_table[s-1][i]) == 0:
            return -1  # Indicate no possible next state with -1
        # Return the list containing all possible next states
        return self.transition_table[s-1][i]