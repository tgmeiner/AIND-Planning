from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            # TODO create all load ground actions from the domain Load action

            loads = []
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        precond_pos = [
                            expr("At({}, {})".format(cargo, airport)),
                            expr("At({}, {})".format(plane, airport))
                        ]
                        precond_neg = []
                        effect_add = [expr("In({}, {})".format(cargo, plane))]
                        effect_rem = [expr("At({}, {})".format(cargo, airport))]
                        load = Action(
                            expr("Load({}, {}, {})".format(cargo, plane, airport)),
                            [precond_pos, precond_neg],
                            [effect_add, effect_rem]
                         )
                        loads.append(load)
            return loads


        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            # TODO create all Unload ground actions from the domain Unload action

            unloads = []

            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        precond_pos = [
                            expr("In({}, {})".format(cargo, plane)),
                            expr("At({}, {})".format(plane, airport))
                        ]
                        precond_neg = []
                        effect_add = [expr("At({}, {})".format(cargo, airport))]
                        effect_rem = [expr("In({}, {})".format(cargo, plane))]
                        unload = Action(
                            expr("Unload({}, {}, {})".format(cargo, plane, airport)),
                            [precond_pos, precond_neg],
                            [effect_add, effect_rem]
                        )
                        unloads.append(unload)
            return unloads



        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:

        #see example_have_cake.py
        possible_actions =  []
        # knowledge base of logical expressions (propositional logic)
        kb = PropKB()
        # Add current state's positive sentence's clauses to the propositional logic KB
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for action in self.actions_list:
            #Assume action to be possible
            is_possible = True
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False
                    break #don't keep searching

            #check if action is possible still
            if is_possible:
                for clause in action.precond_pos:
                    if clause not in kb.clauses:
                        is_possible = False
                        break #don't keep searching

                if is_possible:
                    possible_actions.append(action)

        return possible_actions


    def result(self, state: str, action: Action):
                #see example_have_cake.py
        new_state = FluentState([], [])

        old_state = decode_state(state, self.state_map)

        # The precondition and effect of an action are each conjunctions of literals
        # (positive or negated atomic sentences). The precondition defines the states
        # in which the action can be executed, and the effect defines the result of
        # executing the action. So, for each fluent (positive or negative) in the old state,
        # we look in the action effects (both the add and remove effects) to figure out
        # how the fluent will be carried over to the new state, as a positive or a negative fluent.

        for fluent in old_state.pos:
            if fluent not in action.effect_rem:
                    new_state.pos.append(fluent)
        for fluent in action.effect_add:
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)

        # The action may have positive and negative effects INDEPENDENTLY of the old state.
        # If they add effects need to make sure these fluents are added to the new state's
        # positive sentences and vice versa
        for fluent in old_state.neg:
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)
        for fluent in action.effect_rem:
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)

        #return the new state
        return encode_state(new_state, self.state_map)


    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)

        #Referencing discussion at https://discussions.udacity.com/t/understanding-ignore-precondition-heuristic/225906/28

        #The Node object describes the current state.
        # Ignore preconditions and look at each goal condition in the goal state,
        # count the number of individual actions that will make them happen.
        # No additional action for this condition needs to be perfomed.
        # if a goal condition is already satisfied by the current state

        #get logical expresssions
        kb = PropKB()

        # Add the current state's positive sentence's clauses to the propositional logic KB
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())

        # modify the goal_test() fn
        kb_clauses = kb.clauses
        actions_count = 0
        for clause in self.goal:
            if clause not in kb_clauses:
                actions_count += 1
        return actions_count

def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)



def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1','C2','C3']
    planes = ['P1','P2','P3']
    airports = ['JFK','SFO','ATL']
    #where cargo and planes are
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
    #where cargo and planes are not;
    #show that cargo has yet to be loaded in planes
    neg = [expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),
           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C2, P3)'),
           expr('At(C3, SFO)'),
           expr('At(C3, JFK)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P3, SFO)'),
           expr('At(P3, JFK)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)

def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1','C2','C3','C4']
    planes = ['P1','P2','P3','P4']
    airports = ['JFK','SFO','ATL','ORD']
    #where cargo and planes are
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]

    #where cargo and planes are not;
    #show that cargo has yet to be loaded in planes
    neg = [expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('At(C1, ORD)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('At(C2, ORD)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C3, SFO)'),
           expr('At(C3, JFK)'),
           expr('At(C3, ORD)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('At(C4, SFO)'),
           expr('At(C4, JFK)'),
           expr('At(C4, ATL)'),
           expr('At(C4, P1)'),
           expr('At(C4, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P1, ORD)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P2, ORD)'),
           ]

    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C3, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C4, SFO)'),
            ]

    return AirCargoProblem(cargos, planes, airports, init, goal)
