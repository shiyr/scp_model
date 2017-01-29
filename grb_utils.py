import logging
from collections import namedtuple
import gurobipy as grb
import numpy as np
from pandas import Series
import utils

GRB = grb.GRB
logger = logging.getLogger("grb_utils")

SavedBasis = namedtuple('SavedBasis', ['v_basis', 'c_basis'])

def status_name(model):
    status_num = model.Status
    names = ['Unknown', 'LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD',
             'UNBOUNDED', 'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT',
             'TIME_LIMIT', 'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL']
    if status_num < 0 or status_num >= len(names):
        return str(status_num)
    return names[status_num]


class UnexpectedInfeasibleModel(Exception):
    pass


class UnexpectedNumericalIssue(Exception):
    pass


@utils.logged
def get_basis(model):
    if model.SolCount == 0:
        return SavedBasis([], [])
    v_basis = [(var, var.VBasis, var.X) for var in model.getVars()]
    c_basis = [(constr, constr.CBasis, constr.Slack) for constr in model.getConstrs()]
    return SavedBasis(v_basis, c_basis)

@utils.logged
def reset_lp(model, saved_basis):
    model.reset()

    for var, basis, value in saved_basis.v_basis:
        try:
            var.VBasis = basis
        except grb.GurobiError:
            pass
    for constr, basis, slack in saved_basis.c_basis:
        try:
            constr.CBasis = basis
        except grb.GurobiERror:
            pass


class StoppingCriteria(object):
    def __init__(self, min_seconds, gap_to_plateau, nodes_per_second):
        """
        min_seconds -- number of seconds to wait for improvement
        gap_to_plateau -- number of additional seconds to wait per mipgap %
        nodes_per_second -- assumed number of nodes per second (for determinism)
        """
        self.min_seconds = min_seconds
        self.gap_to_plateau = gap_to_plateau
        self.nodes_per_second = nodes_per_second
        self.min_nodes = np.ceil(self.nodes_per_second * self.min_seconds)
        logger.info("mip stopping criteria: min_nodes=%d", self.min_nodes)
        self.last_improvement_node = 0

    def __call__(self, model, where):
        # where    numerical value      optimizer status
        # POLLING       = 0,            Periodic polling callback
        # PRESOLVE      = 1,            Currently performing presolve
        # SIMPLEX       = 2,            Currently in simplex
        # MIP           = 3,            Currently in MIP
        # MIPSOL        = 4,            Found a new MIP incumbent
        # MIPNODE       = 5,            Currently exploring a MIP node
        # MESSAGE       = 6,            Printing a log message
        # BARRIER       = 7,            Currently in barrier
        
        # what          where     result type       description
        # MIP_OBJBST	MIP         double          Current best objective.
        # MIP_OBJBND	MIP         double          Current best objective bound.
        # MIP_NODCNT	MIP         double          Current explored node count.
        # MIP_SOLCNT	MIP         int             Current count of feasible solutions found.
        # MIPSOL_NODCNT	MIPSOL      double          Current explored node count.
        if where == GRB.callback.POLLING or where == GRB.callback.MESSAGE:
            return
        if where == GRB.callback.MIP:
            self.num_solutions = model.cbGet(GRB.callback.MIP_SOLCNT)
            self.node_idx = model.cbGet(GRB.callback.MIP_NODCNT)
            self.obj_bound = model.cbGet(GRB.callback.MIP_OBJBND)
            self.obj_best = model.cbGet(GRB.callback.MIP_OBJBST)
            if self.termination_criteria(model):
                model.terminate()
        elif where == GRB.callback.MIPSOL:
            self.node_idx = model.cbGet(GRB.callback.MIPSOL_NODCNT)
            self.last_improvement_node = self.node_idx

    def termination_criteria(self, model):
        wall_clock_limit = self.min_seconds * 5
        logger.debug("in mip callback %f < %f (%f) %d",
                     model.cbGet(GRB.callback.RUNTIME),
                     wall_clock_limit,
                     model.cbGet(GRB.callback.MIP_OBJBST),
                     model.cbGet(GRB.callback.MIP_SOLCNT))
        if self.num_solutions > 0 and model.cbGet(GRB.callback.RUNTIME) > wall_clock_limit:
            logger.info("stopping MIP due to total time exceeding wall clock time limit of %f",
                        wall_clock_limit)
            return True

        if self.node_idx <= self.min_nodes:
            return False
        if self.num_solutions == 0:
            return False

        abs_gap = np.abs(self.obj_best - self.obj_bound)
        if abs_gap < 1e-5:
            logger.info("stopping mip due to zero gap %.2f %.2f", self.obj_best, self.obj_bound)
            return True
        pct_gap = 100.0 * abs_gap / np.max(np.abs(self.obj_best), 1e-5)
        nodes_since_last_solution = self.node_idx - self.last_improvement_node
        max_plateau = self.min_nodes + self.nodes_per_second * pct_gap * self.gap_to_plateau
        if nodes_since_last_solution > max_plateau:
            logger.info("stopping mip due to solution plateau (%d > %d, gap=%.2f)",
                        nodes_since_last_solution, max_plateau, pct_gap)
            return True

def object_to_callback(obj):
    """
    shows callback object as a function to gurobi
    """
    def cb(model, where):
        try:
            return obj(model, where)
        except grb.GurobiError as ex:
            logger.critical("gurobi exception in callback %s", ex)
            raise
    return cb

def ones_column(*rows):
    return grb.Column([1]*len(rows), rows)

def add_columns(*columns):
    result = grb.Column()
    for c in columns:
        if c is not None and c.size() > 0:
            coeffs, constrs = zip(*[(c.getCoeff(i), c.getConstr(i)) for i in range(c.size())])
            result.addTerms(coeffs, list(constrs))
    return result

def equate_expressions(model, lhs, rhs, name, key = None):
    if keys is None:
        keys = set(lhs.keys())
        rhs_keys = set(rhs.keys())
        if keys != rhs_keys:
            keys = keys.union(rhs_keys)
            logger.debug("equate expression keys for %s don't match (2 * %d > %d + %d)",
                         name, len(keys), len(lhs), len(rhs))
    return Series({key: model.addConstr(lhs.get(key, 0) - rhs.get(key, 0) == 0, name=get_name(name, *key))
                   for key in keys})

def unpack_name(name):
    if type(name) == tuple:
        return '.'.join(str(n) for n in name)
    else:
        return str(name)

def get_name(*names):
    return '.'.join(unpack_name(name) for name in names).replace(' ', '_').replace('+','_').replace(':','_')

def expression_to_constraint(model, name, expression, value, model_sense, relaxation_factor):
    if model_sense == GRB.MAXIMIZE:
        if expr_is_maximized(expression):
            bound = np.floor(value)
        else:
            bound = np.floor(value * np.power(relaxation_factor, np.sign(-value)))
        constr = model.addConstr(expression >= bound, name=name)
    else:
        if expr_is_minimized(expression):
            bound = np.ceil(value)
        else:
            bound = np.ceil(value * np.power(relaxation_factor, np.sign(value)))
        constr = model.addConstr(expression <= bound, name=name)
    return constr

