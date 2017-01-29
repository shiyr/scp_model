import os
import logging
import time
import xlsxwriter
from collections import defaultdict, namedtuple
from pandas import Series, DataFrame
import numpy as np
import gurobipy as grb
import utils
from grb_utils import (SavedBasis,
                       status_name,
                       get_basis,
                       reset_lp,
                       StoppingCriteria,
                       object_to_callback,
                       ones_column,
                       add_columns,
                       get_name,
                       equate_expressions)

GRB = grb.GRB
logger = logging.getLogger("scp_optimizer")

eps = 1e-5

ObjectiveConstraintDual = namedtuple('ObjectiveConstraintDual', ['parent_level', 'child_level', 'dual'])


def fix_var(dvar, x):
    """
    sets a variable to be fixed at a specific value x
    """
    dvar.LB = x
    dvar.UB = x


def get_production_variables(model, site_prod_pairs, weeks, prefix, lb=0):
    return Series({(n,k,t): model.addVar(lb=lb, name=get_name(prefix,n,k,t))
                            for (n,k) in site_prod_pairs
                            for t in weeks})

def get_shipment_variables(model, lanes, weeks, prefix, lb=0):
    return Series({(i,j,k,m,t): model.addVar(lb=lb, name=get_name(prefix,i,j,k,m,t))
                                for (i,j,k,m) in lanes
                                for t in weeks})

def get_ship_to_receive_variables(model, lanes, weeks, lt, prefix, lb=0):
    vars = {}
    for (i,j,k,m) in lanes:
        for t in weeks:
            s = t - lt[i,j,m]
            if s >= 0:
                vars[i,j,k,m,t,s] = model.addVar(lb=lb, name=get_name(prefix,i,j,k,m,t,s))
    return Series(vars)

def get_inventory_variables(model, site_prod_pairs, weeks, prefix, lb=0):
    return Series({(n,k,t): model.addVar(lb=lb, name=get_name(prefix,n,k,t))
                            for (n,k) in site_prod_pairs
                            for t in weeks})

def get_production_time_variables(model, sites, weeks, prefix, lb=0):
    return Series({(n,t): model.addVar(lb=lb, name=get_name(prefix,n,t))
                          for n in sites
                          for t in weeks})

def get_line_variables(model, sites, weeks, prefix):
    return Series({(n,t): model.addVar(ub=5.0, vtype=GRB.INTEGER, name=get_name(prefix,n,t))
                          for n in sites
                          for t in weeks})

def get_cap_switch_variables(model, sites, weeks, prefix, lb=0):
    return Series({(n,t): model.addVar(lb=lb, name=get_name(prefix,n,t))
                          for n in sites
                          for t in range(1, max(weeks)+1)})

def get_demand_backlog_variables(model, cust_prod_pairs, weeks, prefix, lb=0):
    return Series({(n,k,t): model.addVar(lb=lb, name=get_name(prefix,n,k,t))
                            for (n,k) in cust_prod_pairs
                            for t in weeks})

def get_quality_hold_variables(model, site_prod_pairs, weeks, prefix, lb=0):
    return Series({(n,k,t): model.addVar(lb=lb, name=get_name(prefix,n,k,t))
                  for (n,k) in site_prod_pairs
                  for t in weeks})

def get_flow_expressions(to_ship, to_receive):
    flow_out = defaultdict(grb.LinExpr)
    flow_in = defaultdict(grb.LinExpr)
    for (i,j,k,m,t) in to_ship.keys():
        flow_out[i,k,t] += to_ship[i,j,k,m,t]
        flow_in[j,k,t] += to_receive[i,j,k,m,t]
    return Series(flow_out), Series(flow_in)


def get_production_constraints(model, to_produce, ut, ot, line, pcap, prefix):
    # ut[n,t] + S p[n,k,t] = PC[n] * l[n,t] + ot[n,t]
    return Series({(n,t): model.addConstr(ut[n,t] + var == pcap[n] * line[n,t] + ot[n,t],
                                          name=get_name(prefix,n,t))
                          for (n,t), var in to_produce.iteritems()})

def get_quality_hold_constraints(model, yields, to_produce, q_hold, prefix):
    # (1 - y[n,t]) * p[n,k,t] = q[n,k,t]
    return Series({(n,k,t): model.addConstr((1 - yields[n,t]) * var == q_hold[n,k,t],
                                            name=get_name(prefix,n,k,t))
                            for (n,k,t), var in to_produce.iteritems()})

def get_ot_ut_constraints(model, produce, line, cap, prefix):
    # ot[n,t] <= OC[n] * l[n,t]
    # ut[n,t] <= UC[n] * l[n,t]
    return Series({(n,t): model.addConstr(var <= cap[n] * line[n,t], name=get_name(prefix,n,t))
                          for (n,t), var in produce.iteritems()})

def get_cap_switch_constraints(model, line, switch, prefix):
    # w[n,t] >= l[n,t] - l[n,t-1]
    # w[n,t] >= - (l[n,t] - l[n,t-1])
    pos = {}
    neg = {}
    for (n,t), var in switch.iteritems():
        pos[n,t] = model.addConstr(var >= line[n,t] - line[n,t-1], name=get_name(prefix+'pos',n,t))
        neg[n,t] = model.addConstr(var >= -(line[n,t] - line[n,t-1]), name=get_name(prefix+'neg',n,t))
    return Series(pos), Series(neg)

def get_flow_conservation_constraints(model, start_inv, inv, flow_in, to_produce, flow_out,
                                      to_consume, q_hold, prefix):
    lhs_vars = ['inv', 'flow_in', 'to_produce']
    rhs_vars = ['flow_out', 'to_consume', 'inv']
    constrs = {}
    for (n,k,t) in inv.keys():
        lhs_expr = grb.LinExpr()
        rhs_expr = grb.LinExpr()
        for (lhs_var, rhs_var) in zip(lhs_vars, rhs_vars):
            try:
                if t == 0 and lhs_var == 'inv':
                    lhs_expr += start_inv[n,k]
                elif lhs_var == 'inv':
                    lhs_expr += eval(lhs_var)[n,k,t-1]
                else:
                    lhs_expr += eval(lhs_var)[n,k,t]
            except KeyError:
                pass
            try:
                rhs_expr += eval(rhs_var)[n,k,t]
            except KeyError:
                pass
        try:
            rhs_expr += q_hold[n,k,t]
        except KeyError:
            pass
        constrs[n,k,t] = model.addConstr(lhs_expr == rhs_expr, name=get_name(prefix,n,k,t))
    return Series(constrs)

def get_total_receive_constraints(model, start_to_rec, to_receive, ship_to_rec, prefix):
    receive = defaultdict(grb.LinExpr)
    for (i,j,k,m,t,s), var in ship_to_rec.iteritems():
        receive[i,j,k,m,t] += ship_to_rec[i,j,k,m,t,s]
    return Series({(i,j,k,m,t): model.addConstr(var == receive[i,j,k,m,t] + start_to_rec[i,j,k,m,t],
                                                name=get_name(prefix,i,j,k,m,t))
                                for (i,j,k,m,t), var in to_receive.iteritems()})

def get_trans_cap_constraints(model, tcap, to_ship, prefix):
    ship = defaultdict(grb.LinExpr)
    for (i,j,k,m,t) in to_ship.keys():
        ship[i,j,m,t] += to_ship[i,j,k,m,t]
    return Series({(i,j,m,t): model.addConstr(var <= tcap[i,j,m], name=get_name(prefix,i,j,m,t))
                              for (i,j,m,t), var in ship.iteritems()})

def get_inv_cap_constraints(model, icap, inv, prefix):
    total_inv = defaultdict(grb.LinExpr)
    for (n,k,t) in inv.keys():
        total_inv[n,t] += inv[n,k,t]
    return Series({(n,t): model.addConstr(var <= icap[n], name=get_name(prefix,n,t))
                          for (n,t), var in total_inv.iteritems()})

def get_ship_to_rec_constraints(model, weeks, lt, to_ship, ship_to_rec, prefix):
    constrs = {}
    for (i,j,k,m,s), var in to_ship.iteritems():
        t = s + lt[i,j,m]
        if t <= max(weeks):
            constrs[i,j,k,m,s] = model.addConstr(var == ship_to_rec[i,j,k,m,t,s],name=get_name(prefix,i,j,k,m,s))
    return Series(constrs)

def get_demand_constraints(model, demand, start_backlog, flow_in, backlog, prefix):
    constrs = {}
    for (n,k,t), var in backlog.iteritems():
        if t == 0:
            constrs[n,k,t] = model.addConstr(flow_in[n,k,t] + var == demand[n,k,t] + start_backlog[n,k],
                                             name=get_name(prefix,n,k,t))
        else:
            constrs[n,k,t] = model.addConstr(flow_in[n,k,t] + var == demand[n,k,t] + backlog[n,k,t-1],
                                             name=get_name(prefix,n,k,t))
    return Series(constrs)



class SCPOptimizer(object):
    def __init__(self, data):
        self.data = data
        self.weeks = range(self.data.num_weeks)
        logger.info("creating gurobi model object, local gurobi version=%s local platform=%s",
                    grb.gurobi.version(), grb.gurobi.platform())
        self.model = grb.Model()
        logger.info("created gurobi model")
    
        self.model.setParam('Method', 3)
        # -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent
        # Automatic will typically choose non-deterministic concurrent (Method=3) for LP, barrier (Method=2) for QP or QCP, and dual (Method=1) for MIP root node.
        # Only simplex and barrier algorithms are available for continuous QP models.
        # Only primal and dual simplex are available for solving the root of an MIQP mpde.
        # Only barrier is available for continuous QCP models.
        # Concurrent (Method=3) run multiple solvers on multiple threads simultaneously, and choose the one that finishes first. Method=3 is often faster than Method=4 but can produce different optimal bases.
        self.model.setParam('InfUnbdInfo', 1)
        # Determines whether simplex (and crossover) will compute additional information when a model is determined to be infeasible or unbounded.
        # Set to 1 if you want to query the unbounded ray for unbounded models, or the infeasibility proof for infeasible models.
        # This parameter applies to LP only.
        self.model.setParam('IntFeasTol', 1e-5)
        # An integrality restriction on a variable is considered satisfied when the variable's value is less than IntFeasTol from the nearest integer value.
        # Only affects MIP models.
        self.model.setParam('Threads', 0)
        # Controls the number of threads to apply to parallel algorithms (concurrent LP, parallel barrier, parallel MIP, etc.)
        # Threads=0 will generally use all of the cores in the machine, but it may choose to use fewer.
    
        if self.get_global_parameter('gurobi_numeric_stability_flag', False):
            self.model.setParam('Quad', 1)
            # Enables or disables quad precision computation in simplex. -1 allows the algorithm to decide.
            # Quad precision can sometimes help solve numerically challenging models, but increase in runtime.
            self.model.setParam('NumericFocus', 3)
            # Controls the degree to which the code attempts to detect and manage numerical issues.
            # 0 makes automatic choice, with a slight preference for speed. 1-3 increasingly shift the focus towards being more careful in numerical computations.
            self.model.setParam('OptimalityTol', 1e-5)
            # Reduced costs must all be smaller than OptimalityTol in the improving direction in order for a model to be declared optimal.
        
        self._build_model()
    
    def _build_model(self):
        self.model.setParam('UpdateMode', 1)
        self.model.__dict__['zero_var'] = self.model.addVar(ub=0, name='zero_var')
    
        self.to_produce = get_production_variables(self.model, self.data.site_prod_produces,
                                                   self.weeks, 'to_produce')
        logger.info("Read %d to_produce variables", len(self.to_produce))
        self.q_hold = get_quality_hold_variables(self.model, self.data.site_prod_produces,
                                                 self.weeks, 'q_hold')
        logger.info("Read %d q_hold variables", len(self.q_hold))
        self.to_ship = get_shipment_variables(self.model, self.data.lanes,
                                              self.weeks, 'to_ship')
        logger.info("Read %d to_ship variables", len(self.to_ship))
        self.ship_to_rec = get_ship_to_receive_variables(self.model, self.data.lanes,
                                                         self.weeks, self.data.lead_times, 'ship_to_rec')
        logger.info("Read %d ship_to_rec variables", len(self.ship_to_rec))
        self.to_receive = get_shipment_variables(self.model, self.data.lanes,
                                                 self.weeks, 'to_rec')
        logger.info("Read %d to_receive variables", len(self.to_receive))
        self.inv = get_inventory_variables(self.model, self.data.site_prod_invs, self.weeks, 'inv')
        logger.info("Read %d inv variables", len(self.inv))
        self.ut = get_production_time_variables(self.model, self.data.sites, self.weeks, 'ut')
        logger.info("Read %d ut variables", len(self.ut))
        self.ot = get_production_time_variables(self.model, self.data.sites, self.weeks, 'ot')
        logger.info("Read %d ot variables", len(self.ot))
        self.line = get_line_variables(self.model, self.data.sites, self.weeks, 'line')
        logger.info("Read %d line variables", len(self.line))
        self.switch = get_cap_switch_variables(self.model, self.data.sites, self.weeks, 'switch')
        logger.info("Read %d switch variables", len(self.switch))
        self.backlog = get_demand_backlog_variables(self.model, self.data.cust_prods, self.weeks, 'backlog')
        logger.info("Read %d backlog variables", len(self.backlog))
        
        self.model_update()
        
        self.to_produce_by_site = Series({(n,t): grb.quicksum(self.to_produce[n,k,t]
                                                 for k in self.data.produced_at_site[n])
                                                 for n in self.data.sites
                                                 for t in self.weeks})
        
        self.to_consume = Series({(n,p,t): grb.quicksum(self.to_produce[n,g.output,t] for g in p.outputs)
                                           for p in self.data.products if p.type == 'part'
                                           for n in self.data.nodes if n.type == 'manufacturer'
                                           for t in self.weeks})
                                           
        self.flow_out, self.flow_in = get_flow_expressions(self.to_ship, self.to_receive)
        
        for (n,t), var in self.line.iteritems():
            if n.type == 'supplier':
                # fix_var(var, 1)
                var.UB = 1
            # elif t > 0 and t < (self.data.num_weeks - 1):
            #     var.LB = 1
        
        self.production_constraints = get_production_constraints(self.model, self.to_produce_by_site,
                                                                 self.ut, self.ot, self.line,
                                                                 self.data.pcap, 'pcap')
        self.quality_hold_constraints = get_quality_hold_constraints(self.model, self.data.yields,
                                                                     self.to_produce, self.q_hold, 'qhold')
        self.undertime_constraints = get_ot_ut_constraints(self.model, self.ut, self.line,
                                                           self.data.ucap, 'ucap')
        self.overtime_constraints = get_ot_ut_constraints(self.model, self.ot, self.line,
                                                          self.data.ocap, 'ocap')
        self.pos_cap_switch_constrs, self.neg_cap_switch_constrs = get_cap_switch_constraints(self.model,
                                                                   self.line, self.switch, 'switch')
        self.flow_conservation_constraints = get_flow_conservation_constraints(self.model,
                                                            self.data.start_inv, self.inv,
                                                            self.flow_in, self.to_produce, self.flow_out,
                                                            self.to_consume, self.q_hold, 'flow_consv')
        self.total_receive_constraints = get_total_receive_constraints(self.model, self.data.start_to_rec,
                                                            self.to_receive, self.ship_to_rec, 'total_rec')
        self.trans_cap_constraints = get_trans_cap_constraints(self.model, self.data.tcap,
                                                               self.to_ship, 'tcap')
        self.inv_cap_constraints = get_inv_cap_constraints(self.model, self.data.icap, self.inv, 'icap')
        self.ship_to_rec_constraints = get_ship_to_rec_constraints(self.model, self.weeks, self.data.lead_times,
                                                            self.to_ship, self.ship_to_rec, 'ship_to_rec_ctr')
        self.demand_constraints = get_demand_constraints(self.model, self.data.demands, self.data.start_backlog,
                                                         self.flow_in, self.backlog, 'demand_ctr')
        
        self.model_update()
        
        # for k, constr in self.demand_constraints.iteritems():
        #     row = self.model.getRow(constr)
        #     print row
    
    
    @utils.logged
    def model_update(self):
        self.model.update()
    
    def optimize_objective(self):
        inv_cost = grb.quicksum(self.data.icost[n,k] * var for (n,k,t), var in self.inv.iteritems())
        prod_cost = grb.quicksum(self.data.fcost[n] * var for (n,t), var in self.line.iteritems()) + \
                    grb.quicksum(self.data.ucost[n] * var for (n,t), var in self.ut.iteritems()) + \
                    grb.quicksum(self.data.ocost[n] * var for (n,t), var in self.ot.iteritems())
        tran_cost = grb.quicksum(self.data.tcost[i,j,k,m] * var for (i,j,k,m,t), var in self.to_ship.iteritems())
        pen_cost = grb.quicksum(self.data.pcost[n,k] * var for (n,k,t), var in self.backlog.iteritems())
        cap_switch_cost = grb.quicksum(self.data.wcost[n] * var for (n,t), var in self.switch.iteritems())
        obj = inv_cost + prod_cost + tran_cost + pen_cost + cap_switch_cost
        self.optimize('min_backlog', objs=obj)
        print 'inv_cost:', sum(self.data.icost[n,k] * var.X for (n,k,t), var in self.inv.iteritems())
        print 'fixed_cost:', sum(self.data.fcost[n] * var.X for (n,t), var in self.line.iteritems())
        print 'ut_cost:', sum(self.data.ucost[n] * var.X for (n,t), var in self.ut.iteritems())
        print 'ot_cost:', sum(self.data.ocost[n] * var.X for (n,t), var in self.ot.iteritems())
        print 'tran_cost:',sum(self.data.tcost[i,j,k,m] * var.X for (i,j,k,m,t), var in self.to_ship.iteritems())
        print 'pen_cost:', sum(self.data.pcost[n,k] * var.X for (n,k,t), var in self.backlog.iteritems())
        print 'cap_switch_cost:', sum(self.data.wcost[n] * var.X for (n,t), var in self.switch.iteritems())
        print 'obj:', self.model.ObjVal
        # for name, value in zip(self.model.getAttr('VarName', list(self.to_produce)), self.model.getAttr('X', list(self.to_produce))):
        #     print name, value
        self.output_to_excel()
    

    def optimize(self, name, objs = None, sense = GRB.MINIMIZE, mip_start = None):
        if objs is not None:
            if isinstance(objs, grb.LinExpr):
                objs = [objs]
            self.model.setObjective(grb.quicksum(objs))
            self.model.ModelSense = sense

        time_limit = self.get_global_parameter('time_limit', 600)
        mipgap = self.get_global_parameter('mipgap', 0.001)
        callback = None

        if self.model.isMIP:
            if self.get_global_parameter('write_mip_flag', False):
                file_name = "%s.mps" %name
                logger.info("writing MIP to %s", file_name)
                self.model.write(file_name)
            self.model.setParam('MIPGap', mipgap)
            if self.get_global_parameter('plateau_stopping_flag', True):
                logger.info("using node based mip stopping criteria")
                callback = StoppingCriteria(np.ceil(time_limit / 10.0),
                                            gap_to_plateau = 0.1,
                                            nodes_per_second = 10)
                callback = object_to_callback(callback)
                time_limit *= 10
        else:
            self.model.setParam('TimeLimit', 1e6)

        do_reset = self.get_global_parameter('reset_lp_flag', False)
        if do_reset:
            reset_lp(self.model, self.saved_basis)
        if mip_start:
            for dvar, start in mip_start:
                dvar.Start = start

        logger.info("optimizing objective %s with a time limit of %d seconds", name, time_limit)
        if self.get_global_parameter('no_crossover_flag', False):
            self.model.setParam('Crossover', 0)
            self.model.setParam('Method', 2)

        self.model.optimize(callback)
        logger.info("gurobi returned with status %s and reported runtime of %.4f seconds",
                    status_name(self.model), self.model.runtime)
        self.write_lp_file(name)

        if do_reset and self.has_feasible_solution and not self.model.isMIP:
            self.saved_basis = get_basis(self.model)

        if self.has_feasible_solution:
            elements = (name, self.model.ObjVal, status_name(self.model))
            logger.info("objective value for %s = %10.2f, status = %s", *elements)
            if not self.model.isMIP:
                self.log_nonzero_duals(name)
                self.save_fairshare_lb_duals(name)
                self.save_objective_constraint_duals(name)
        else:
            logger.critical("optimizer %s did not find a feasible solution! status = %s",
                            name, status_name(self.model))
            self.write_gurobi_file("%s.mps" %name)
            if self.model.status == GRB.INFEASIBLE or self.model.status == GRB.INF_OR_UNBD:
                logger.info("computing IIS")
                self.model.computeIIS()
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print c.constrName, c
                for v in self.model.getVars():
                    if v.IISLB:
                        print v.varName, v, 'LB Vio'
                    if v.IISUB:
                        print v.varName, v, 'UB Vio'
            else:
                if self.model.status == GRB.TIME_LIMIT:
                    raise Exception("Time limit of {0} seconds exceeded.".format(time_limit))

        logger.info("optimize %s completed", name)
    
    @property
    def has_feasible_solution(self):
        return self.model.SolCount > 0
    
    def save_fairshare_lb_duals(self, name):
        """
        gets the fairshare lower bounds from previous iterations and saves them to a new rc.<name>
        """
        if len(self.fairshare_lbs) == 0:
            return
        pi = self.fairshare_lbs.apply(lambda row: np.around(row.constr.Pi, 4), axis=1)
        self.fairshare_lbs['pi.%s' %name] = pi

    @utils.logged
    @utils.skip_if_exception(grb.GurobiError, "no duals available")
    def save_objective_constraint_duals(self, name):
        epsilon = 1e-6
        count = 0
        for parent_name, constr in self.objective_constraints:
            dual = np.around(constr.Pi, 4)
            if abs(dual) > epsilon:
                count += 1
                self.objective_constraint_duals.append(ObjectiveConstraintDual(parent_name, name, dual))
        logger.info("objective %s affected by %d higher level objectives", name, count)

    def get_global_parameter(self, name, default = None):
        return self.data.parameters.get(name, default)

    def write_lp_file(self, name):
        self.model.write(name + '.lp')

    def write_gurobi_file(self, name):
        self.model.write(name)

    def output_to_excel(self, input_filename=None):
        if input_filename is not None:
            str = input_filename[:-5]
        else:
            str = time.strftime("%y%m%d_%H%M%S")
        wb = xlsxwriter.Workbook('output_'+str+'.xlsx')
        ws = wb.add_worksheet('to_produce')
        self.output_to_sheet(ws, self.to_produce)
        ws = wb.add_worksheet('undertime')
        self.output_to_sheet(ws, self.ut)
        ws = wb.add_worksheet('overtime')
        self.output_to_sheet(ws, self.ot)
        ws = wb.add_worksheet('to_ship')
        self.output_to_sheet(ws, self.to_ship)
        ws = wb.add_worksheet('to_receive')
        self.output_to_sheet(ws, self.to_receive)
        ws = wb.add_worksheet('inv')
        self.output_to_sheet(ws, self.inv)
        ws = wb.add_worksheet('backlog')
        self.output_to_sheet(ws, self.backlog)
        ws = wb.add_worksheet('line')
        self.output_to_sheet(ws, self.line)
        wb.close()

    def output_to_sheet(self, ws, dic):
        names = self.model.getAttr('VarName', list(dic))
        values = self.model.getAttr('X', list(dic))
        for row in range(1, len(names) + 1):
            keys = names[row-1].split('.')
            for col in range(len(keys) - 1):
                ws.write(row,col,keys[col+1])
            ws.write(row,col+1,values[row-1])

    def satisfied_demand(self, ws, dic, data):
        keys = dic.keys()
        row = 0
        for key in keys:
            row += 1
            names = dic[key].VarName.split('.')
            for col in range(len(names) - 1):
                ws.write(row,col,names[col+1])
            ws.write(row,col+1,data[key] - dic[key].X)
