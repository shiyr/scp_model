from __future__ import division
from collections import defaultdict
from operator import attrgetter
import numpy as np
from pandas import Series, DataFrame
import logging

logging.basicConfig(level=logging.INFO, format="%(relativeCreated)5d:%(levelname)-5s:%(name)-8s:%(message)s")
logger = logging.getLogger("production_data")


class ProductionData(object):
    def __init__(self, optimization_run):
        self.num_weeks = optimization_run.num_weeks
        self.parameters = {name: p.value for name, p in optimization_run.parameters.iteritems()}
        optimization_run.log_parameters()
        logger.info("Read parameters table")
        self.s1_weeks = self.parameters.get('stage_1_size', 1)
        self.sample_size = self.parameters.get('sample_size', 100)

        self.products = optimization_run.products
        logger.info("Read %d products", len(self.products))

        self.nodes = optimization_run.nodes
        logger.info("Read %d nodes", len(self.nodes))

        self.modes = optimization_run.modes
        logger.info("Read %d modes", len(self.modes))

        self.edges = [(e.origin, e.destination, e.mode) for e in optimization_run.edges]
        self.lead_times = {(e.origin, e.destination, e.mode): e.lead_time for e in optimization_run.edges}
        self.tcap = {(e.origin, e.destination, e.mode): e.tcap for e in optimization_run.edges}
        logger.info("Read %d edges", len(self.edges))

        self.lanes = [(l.origin, l.destination, l.prod, l.mode) for l in optimization_run.lanes]
        self.tcost = {(l.origin, l.destination, l.prod, l.mode): l.tcost for l in optimization_run.lanes}
        logger.info("Read %d lanes", len(self.lanes))
        
        self.weekly_receives = optimization_run.weekly_receives
        self.start_to_rec = {(r.lane.origin, r.lane.destination, r.lane.prod, r.lane.mode, r.week):
                              r.start_to_rec for r in self.weekly_receives}
        logger.info("Read %d weekly_receives", len(self.start_to_rec))

        self.site_prod_produces = [(map.site, map.prod) for map in optimization_run.site_prod_produces]
        self.sites = list(np.unique([map.site for map in optimization_run.site_prod_produces]))
        logger.info("Read %d site_prod_produces", len(self.site_prod_produces))

        self.site_prod_invs = [(map.site, map.prod) for map in optimization_run.site_prod_invs]
        self.start_inv = {(map.site, map.prod): map.start_inv for map in optimization_run.site_prod_invs}
        self.icost = {(map.site, map.prod): map.icost for map in optimization_run.site_prod_invs}
        logger.info("Read %d site_prod_invs", len(self.site_prod_invs))

        self.cust_prods = [(map.cust, map.prod) for map in optimization_run.cust_prods]
        self.customers = list(np.unique([map.cust for map in optimization_run.cust_prods]))
        self.start_backlog = {(map.cust, map.prod): map.start_backlog for map in optimization_run.cust_prods}
        self.pcost = {(map.cust, map.prod): map.pcost for map in optimization_run.cust_prods}
        logger.info("Read %d cust_prods", len(self.cust_prods))

        self.demands = self.get_random_demands(optimization_run.weekly_demands, self.sample_size)
        logger.info("Read %d weekly_demands", len(self.demands))
        self.stage_1_demands = {(n,k,t): val for (n,k,t,w), val in self.demands.iteritems() if w == 0}
        logger.info("Read %d stage_1_demands", len(self.stage_1_demands))
        self.stage_2_demands = defaultdict(dict)
        for (n,k,t,w), val in self.demands.iteritems():
            if w > 0:
                self.stage_2_demands[w][n,k,t] = val
        logger.info("Read %d stage_2_demands", len(self.stage_2_demands))
        
        self.yields = self.get_random_yields(self.sites, self.num_weeks, self.s1_weeks, self.sample_size)
        logger.info("Read %d weekly_yields", len(self.yields))
        self.stage_1_yields = {(n,t): val for (n,t,w), val in self.yields.iteritems() if w == 0}
        logger.info("Read %d stage_1_yields", len(self.stage_1_yields))
        self.stage_2_yields = defaultdict(dict)
        for (n,t,w), val in self.yields.iteritems():
            if w > 0:
                self.stage_2_yields[w][n,t] = val
        logger.info("Read %d stage_2_yields", len(self.stage_2_yields))

        self.site_capacities = optimization_run.site_capacities
        self.pcap = {s.site: s.pcap for s in optimization_run.site_capacities}
        self.ucap = {s.site: s.ucap for s in optimization_run.site_capacities}
        self.ocap = {s.site: s.ocap for s in optimization_run.site_capacities}
        self.lcap = {s.site: s.lcap for s in optimization_run.site_capacities}
        self.icap = {s.site: s.icap for s in optimization_run.site_capacities}
        self.fcost = {s.site: s.fcost for s in optimization_run.site_capacities}
        self.ucost = {s.site: s.ucost for s in optimization_run.site_capacities}
        self.ocost = {s.site: s.ocost for s in optimization_run.site_capacities}
        self.wcost = {s.site: s.wcost for s in optimization_run.site_capacities}
        logger.info("Read %d site_capacities", len(self.site_capacities))
        
        self.prod_inputs = {p: [obj.input for obj in p.inputs] for p in self.products}
        self.prod_outputs = {p: [obj.output for obj in p.outputs] for p in self.products}
        self.produced_at_site = {n: [map.prod for map in n.produced_prods] for n in self.nodes}
        self.inv_at_site = {n: [map.prod for map in n.stored_prods] for n in self.nodes}
        self.sites_produce_prod = {k: [map.site for map in k.produced_in_sites] for k in self.products}
        self.sites_store_prod = {k: [map.site for map in k.stored_in_sites] for k in self.products}

    def get_random_demands(self, weekly_demands, size=100):
        demands = {}
        ref_demands = {(d.cust_prod.cust, d.cust_prod.prod, d.week, d.sample_id): d.quantity
                       for d in weekly_demands}
        for (n,k,t,w), val in ref_demands.iteritems():
            demands[n,k,t,w] = val
            if w == 1:
                new_demands = np.random.normal(val, 0.1*val, size-1)
                for i in range(2, size+1):
                    demands[n,k,t,i] = new_demands[i-2]
        return demands

    def get_random_yields(self, sites, num_weeks, s1_weeks, size=100):
        yields = {}
        for n in sites:
            for t in range(num_weeks):
                if t < s1_weeks:
                    yields[n,t,0] = 1
                else:
                    for w in range(1, size+1):
                        yields[n,t,w] = np.random.uniform(0.9,1)
        return yields

    def get_weekly_states(self, num_weeks, s1_weeks, size=100):
        weekly_states = []
        for t in range(num_weeks):
            if t < s1_weeks:
                weekly_states.append((t,0))
            else:
                for w in range(1, size+1):
                    weekly_states.append((t,w))
        return weekly_states
    
    def get_state_probabilities(self, weekly_states, size=100):
        p_state = defaultdict(float)
        for (t,w) in weekly_states:
            if w == 0:
                p_state[t,w] = 1
            else:
                p_state[t,w] = 1/size
        return p_state

    def get_pres_and_sucs(self, weekly_states, s1_weeks, size=100):
        pres = {}
        sucs = {}
        for (t,w) in weekly_states:
            if t <= s1_weeks:
                pres[t,w] = 0
            else:
                pres[t,w] = w
        for (t,w) in weekly_states:
            if t < s1_weeks - 1:
                sucs[t,w] = [0]
            elif t == s1_weeks - 1:
                sucs[t,w] = range(1, size+1)
            else:
                sucs[t,w] = [w]
        return pres, sucs
