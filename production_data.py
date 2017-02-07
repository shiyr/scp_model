from collections import defaultdict
from operator import attrgetter
from pandas import Series, DataFrame
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(relativeCreated)5d:%(levelname)-5s:%(name)-8s:%(message)s")
logger = logging.getLogger("production_data")


class ProductionData(object):
    def __init__(self, optimization_run):
        self.optimization_run = optimization_run
        self.num_weeks = self.optimization_run.num_weeks
        self.parameters = {p.name: p.value for p in self.optimization_run.parameters}
        self.optimization_run.log_parameters()
        logger.info("Read parameters table")

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

        self.weekly_demands = optimization_run.weekly_demands
        # self.demands = {(d.cust_prod.cust, d.cust_prod.prod, d.week): d.quantity for d in self.weekly_demands}
        self.demands = {}
        logger.info("Read %d weekly_demands", len(self.demands))
        
        self.weekly_yields = optimization_run.weekly_yields
        # self.yields = {(y.site, y.week): y.quantity for y in self.weekly_yields}
        self.yields = {}
        logger.info("Read %d weekly_yields", len(self.yields))

        self.site_capacities = optimization_run.site_capacities
        self.pcap = {s.site: s.pcap for s in optimization_run.site_capacities}
        self.ucap = {s.site: s.ucap for s in optimization_run.site_capacities}
        self.ocap = {s.site: s.ocap for s in optimization_run.site_capacities}
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

    def get_random_demands(self):
        self.demands = {}
        for d in self.weekly_demands:
            self.demands[d.cust_prod.cust,d.cust_prod.prod,d.week] = np.random.normal(d.quantity, 0.1*d.quantity)

    def get_random_yields(self):
        self.yields = {}
        for n in self.sites:
            for t in range(self.num_weeks):
                self.yields[n,t] = np.random.uniform(0.9,1)
