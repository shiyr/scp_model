import sys
import logging
import gc
import pandas as pd
import data_model as dm
from production_data import ProductionData
from scp_optimizer import SCPOptimizer
from utils import logged
from db_utils import get_session, get_latest_optimization_id
from grb_utils import UnexpectedInfeasibleModel

logger = logging.getLogger('run_optimizer')


@logged(logger=logging.getLogger("run_optimizer"))
def run_optimizer(run_id=None, filename=None):
    session = get_session()
    if run_id is None:
        run_id = get_latest_optimization_id(session, dm)
    
    optimization_run = session.query(dm.OptimizationRun).get(run_id)
    if optimization_run is None:
        raise RuntimeError("cannot find run_id %d in database" % (run_id))
    
    data = ProductionData(optimization_run)
    logger.info("created production data object")

    ref_inv = pd.read_csv(filename+'_inv.csv')
    ref_inv = ref_inv[ref_inv['week'] == 2]
    ref_db = pd.read_csv(filename+'_db.csv')
    ref_db = ref_db[ref_db['week'] == 2]
    ref_rec = pd.read_csv(filename+'_rec.csv')
    ref_rec = ref_rec[ref_rec['week'] <= 2]
    for (n,k), val in data.start_inv.iteritems():
        qty = ref_inv[(ref_inv['site'] == n.key_name) & (ref_inv['prod'] == k.key_name)].iloc[0].quantity
        data.start_inv[n,k] = qty
    for (n,k), val in data.start_backlog.iteritems():
        qty = ref_db[(ref_db['cust'] == n.key_name) & (ref_db['prod'] == k.key_name)].iloc[0].quantity
        data.start_backlog[n,k] = qty
    for (i,j,k,m,t), val in data.start_to_rec.iteritems():
        lt = data.lead_times[i,j,m]
        s = t + 3 - lt
        if s < 3 and s >= 0:
            qty = ref_rec[(ref_rec['origin'] == i.key_name) & (ref_rec['destination'] == j.key_name) & (ref_rec['prod'] == k.key_name) & (ref_rec['mode'] == m.key_name) & (ref_rec['week'] == s)].iloc[0].quantity
            data.start_to_rec[i,j,k,m,t] = qty
    logger.info("finished loading stage 1 data")

    results = pd.DataFrame(columns=['objs', 'pens', 'invs', 'trans', 'uts', 'ots', 'fixs', 'caps'])
    for i in range(1000):
        data.get_random_demands()
        data.get_random_yields()
        results.loc[i] = run_optimizer_from_object(data)
    results.to_csv('output_'+filename+'.csv')
    logger.info("finished writing output to csv")
    print results


def run_optimizer_from_object(data):
    try:
        logger.info("creating optimizer object")
        optimizer = SCPOptimizer(data)
        return optimizer.optimize_objective()
    except UnexpectedInfeasibleModel:
        raise



if __name__ == '__main__':
    logger = logging.getLogger()
    logger.info("turning garbage collection off")
    gc.disable()

    if len(sys.argv) > 1:
        run_id = int(sys.argv[1])
    else:
        run_id = None

    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = None

    try:
        run_optimizer(run_id, filename)
    except Exception as e:
        logger.exception(e)
        raise
    else:
        logger.info("exiting normally")

