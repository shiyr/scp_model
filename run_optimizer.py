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

    file = pd.ExcelFile(filename)
    ref_produce = pd.read_excel(filename, 'to_produce')
    ref_ship = pd.read_excel(filename, 'to_ship')
    ref_inv = pd.read_excel(filename, 'inv')
    ref_db = pd.read_excel(filename, 'backlog')
    ref_ot = pd.read_excel(filename, 'ot')
    ref_ut = pd.read_excel(filename, 'ut')
    data.get_phase_one_results(ref_produce, ref_ship, ref_inv, ref_db, ref_ot, ref_ut)
    logger.info("finished loading phase 1 data")

    results = pd.DataFrame(columns=['objs', 'pens', 'invs', 'trans', 'uts', 'ots', 'fixs', 'caps'])
    for i in range(10):
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
        optimizer.fix_phase_one_variables(data)
        logger.info("finished loading phase 1 data")
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

