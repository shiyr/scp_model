import sys
import logging
import gc
from pandas import ExcelWriter, DataFrame
import data_model as dm
from production_data import ProductionData
from scp_optimizer import SCPOptimizer
from utils import logged
from db_utils import get_session, get_latest_optimization_id
from grb_utils import UnexpectedInfeasibleModel

logger = logging.getLogger('run_optimizer')


@logged(logger=logging.getLogger("run_optimizer"))
def run_optimizer(run_id=None):
    session = get_session()
    if run_id is None:
        run_id = get_latest_optimization_id(session, dm)
    
    optimization_run = session.query(dm.OptimizationRun).get(run_id)
    if optimization_run is None:
        raise RuntimeError("cannot find run_id %d in database" % (run_id))

    run_optimizer_from_object(optimization_run, session)


def run_optimizer_from_object(optimization_run, session):
    data = ProductionData(optimization_run)
    logger.info("created production data object")
    try:
        logger.info("creating optimizer object")
        optimizer = SCPOptimizer(data)
        optimizer.optimize_objective()
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

    try:
        run_optimizer(run_id=run_id)
    except Exception as e:
        logger.exception(e)
        raise
    else:
        logger.info("exiting normally")

