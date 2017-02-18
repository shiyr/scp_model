from __future__ import division
import sys
import logging
import gc
from pandas import ExcelWriter, DataFrame
import data_model as dm
from production_data import ProductionData
from scp_optimizer import MasterProblem, SubProblem
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
    
    sample_size = data.sample_size
    epsilon = 1e-5
    max_iterations = 1
    
    try:
        logger.info("creating optimizer object")
        optimizer = MasterProblem(data)
        mp_obj, mp_1_obj, receives, invs, backlogs = optimizer.optimize_objective()
        
        sub_optimizer = {}
        for w in range(1, sample_size+1):
            sub_optimizer[w] = SubProblem(data, w, receives, invs, backlogs)
    
        r = 0
        sub_obj = {}
        while r < max_iterations:
            print '################ iteration', r, '##################'
            for w in range(1, sample_size+1):
                sub_optimizer[w].update_model_parameters(receives, invs, backlogs)
                sub_obj[w], pi_flow, pi_rec, pi_db = sub_optimizer[w].optimize_objective()
                if optimizer.mp_cut[w] < sub_obj:
                    cut_val = sub_optimizer[w].add_cut_to_master_problem()
                    optimizer.add_cut_to_master_problem(w, r, pi_flow, pi_rec, pi_db, cut_val)
        
            mp_obj_est = mp_1_obj + sum(sub_obj.values()) / sample_size
            print '##################################################################'
            print 'MP objective upper bound estimate in iteration', r, 'is', mp_obj_est
            print 'MP objective lower bound estimate in iteration', r, 'is', mp_obj
            print '##################################################################'
            if mp_obj_est - mp_obj <= epsilon:
                print 'mp_obj_est =', mp_obj_est, 'mp_obj =', mp_obj, 'delta =', mp_obj_est - mp_obj
                print 'Within', epsilon, 'tolerance, break.'
                break
            
            r += 1
            mp_obj, mp_1_obj, receives, invs, backlogs = optimizer.optimize_objective()


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

