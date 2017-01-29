from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.interfaces import PoolListener
import utils
import logging

logger = logging.getLogger("db_utils")


current_session_makers = {}
def get_session(uri = None, dm = None):
    logger = logging.getLogger("get_session")
    config = utils.get_config()
    if not uri:
        uri = config['database']['uri']
    logger.info("current_session_makers size = %d", len(current_session_makers))

    if uri in current_session_makers:
        logger.debug("re-using session maker for db %s", uri)
        Session = current_session_makers[uri]
    else:
        logger.debug("%s not in [%s]", uri, ".".join(current_session_makers.keys()))
        engine_args = utils.get_config_value(['database', 'engine'], {})
        logger.info("database engine args: %s", engine_args)

        engine = create_engine(uri, pool_recycle = 7200, echo = False, listeners=[MyListener()])
        print 'created engine', uri
        if utils.get_config_value(['database', 'text_factory_str'], False):
            engine.raw_connection().connection.text_factory = str
        if dm:
            print 'start creating tables in database'
            dm.Base.metadata.create_all(engine)
            print 'created all tables in database'
        Session = scoped_session(sessionmaker(bind = engine, expire_on_commit = False))
        current_session_makers[uri] = Session
    session = Session()
    return session

def get_latest_optimization_id(session, dm):
    """
    Raises exception if no optimization_run records in db
    """
    run_id = session.query(dm.OptimizationRun.id).order_by(
                           dm.OptimizationRun.created_at.desc()).limit(1).scalar()
    if run_id is None:
        raise Exception("Could not find any optimizer data in database.")
    return run_id

class MyListener(PoolListener):
    def connect(self, dbapi_con, con_record):
        db_cursor = dbapi_con.execute('pragma foreign_keys=ON')