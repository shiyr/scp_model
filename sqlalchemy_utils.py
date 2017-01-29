from itertools import groupby
from operator import attrgetter
from sqlalchemy import ForeignKeyConstraint, String
from sqlalchemy.types import TypeDecorator
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import logging

logger = logging.getLogger("sqlalchemy_utils")


def get_table_classes(datamodel):
    """
    returns a map from tablename to class
    assumes that only database classes have __tablename__ attribute
    """
    result = {}
    for name in dir(datamodel):
        cls = getattr(datamodel, name)
        try:
            tablename = cls.__tablename__
        except AttributeError:
            continue            # skip the remaining part of the for loop and go to next iteration
        result[tablename] = cls
    return result


def add_all(session, objects):
    """
    add all objects to the session
    """
    logger = logging.getLogger("add_all")
    if not isinstance(objects, list):
        objects = list(objects)
    num_objects = len(objects)
    if num_objects == 0:
        logger.info("0 objects being added to the database")
        return

    for o in objects:
        if hasattr(o, '__iter__'):
            add_all(session, o)
        else:
            logger.info("adding %d %s objects to database", len(objects), o.__class__.__name__)
            logger.debug("first object = %s", str(o))
            break
    else:
        logger.info("done adding list of objects to database")
        return

    try:
        session.commit()
        session.bulk_save_objects(objects)
        session.flush()
        session.commit()
    except SQLAlchemyError as ex:
        logger.fatal("error adding objects to database, adding individually: %s", ex)
        logger.fatal("%s", ex)
        session.rollback()
        logger.fatal("completed rollback of bulk add")
        for i, obj in enumerate(objects):
            logger.fatal("adding object %d of %d", i, len(objects))
            try:
                session.add(obj)
                session.flush()
                session.commit()
            except SQLAlchemyError:
                logger.fatal("error adding object %d of %d", i, num_objects)
                session.rollback()
                logger.fatal("failed object: %s", str(obj))
                logger.fatal("failed object type: %s", type(obj))
                if i > 0:
                    logger.fatal("previous ok object: %s", objects[i-1])
                raise
            except Exception as ex:
                logger.fatal("exception %s raised in individual adds", ex)
                raise
        raise

def versioned_parent_fk(parent_table, child_names, parent_names=None):
    """
    defines a foreign key constraint to a master table
    containing all allowed attribute name / attribute value pairs
    """
    child_names = ['run_id'] + child_names
    if parent_names is None:
        parent_names = child_names
    else:
        parent_names = ['run_id'] + parent_names
    parent_keys = ['{0}.{1}'.format(parent_table, name) for name in parent_names]
    return ForeignKeyConstraint(child_names, parent_keys)
