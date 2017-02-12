from sqlalchemy import Column, String, Integer, Float, DateTime, Table
from sqlalchemy import ForeignKey, ForeignKeyConstraint, Index, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, backref
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.associationproxy import association_proxy

import os
import logging
from utils import uncamel_case, as_repr, pluralize, get_parameter_value, lazy_property
from sqlalchemy_utils import versioned_parent_fk

logging.basicConfig(level=logging.INFO, format="%(relativeCreated)5d:%(levelname)-5s:%(name)-8s:%(message)s")
logger = logging.getLogger("production_data")


class DatabaseObject(object):
    @declared_attr
    def __tablename__(cls):
        return uncamel_case(cls.__name__)
    
    @property
    def columns(self):
        return [c.name for c in self.__table__.columns if c.name != 'run_id']

    @property
    def primary_key_names(self):
        return [c.name for c in self.__table__.primary_key.columns]

    @property
    def key_values(self):
        return [getattr(self, name) for name in self.primary_key_names if name != 'run_id']
    
    @property
    def key_name(self):
        return as_repr(*self.key_values)
    
    def __repr__(self):
        return as_repr(*self.key_values)

    def as_dict(self):
        return {c: getattr(self, c) for c in self.columns}

    def as_record(self):
        return [getattr(self, c) for c in self.columns]


Base = declarative_base(cls = DatabaseObject)


class OptimizationRun(Base):
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    description = Column(String(255), default='')
    parameters = relationship("Parameter", collection_class=attribute_mapped_collection('name'))

    @property
    def run_id(self):
        return self.id
    
    @lazy_property
    def num_weeks(self):
        return max(d.week for d in self.weekly_demands) + 1

    def log_parameters(self):
        logger.info("optimization run parameters")
        for parameter in self.parameters.itervalues():
            logger.info("%s: %s", parameter.name, parameter.value)


class Parameter(Base):
    run_id = Column(Integer, ForeignKey('optimization_run.id'), primary_key=True)
    name = Column(String, primary_key=True)
    _value = Column('value', String)

    @property
    def value(self):
        return get_parameter_value(self.name, self._value)

    @value.setter
    def value(self, new_value):
        self._value = str(new_value)


class VersionedMixin(object):
    @declared_attr
    def run_id(cls):
        return Column(None, ForeignKey('optimization_run.id'), primary_key=True, index=True)

    @declared_attr
    def optimization_run(cls):
        return relationship("OptimizationRun", backref=backref(pluralize(cls.__tablename__)), viewonly=True)


# Product set (Part, Finished Goods)
class Product(Base, VersionedMixin):
    id = Column(String(255), primary_key=True, nullable=False)
    type = Column(String(255))
    product_attributes = relationship("ProductAttribute",
                                      collection_class=attribute_mapped_collection('attr_name'),
                                      lazy='subquery')
    attributes = association_proxy('product_attributes', 'attr_value',
                                    creator=lambda n, v: ProductAttribute(attr_name=n, attr_value=v))


class ProductAttribute(Base, VersionedMixin):
    prod = Column(String(255), primary_key=True, nullable=False)
    attr_name = Column(String(255), primary_key=True, nullable=False)
    attr_value = Column(String(255), nullable=False)

    __table_args__ = (versioned_parent_fk('product', ['prod'], ['id']),)


# Node set (Supplier, Manufacturer, Hub, Customer)
class Node(Base, VersionedMixin):
    id = Column(String(255), primary_key=True, nullable=False)
    type = Column(String(255))
    node_attributes = relationship("NodeAttribute",
                                   collection_class=attribute_mapped_collection('attr_name'),
                                   lazy='subquery')
    attributes = association_proxy('node_attributes', 'attr_value',
                                    creator=lambda n, v: NodeAttribute(attr_name=n, attr_value=v))


class NodeAttribute(Base, VersionedMixin):
    node = Column(String(255), primary_key=True, nullable=False)
    attr_name = Column(String(255), primary_key=True, nullable=False)
    attr_value = Column(String(255))

    __table_args__ = (versioned_parent_fk('node', ['node'], ['id']),)


class SiteCapacity(Base, VersionedMixin):
    site_id = Column(String(255), primary_key=True)
    pcap = Column(Float, default=0)
    ucap = Column(Float, default=0)
    ocap = Column(Float, default=0)
    lcap = Column(Integer, default=0)
    icap = Column(Float, default=0)
    fcost = Column(Float, default=0)
    ucost = Column(Float, default=0)
    ocost = Column(Float, default=0)
    wcost = Column(Float, default=0)

    __table_args__ = (versioned_parent_fk('node', ['site_id'], ['id']),)

    site = relationship("Node", viewonly=True, lazy='subquery')


# Transportation Mode set
class Mode(Base, VersionedMixin):
    id = Column(String(255), primary_key=True, nullable=False)


# BOM mapping table
class BOM(Base, VersionedMixin):
    input_id = Column(String(255), primary_key=True, nullable=False)
    output_id = Column(String(255), primary_key=True, nullable=False)

    __table_args__ = (versioned_parent_fk('product', ['input_id'], ['id']),
                      versioned_parent_fk('product', ['output_id'], ['id']))

    input = relationship("Product", foreign_keys= '[BOM.run_id, BOM.input_id]', backref='outputs', lazy='subquery')
    output = relationship("Product", foreign_keys= '[BOM.run_id, BOM.output_id]', backref='inputs', lazy='subquery')


# production mapping table
class SiteProdProduce(Base, VersionedMixin):
    site_id = Column(String(255), primary_key=True, nullable=False)
    prod_id = Column(String(255), primary_key=True, nullable=False)

    __table_args__ = (versioned_parent_fk('node', ['site_id'], ['id']),
                      versioned_parent_fk('product', ['prod_id'], ['id']))

    site = relationship("Node", backref='produced_prods', viewonly=True, lazy='subquery')
    prod = relationship("Product", backref='produced_in_sites', viewonly=True, lazy='subquery')


# inventory storage mapping table
class SiteProdInv(Base, VersionedMixin):
    site_id = Column(String(255), primary_key=True, nullable=False)
    prod_id = Column(String(255), primary_key=True, nullable=False)
    start_inv = Column(Float, default=0)
    icost = Column(Float, default=0)

    __table_args__ = (versioned_parent_fk('node', ['site_id'], ['id']),
                      versioned_parent_fk('product', ['prod_id'], ['id']))

    site = relationship("Node", backref='stored_prods', viewonly=True, lazy='subquery')
    prod = relationship("Product", backref='stored_in_sites', viewonly=True, lazy='subquery')


# valid transportation route mapping
class Edge(Base, VersionedMixin):
    origin_id = Column(String(255), primary_key=True, nullable=False, index=True)
    destination_id = Column(String(255), primary_key=True, nullable=False, index=True)
    mode_id = Column(String(255), primary_key=True, nullable=False, index=True)
    lead_time = Column(Integer, default=0)
    tcap = Column(Float, default=0)
    
    __table_args__ = (versioned_parent_fk('node', ['origin_id'], ['id']),
                      versioned_parent_fk('node', ['destination_id'], ['id']),
                      versioned_parent_fk('mode', ['mode_id'], ['id']))
    
    origin = relationship("Node", foreign_keys='[Edge.run_id, Edge.origin_id]', viewonly=True, lazy='subquery')
    destination = relationship("Node", foreign_keys='[Edge.run_id, Edge.destination_id]', viewonly=True, lazy='subquery')
    mode = relationship("Mode", viewonly=True, lazy='joined')


# shipment mapping table
class Lane(Base, VersionedMixin):
    id = Column(Integer, primary_key=True, nullable=False)
    origin_id = Column(String(255), nullable=False)
    destination_id = Column(String(255), nullable=False)
    prod_id = Column(String(255), nullable=False)
    mode_id = Column(String(255), nullable=False)
    tcost = Column(Float, default=0)

    __table_args__ = (versioned_parent_fk('edge', ['origin_id', 'destination_id', 'mode_id']),
                      versioned_parent_fk('node', ['origin_id'], ['id']),
                      versioned_parent_fk('node', ['destination_id'], ['id']),
                      versioned_parent_fk('product', ['prod_id'], ['id']),
                      versioned_parent_fk('mode', ['mode_id'], ['id']))

    origin = relationship("Node", foreign_keys='[Lane.run_id, Lane.origin_id]', viewonly=True, lazy='subquery')
    destination = relationship("Node", foreign_keys='[Lane.run_id, Lane.destination_id]', viewonly=True, lazy='subquery')
    prod = relationship("Product", viewonly=True, lazy='subquery')
    mode = relationship("Mode", viewonly=True, lazy='subquery')
    weekly_data = relationship("WeeklyReceive", lazy='subquery', backref='lane')


class WeeklyReceive(Base, VersionedMixin):
    lane_id = Column(Integer, primary_key=True, nullable=False)
    week = Column(Integer, primary_key=True, nullable=False)
    start_to_rec = Column(Float, default=0)

    __table_args__ = (versioned_parent_fk('lane', ['lane_id'], ['id']),)


# customer product mapping
class CustProd(Base, VersionedMixin):
    id = Column(Integer, primary_key=True, nullable=False)
    cust_id = Column(String(255), nullable=False)
    prod_id = Column(String(255), nullable=False)
    start_backlog = Column(Float, default=0)
    pcost = Column(Float, default=0)

    cust = relationship("Node", backref='need_prods', viewonly=True, lazy='subquery')
    prod = relationship("Product", backref='sold_to_custs', viewonly=True, lazy='subquery')

    weekly_data = relationship("WeeklyDemand", lazy='subquery', backref='cust_prod')

    __table_args__ = (versioned_parent_fk('node', ['cust_id'], ['id']),
                      versioned_parent_fk('product', ['prod_id'], ['id']))


class WeeklyDemand(Base, VersionedMixin):
    sample_id = Column(Integer, primary_key=True, nullable=False)
    cust_prod_id = Column(Integer, primary_key=True, nullable=False)
    flavor = Column(String(255), primary_key=True, nullable=False)
    week = Column(Integer, primary_key=True, nullable=False)
    quantity = Column(Float, default=0)

    __table_args__ = (versioned_parent_fk('cust_prod', ['cust_prod_id'], ['id']),)


class WeeklyYield(Base, VersionedMixin):
    sample_id = Column(Integer, primary_key=True, nullable=False)
    site_id = Column(String(255), primary_key=True, nullable=False)
    week = Column(Integer, primary_key=True, nullable=False)
    quantity = Column(Float, default=0)

    site = relationship("Node", viewonly=True)

    __table_args__ = (versioned_parent_fk('node', ['site_id'], ['id']),)


tables_in_order = [t for t in OptimizationRun.metadata.sorted_tables
                   if t.name not in ('optimization_run', 'log_message')]

print 'tables_in_order', [t.name for t in tables_in_order]
