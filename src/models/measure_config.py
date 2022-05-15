# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""blueprint for datamodel Configuration Limits """
from __future__ import annotations
from typing import List
import datetime
from sqlalchemy_utils import UUIDType
from dataclasses import dataclass
from dataclasses import field
from sqlalchemy import Column, Integer, String, Boolean, MetaData, Float, DateTime
from sqlalchemy.orm import registry, relationship
from src.models.measure import Measure
import uuid

mapper_registry = registry()
metadata = MetaData()

'''
createddate timestamp NULL,
isdeleted bool NULL,
"name" varchar(80) NULL,
systemmodstamp timestamp NULL,
limitname__c varchar(18) NULL,
max_value__c float8 NULL,
remaining_value__c float8 NULL,
smape float8 NULL,
model_name varchar(255) NULL,
rmse float8 NULL,
sfid varchar(18) NULL COLLATE "ucs_basic",
id serial NOT NULL,
"_hc_lastop" varchar(32) NULL,
"_hc_err" text NULL,
externalid__c varchar(255) NULL,
last_train_time timestamp NULL,
CONSTRAINT measure__c_pkey PRIMARY KEY (id)
'''


@mapper_registry.mapped
@dataclass
class MeasureConfig():
    # __slots__ = ('orgid__c')
    __tablename__ = "measureconfig__c"
    __table_args__ = {"schema": "salesforce"}

    __sa_dataclass_metadata_key__ = "sa"
    mapper_registry.metadata
    # id: int = field(
    #     init=False, metadata={"sa": Column(Integer, primary_key=True)}
    # )
    frequency: str = field(default=None, metadata={
        "sa": Column('frequency__c', String(255))})
    source: str = field(default=None, metadata={
        "sa": Column('source__c', String(255))})

    model_name: str = field(default=None, metadata={
        "sa": Column('model_name__c', String(255))})
    model_path: str = field(default=None, metadata={
        "sa": Column('model_path__c', String(255))})

    smape: float = field(default=None, metadata={
        "sa": Column('smape__c', Float)})
    rmse: float = field(default=None, metadata={
        "sa": Column('rmse__c', Float)})

    last_train_time: datetime = field(default_factory=lambda: datetime.datetime.utcnow, metadata={
        "sa": Column('last_train_time__c', DateTime)})

    name: str = field(default=None, metadata={
        "sa": Column('name', String(80), primary_key=True)})

    externalid: uuid.UUID = field(default_factory=uuid.uuid4, metadata={
        "sa": Column('externalid__c', UUIDType(), default=uuid.uuid4)})

    max_value: int = field(default=None, metadata={
                           "sa": Column('max_value__c', Integer)})
    query: str = field(default=None, metadata={
        "sa": Column('query__c', String(255))})
    active: str = field(default=None, metadata={
        "sa": Column('active__c', Boolean)})
    sfid: str = field(default=None, metadata={
        "sa": Column('sfid', String(18))})

    # sfid: str = field(default=None, metadata={
    #     "sa": Column('sfid', String(18), primary_key=True)})

    # eventrestlimits = relationship(
    #     "Measure", back_populates="measureconfig__c")
    # eventrestlimits = relationship("Measure", primaryjoin="and_(MeasureConfig.sfid==Measure.limitname)",
    #                                backref="measureconfig__c")
    # eventrestlimits: List[Measure] = field(
    #     default_factory=list, metadata={"sa": relationship("Measure")}
    # )

    # eventrestlimits: List[Measure] = field(default_factory=list)
    # __mapper_args__ = {   # type: ignore
    #     "properties": {
    #         "eventrestlimits": relationship("Measure")
    #     }
    # }

    # eventrestlimits: List[Measure] = field(
    #     default_factory=list, metadata={"sa": relationship("Measure")}
    # )
