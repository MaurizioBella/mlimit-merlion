# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""blueprint for datamodel Event Rest Limits """
from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from sqlalchemy_utils import UUIDType
from sqlalchemy import Column, ForeignKey, DateTime, Integer, String, MetaData
from sqlalchemy.orm import registry, relationship
# from src.models.measure_config import MeasureConfig
from sqlalchemy import Table
from typing import Optional
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
sfid varchar(18) NULL COLLATE "ucs_basic",
id serial NOT NULL,
"_hc_lastop" varchar(32) NULL,
"_hc_err" text NULL,
externalid__c varchar(255) NULL,
eventdate__c timestamp NULL,
CONSTRAINT measure__c_pkey PRIMARY KEY (id)
'''


@mapper_registry.mapped
@dataclass
class Measure():
    # __slots__ = ('orgid__c')
    __tablename__ = "measure__c"
    __table_args__ = {"schema": "salesforce"}
    __sa_dataclass_metadata_key__ = "sa"
    mapper_registry.metadata

    id: int = field(
        init=False, metadata={"sa": Column(Integer, primary_key=True)}
    )
    eventdate: datetime = field(default_factory=lambda: datetime.datetime.utcnow, metadata={
        "sa": Column('eventdate__c', DateTime, default=datetime.datetime.utcnow)})

    externalid: uuid.UUID = field(default_factory=uuid.uuid4, metadata={
        "sa": Column('externalid__c', UUIDType(), default=uuid.uuid4)})

    # limitname: str = field(default=None, metadata={"sa": Column('limitname__c', String(18), ForeignKey(
    #     MeasureConfig.sfid))})
    # limitname: str = field(default=None, metadata={"sa": Column('limitname__c', String(18), ForeignKey(
    #     "salesforce.measureconfig__c.sfid"))})
    limitname: str = field(default=None, metadata={
                           "sa": Column('limitname__c', String(18))})
    max_value: int = field(default=None, metadata={
        "sa": Column('max_value__c', Integer)})
    remaining_value: int = field(default=None, metadata={
        "sa": Column('remaining_value__c', Integer)})
    # measureconfig = relationship("MeasureConfig", back_populates="measures")
