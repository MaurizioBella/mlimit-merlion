# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""blueprint for datamodel Prophet Daily """
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import datetime
from sqlalchemy_utils import UUIDType
from sqlalchemy import Column, ForeignKey, DateTime, Integer, String, MetaData
from sqlalchemy.orm import registry
from src.models.measure_config import MeasureConfig
import uuid

mapper_registry = registry()
metadata = MetaData()

'''
anomaly__c float8 NULL,
-- yhat__c float8 NULL,
-- multiplicative_terms_upper__c float8 NULL,
-- multiplicative_terms_lower__c float8 NULL,
-- trend_upper__c float8 NULL,
-- max_limit__c float8 NULL,
trend_lower__c float8 NULL,
multiplicative_terms__c float8 NULL,
trend__c float8 NULL,
"name" varchar(80) NULL,
additive_terms_upper__c float8 NULL,
additive_terms_lower__c float8 NULL,
additive_terms__c float8 NULL,
anomaly_weight__c float8 NULL,
isdeleted bool NULL,
systemmodstamp timestamp NULL,
-- ds__c timestamp NULL,
createddate timestamp NULL,
fact__c float8 NULL,
-- limitname__c varchar(18) NULL,
-- yhat_upper__c float8 NULL,
-- yhat_lower__c float8 NULL,
-- sfid varchar(18) NULL COLLATE "ucs_basic",
-- id serial NOT NULL,

-- externalid__c varchar(255) NULL,
CONSTRAINT forecaster__c_pkey PRIMARY KEY (id)
);
'''


@mapper_registry.mapped
@dataclass()
class Forecaster:
    # __slots__ = ('orgid__c')
    __tablename__ = "forecaster__c"
    __table_args__ = {"schema": "salesforce"}
    __sa_dataclass_metadata_key__ = "sa"
    id: int = field(
        init=False, metadata={"sa": Column(Integer, primary_key=True)}
    )
    ds: datetime.datetime = field(
        default=None, metadata={"sa": Column('ds__c', DateTime)})
    externalid: uuid.UUID = field(default_factory=uuid.uuid4, metadata={
        "sa": Column('externalid__c', UUIDType(), default=uuid.uuid4)})
    limitname: str = field(default=None, metadata={"sa": Column('limitname__c', String(18), ForeignKey(
        MeasureConfig.sfid))})
    yhat: int = field(default=None, metadata={
        "sa": Column('yhat__c', Integer)})
    yhat_upper: int = field(default=None, metadata={
        "sa": Column('yhat_upper__c', Integer)})
    yhat_lower: int = field(default=None, metadata={
        "sa": Column('yhat_lower__c', Integer)})
    sfid: str = field(default=None, metadata={
        "sa": Column('sfid', String(18))})
    trend_upper: int = field(default=None, metadata={
        "sa": Column('trend_upper__c', Integer)})
    multiplicative_terms_upper: int = field(default=None, metadata={
        "sa": Column('multiplicative_terms_upper__c', Integer)})
    multiplicative_terms_lower: int = field(default=None, metadata={
        "sa": Column('multiplicative_terms_lower__c', Integer)})
    max_limit: int = field(default=None, metadata={
        "sa": Column('max_limit__c', Integer)})
    anomaly: int = field(default=None, metadata={
        "sa": Column('anomaly__c', Integer)})
    trend_lower: int = field(default=None, metadata={
        "sa": Column('trend_lower__c', Integer)})
    multiplicative_terms: int = field(default=None, metadata={
        "sa": Column('multiplicative_terms__c', Integer)})
    trend: int = field(default=None, metadata={
        "sa": Column('trend__c', Integer)})
    name: str = field(default=None, metadata={
        "sa": Column('name', String(80))})

    additive_terms_upper: int = field(default=None, metadata={
        "sa": Column('additive_terms_upper__c', Integer)})
    additive_terms_lower: int = field(default=None, metadata={
        "sa": Column('additive_terms_lower__c', Integer)})
    additive_terms: int = field(default=None, metadata={
        "sa": Column('additive_terms__c', Integer)})
    anomaly_weight: int = field(default=None, metadata={
        "sa": Column('anomaly_weight__c', Integer)})
    fact: int = field(default=None, metadata={
        "sa": Column('fact__c', Integer)})
