# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""blueprint for datamodel Prophet Holiday """
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import datetime
from sqlalchemy.sql.sqltypes import Date
from sqlalchemy import Column, ForeignKey, Integer, String, MetaData
from sqlalchemy.orm import registry
from src.models.measure_config import MeasureConfig

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
CONSTRAINT measure__c_pkey PRIMARY KEY (id)
'''


@mapper_registry.mapped
@dataclass()
class Holiday:
    # __slots__ = ('orgid__c')
    __tablename__ = "holiday__c"
    __table_args__ = {"schema": "salesforce"}
    __sa_dataclass_metadata_key__ = "sa"
    id: int = field(
        init=False, metadata={"sa": Column(Integer, primary_key=True)}
    )
    ds: datetime.datetime = field(
        default=None, metadata={"sa": Column('ds__c', Date)})
    holiday: str = field(default=None, metadata={
        "sa": Column('holiday__c', String(255))})

    limitname: str = field(default=None, metadata={"sa": Column('limitname__c', String(18), ForeignKey(
        MeasureConfig.sfid))})
    sfid: str = field(default=None, metadata={
        "sa": Column('sfid', String(18))})
