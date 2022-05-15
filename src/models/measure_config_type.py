# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""blueprint for datamodel Configuration Limits """
from __future__ import annotations
from typing import List
from sqlalchemy_utils import UUIDType
from dataclasses import dataclass
from dataclasses import field
from sqlalchemy import Column, Integer, String, Boolean, MetaData
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
	externalid__c varchar(255) NULL,
	active__c bool NULL,
	sfid varchar(18) NULL COLLATE "ucs_basic",
	id serial4 NOT NULL,
	"_hc_lastop" varchar(32) NULL,
	"_hc_err" text NULL,
	CONSTRAINT measure_config_type__c_pkey PRIMARY KEY (id)
'''


@mapper_registry.mapped
@dataclass
class MeasureConfigType():
    # __slots__ = ('orgid__c')
    __tablename__ = "measure_config_type__c"
    __table_args__ = {"schema": "salesforce"}

    __sa_dataclass_metadata_key__ = "sa"
    mapper_registry.metadata
    name: str = field(default=None, metadata={
        "sa": Column('name', String(80), primary_key=True)})

    id: int = field(
        init=False, metadata={"sa": Column(Integer, primary_key=True)}
    )
    externalid: uuid.UUID = field(default_factory=uuid.uuid4, metadata={
        "sa": Column('externalid__c', UUIDType(), default=uuid.uuid4)})

    active: str = field(default=None, metadata={
        "sa": Column('active__c', Boolean)})
    sfid: str = field(default=None, metadata={
        "sa": Column('sfid', String(18))})
