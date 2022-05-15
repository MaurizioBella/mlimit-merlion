# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""blueprint for datamodel Salesforce Authentication."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from sqlalchemy import DateTime, Column, String, Boolean, MetaData, Integer
from sqlalchemy.orm import registry
from sqlalchemy_utils import UUIDType
import uuid
mapper_registry = registry()
metadata = MetaData()

'''
token_type__c varchar(255) NULL,
access_token__c varchar(255) NULL,
"name" varchar(80) NULL,
refresh_token__c varchar(255) NULL,
scope__c varchar(255) NULL,
orgid__c varchar(18) NULL,
oauth_signature__c varchar(255) NULL,
issued_at__c timestamp NULL,
userid__c varchar(255) NULL,
externalid__c varchar(255) NULL,
active__c bool NULL,
instance_url__c varchar(255) NULL,
sfid varchar(18) NULL COLLATE "ucs_basic",
id serial4 NOT NULL,
'''


@mapper_registry.mapped
@dataclass()
class SfdcAuth:
    # __slots__ = ('orgid__c')
    __tablename__ = "configauth__c"
    __table_args__ = {"schema": "salesforce"}
    __sa_dataclass_metadata_key__ = "sa"
    # id: int = field(
    #     init=False, metadata={"sa": Column(Integer, primary_key=True)}
    # )

    heroku_app_name: str = field(default=None, metadata={
        "sa": Column('heroku_app_name__c', String(120))})
    orgid: str = field(default=None, metadata={
        "sa": Column('orgid__c', String(18))})        
    access_token: str = field(default=None, metadata={
        "sa": Column('access_token__c', String(120))})
    refresh_token: str = field(default=None, metadata={
        "sa": Column('refresh_token__c', String(120))})
    oauth_signature: str = field(default=None, metadata={
        "sa": Column('oauth_signature__c', String(50))})
    token_type: str = field(default=None, metadata={
        "sa": Column('token_type__c', String(10))})
    scope: str = field(default=None, metadata={
        "sa": Column('scope__c', String(255))})
    active: str = field(default=None, metadata={
        "sa": Column('active__c', Boolean)})
    externalid: uuid.UUID = field(default_factory=uuid.uuid4, metadata={
        "sa": Column('externalid__c', UUIDType(), default=uuid.uuid4)})
    instance_url: str = field(default=None, metadata={
        "sa": Column('instance_url__c', String(255))})
    userid: str = field(default=None, metadata={
        "sa": Column('userid__c', String(255))})
    id: int = field(
        init=False, metadata={"sa": Column(Integer, primary_key=True)}
    )
    issued_at: DateTime = field(default=None, metadata={
        "sa": Column('issued_at__c', DateTime(timezone=True))})
