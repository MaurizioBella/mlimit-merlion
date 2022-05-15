-- Run this for Salesforce schema
DROP SCHEMA IF EXISTS salesforce CASCADE;
DROP TABLE IF EXISTS salesforce.measureconfig__c CASCADE;
DROP TABLE IF EXISTS salesforce.forecaster__c CASCADE;
DROP TABLE IF EXISTS salesforce.holiday__c CASCADE;
DROP TABLE IF EXISTS salesforce.measure__c CASCADE;
DROP TABLE IF EXISTS salesforce.measure_config_type__c CASCADE;
DROP TABLE IF EXISTS salesforce.configauth__c CASCADE;


CREATE SCHEMA salesforce;

CREATE TABLE salesforce.measure_config_type__c (
	createddate timestamp NULL,
	isdeleted bool NULL,
	"name" varchar(80) NULL,
	systemmodstamp timestamp NULL,
	externalid__c varchar(255) NULL,
	active__c bool NULL,
	sfid varchar(18) NULL COLLATE "ucs_basic",
	id serial NOT NULL,
	"_hc_lastop" varchar(32) NULL,
	"_hc_err" text NULL,
	CONSTRAINT measure_config_type__c_pkey PRIMARY KEY (id)
);
CREATE INDEX hc_idx_measure_config_type__c_systemmodstamp ON salesforce.measure_config_type__c USING btree (systemmodstamp);
CREATE UNIQUE INDEX hcu_idx_measure_config_type__c_externalid__c ON salesforce.measure_config_type__c USING btree (externalid__c);
CREATE UNIQUE INDEX hcu_idx_measure_config_type__c_sfid ON salesforce.measure_config_type__c USING btree (sfid);

CREATE TABLE salesforce.measureconfig__c (
	frequency__c varchar(255) NULL,
	source__c varchar(255) NULL,
	"name" varchar(80) NOT NULL,
	max_value__c float8 NULL,
	isdeleted bool NULL,
	systemmodstamp timestamp NULL,
	createddate timestamp NULL,
    last_train_time__c timestamp NULL,
    smape__c float8 NULL,
    rmse__c float8 NULL,
    model_name__c varchar(255) NULL,
    model_path__c varchar(255) NULL,
    externalid__c varchar(255) NULL,
	query__c varchar(255) NULL,
	active__c bool NULL,
	sfid varchar(18) NULL COLLATE "ucs_basic",
	"_hc_lastop" varchar(32) NULL,
	"_hc_err" text NULL,
	CONSTRAINT measureconfig__c_pkey PRIMARY KEY ("name")
);
CREATE INDEX hc_idx_measureconfig__c_systemmodstamp ON salesforce.measureconfig__c USING btree (systemmodstamp);
CREATE UNIQUE INDEX hcu_idx_measureconfig__c_sfid ON salesforce.measureconfig__c USING btree (sfid);

CREATE TABLE salesforce.measure__c (
	"name" varchar(80) NULL,
	max_value__c float8 NULL,
	isdeleted bool NULL,
	systemmodstamp timestamp NULL,
	createddate timestamp NULL,
	remaining_value__c float8 NULL,
	eventdate__c timestamp NULL,
	externalid__c varchar(255) NULL,
	limitname__c varchar(18) NULL,
	sfid varchar(18) NULL COLLATE "ucs_basic",
	id serial NOT NULL,
	"_hc_lastop" varchar(32) NULL,
	"_hc_err" text NULL,
	CONSTRAINT measure__c_pkey PRIMARY KEY (id)
);
CREATE INDEX hc_idx_measure__c_systemmodstamp ON salesforce.measure__c USING btree (systemmodstamp);
CREATE UNIQUE INDEX hcu_idx_measure__c_externalid__c ON salesforce.measure__c USING btree (externalid__c);
CREATE UNIQUE INDEX hcu_idx_measure__c_sfid ON salesforce.measure__c USING btree (sfid);


CREATE TABLE salesforce.forecaster__c (
	anomaly__c float8 NULL,
	yhat__c float8 NULL,
	multiplicative_terms_upper__c float8 NULL,
	multiplicative_terms_lower__c float8 NULL,
	trend_upper__c float8 NULL,
	max_limit__c float8 NULL,
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
	ds__c timestamp NULL,
	createddate timestamp NULL,
	fact__c float8 NULL,
	externalid__c varchar(255) NULL,
	limitname__c varchar(18) NULL,
	yhat_upper__c float8 NULL,
	yhat_lower__c float8 NULL,
	sfid varchar(18) NULL COLLATE "ucs_basic",
	id serial NOT NULL,
	"_hc_lastop" varchar(32) NULL,
	"_hc_err" text NULL,
	CONSTRAINT forecaster__c_pkey PRIMARY KEY (id)
);
CREATE INDEX hc_idx_forecaster__c_systemmodstamp ON salesforce.forecaster__c USING btree (systemmodstamp);
CREATE UNIQUE INDEX hcu_idx_forecaster__c_externalid__c ON salesforce.forecaster__c USING btree (externalid__c);
CREATE UNIQUE INDEX hcu_idx_forecaster__c_sfid ON salesforce.forecaster__c USING btree (sfid);

CREATE TABLE salesforce.holiday__c (
	createddate timestamp NULL,
	isdeleted bool NULL,
	"name" varchar(80) NULL,
	systemmodstamp timestamp NULL,
	ds__c date NULL,
	limitname__c varchar(18) NULL,
	holiday__c varchar(255) NULL,
	sfid varchar(18) NULL COLLATE "ucs_basic",
	id serial NOT NULL,
	"_hc_lastop" varchar(32) NULL,
	"_hc_err" text NULL,
	CONSTRAINT holiday__c_pkey PRIMARY KEY (id)
);
CREATE INDEX hc_idx_holiday__c_systemmodstamp ON salesforce.holiday__c USING btree (systemmodstamp);
CREATE UNIQUE INDEX hcu_idx_holiday__c_sfid ON salesforce.holiday__c USING btree (sfid);


CREATE TABLE salesforce.configauth__c (
	heroku_app_name__c varchar(255) NULL,
    orgid__c varchar(18) NULL,
    token_type__c varchar(255) NULL,
	access_token__c varchar(255) NULL,
	"name" varchar(80) NULL,
	refresh_token__c varchar(255) NULL,
	scope__c varchar(255) NULL,
	isdeleted bool NULL,
	systemmodstamp timestamp NULL,
	oauth_signature__c varchar(255) NULL,
	createddate timestamp NULL,
	issued_at__c timestamp NULL,
	userid__c varchar(255) NULL,
	externalid__c varchar(255) NULL,
	active__c bool NULL,
	instance_url__c varchar(255) NULL,
	sfid varchar(18) NULL COLLATE "ucs_basic",
	id serial NOT NULL,
	"_hc_lastop" varchar(32) NULL,
	"_hc_err" text NULL,
	CONSTRAINT configauth__c_pkey PRIMARY KEY (id)
);
CREATE INDEX hc_idx_configauth__c_systemmodstamp ON salesforce.configauth__c USING btree (systemmodstamp);
CREATE UNIQUE INDEX hcu_idx_configauth__c_externalid__c ON salesforce.configauth__c USING btree (externalid__c);
CREATE UNIQUE INDEX hcu_idx_configauth__c_sfid ON salesforce.configauth__c USING btree (sfid);