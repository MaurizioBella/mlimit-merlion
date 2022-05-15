INSERT INTO salesforce.measureconfig__c ("name",max_value,frequency,"source",active,eventdate,query) VALUES
	 ('ActiveScratchOrgs',3,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('AnalyticsExternalDataSizeMB',40960,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('BOZosCalloutHourlyLimit',20000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('ConcurrentAsyncGetReportInstances',200,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('ConcurrentEinsteinDataInsightsStoryCreation',5,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('ConcurrentEinsteinDiscoveryStoryCreation',2,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('ConcurrentSyncReportRuns',20,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('CurrentPendingServiceRoutings',200000,'H','soql_count',0,'2021-01-09 01:02:16','SELECT count() FROM PendingServiceRouting'),
	 ('DataStorageMB',47450,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('FileStorageMB',135318,'H','restapi',0,'2021-01-09 01:02:16.655204','');
INSERT INTO salesforce.measureconfig__c ("name",max_value,frequency,"source",active,eventdate,query) VALUES
	 ('AsyncApexJobQueued',999999,'H','soql_count',0,'2021-01-09 01:02:16','SELECT count() FROM AsyncApexJob WHERE status = ''queued'''),
	 ('MassEmail',5000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('PermissionSets',1500,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('SingleEmail',100,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('StreamingApiConcurrentClients',1000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyDashboardStatuses',999999999,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyODataCallout',200000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyPublishedPlatformEvents',250000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyPublishedStandardVolumePlatformEvents',100000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyTimeBasedWorkflow',50,'H','restapi',0,'2021-01-09 01:02:16.655204','');
INSERT INTO salesforce.measureconfig__c ("name",max_value,frequency,"source",active,eventdate,query) VALUES
	 ('DailyApiRequests',135000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyAsyncApexExecutions',661000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyBulkApiBatches',15000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyBulkV2QueryFileStorageMB',976562,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyBulkV2QueryJobs',10000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyDurableGenericStreamingApiEvents',200000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyDurableStreamingApiEvents',200000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyStandardVolumePlatformEvents',25000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyStreamingApiEvents',200000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DurableStreamingApiConcurrentClients',1000,'H','restapi',0,'2021-01-09 01:02:16.655204','');
INSERT INTO salesforce.measureconfig__c ("name",max_value,frequency,"source",active,eventdate,query) VALUES
	 ('MonthlyEinsteinDiscoveryStoryCreation',500,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('MonthlyPlatformEventsUsageEntitlement',750000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('Package2VersionCreates',6,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('Package2VersionCreatesWithoutValidation',500,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyAsyncReportRuns',1200,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyDashboardRefreshes',200,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyDashboardResults',5000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyLongTermIdMapping',100000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyManagedContentPublicRequests',50000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('HourlyShortTermIdMapping',100000,'H','restapi',0,'2021-01-09 01:02:16.655204','');
INSERT INTO salesforce.measureconfig__c ("name",max_value,frequency,"source",active,eventdate,query) VALUES
	 ('HourlySyncReportRuns',500,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyAnalyticsDataflowJobExecutions',60,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyAnalyticsUploadedFilesSizeMB',51200,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyEinsteinDataInsightsStoryCreation',1000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyEinsteinDiscoveryPredictAPICalls',50000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyEinsteinDiscoveryPredictionsByCDC',500000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyEinsteinDiscoveryStoryCreation',100,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyGenericStreamingApiEvents',10000,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyScratchOrgs',6,'H','restapi',0,'2021-01-09 01:02:16.655204',''),
	 ('DailyWorkflowEmails',330500,'H','restapi',0,'2021-01-09 01:02:16.655204','');

INSERT INTO public.prophetholiday (holiday,ds, limit_name,eventdate) VALUES
    ('temp','2021-01-29','DailyApiRequests','2021-01-09 01:02:16.655204');