delete from salesforce."_trigger_log" where created_at < NOW() - INTERVAL '3 HOUR';
delete from salesforce."_trigger_log_archive" where created_at < NOW() - interval '1 DAY';