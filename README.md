# Duplicate-Bug-Retrieval

### *Dataset Download*
> Dataset: http://alazar.people.ysu.edu/msr14data/
> mongorestore
> mongo mozilla
> show collections
> db.mozall.count()

### exporting the dataset to csv
> convert and export the mongodb database to a csv file
> mongoexport --host localhost --db mozilla --collection mozall --type=csv --out bug-dataset-mozilla.csv --fields=_id,bug_id,product,description,bug_severity,dup_id,short_desc,priority,version,component,delta_ts,bug_status,creation_ts,resolution
[!Exporting to CSV](media/convert.png)

### cleaning dataset
> - Changing status of non existent duplicate bugs to nonduplicate