This is a Python project for comparing database logs from ONgDB 1.0.6 to 1.1.0

This was set up for mirroring requests to two databases with matching query log formats
exporting those logs into CSVs, which had a textPayload column with a row format like:

```
"""
'2025-08-13 11:59:58.946+0000 INFO <query-time> ms: bolt-session bolt neo4j neo4j-java/1.7.5-<sha> client/<address> server/<address>> neo4j - <query> - {}
"""
```

DB-A

```
'2025-08-13 11:59:59.946+0000 INFO 10 ms: bolt-session bolt neo4j neo4j-java/1.7.5-<sha> client/<address> server/<address>> neo4j - MATCH (a:Apple) return count(a) - {}
```

DB-B

```
'2025-08-13 12:01:01.946+0000 INFO 2003 ms: bolt-session bolt neo4j neo4j-java/1.7.5-<sha> client/<address> server/<address>> neo4j - MATCH (a:Apple) return count(a) - {}
```

The regression analysis looks for corresponding logs for matching queries, picks the matches with the closest timestamps, and filters the query set to only the regressions.

The regex could be easily modified to allow different formats.
