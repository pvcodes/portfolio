+++
title = "Apache Iceberg: The Backbone of the Modern Open Lakehouse"
date = 2026-08-12
taxonomies = { tags = ["spark", "data-engineering", "apache-iceberg", "lakehouse", "airflow"] }
description = "How Apache Iceberg's snapshots, catalogs, and maintenance routines power a fast multi-engine lakehouse — with Spark config and an Airflow DAG"
link = "https://blog.pvcodes.in/apache-iceberg-the-backbone-of-the-modern-open-lakehouse"
params = { math = true }
+++



The lakehouse pitch is seductive: keep your data as open Parquet on cheap object storage, then let any engine — Spark for heavy ETL, Trino for dashboards, Athena for ad-hoc poking — query the same tables without copying anything. In practice, though, raw files on an object store give you none of the guarantees a database gives you. No transactions. No consistent view across engines. No safe schema evolution. That gap is exactly what **Apache Iceberg** fills — and understanding *why* it works changes how you operate it.

## Why Hive-style tables stopped scaling

In the classic lake setup, the Hive Metastore tracks tables at the *directory* level, and engines discover data by listing prefixes. Three problems fall out of that design:

- **Object stores are not databases.** S3-style stores offer neither atomic file replacement nor native locking, so two engines writing concurrently can corrupt partition state.
- **Planning cost grows with your table.** Listing prefixes to find files gets slower the bigger the table gets — you pay a tax proportional to history.
- **Directory-level tracking is fragile.** Renames, partial writes, and failed jobs leave partitions in states different readers interpret differently.

Iceberg attacks all three by shifting file tracking from the directory level to the *file level*. Instead of scanning prefixes, engines read explicit lists of files stored in Iceberg's metadata documents — so query planning stays fast regardless of table size.

## The format: a metadata tree with atomic swaps

An Iceberg table is a tree of immutable files:

```
catalog ──► metadata.json          (schema, partition spec, snapshot log)
               └── manifest list    (per snapshot: stats about manifests)
                      └── manifests  (one row per data file: path, partition, metrics)
                             └── data files (Parquet/ORC/Avro)
```

Per the [Iceberg spec](https://iceberg.apache.org/spec/), all table state lives in metadata files. **Every change creates a new metadata JSON file, and the old one is replaced by an atomic swap** — that single mechanism is what turns a pile of object-store blobs into a table with transactions. Snapshots capture the table at a point in time; each snapshot references its data files through manifests, which are reused across snapshots when files don't change. Manifest lists carry partition stats so planning can skip whole manifests without reading them.

This is also why time travel is nearly free: an old snapshot *is* a consistent, complete view of the table.

## The catalog is the linchpin

Here's the part teams underestimate. The metadata tree describes a table — but something has to atomically point at the *current* metadata JSON. That's the catalog, and it does three jobs in an open lakehouse: state management, atomic commit coordination, and multi-engine name resolution ([a good deep-dive here](https://iceberglakehouse.com/posts/2026-05-22-apache-iceberg-catalogs-explained/)).

When two engines commit concurrently, they race to swap the metadata pointer. The catalog arbitrates: the loser gets a conflict error, reloads metadata, resolves, and retries. No locks held, no corruption.

You have real choices here — Hive Metastore, AWS Glue, JDBC, Nessie, or the [REST catalog spec](https://apache.github.io/iceberg/catalog/) (available since 0.14.0). Two opinions I'll defend:

1. **Prefer a REST-compliant catalog** (the standard REST server, Polaris, or a managed equivalent). REST decouples client engines from the backend implementation — swap your catalog backend later without reconfiguring every engine.
2. **Never register the same data files under two independent catalogs.** Catalogs don't share transaction state; two writers overwriting each other's pointers is how you get silent corruption.

## Wiring up Spark

Iceberg plugs into Spark through DataSourceV2. A minimal shared-catalog setup looks like this:

```python
spark = (
    SparkSession.builder
    .appName("lakehouse")
    .config("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2")
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.rest", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.rest.type", "rest")
    .config("spark.sql.catalog.rest.uri", "http://rest-catalog-server:8181")
    .config("spark.sql.catalog.rest.warehouse", "s3://my-lakehouse/")
    .getOrCreate()
)
```

([Full configuration reference](https://iceberg.apache.org/docs/latest/spark-configuration/).)

One nuance worth knowing before it bites you: `SparkCatalog` manages Iceberg namespaces directly, while `SparkSessionCatalog` wraps Spark's built-in session catalog and delegates non-Iceberg tables to it — handy for mixed estates. But CTAS/RTAS are only atomic under `SparkCatalog`; with the session-catalog wrapper they lose atomicity ([DDL docs](https://iceberg.apache.org/docs/latest/spark-ddl/)). If you're migrating legacy Hive tables, the `snapshot` and `migrate` procedures register existing Parquet files as Iceberg tables *without rewriting the data*.

## Time travel, branches, and tags

Every commit produces a snapshot, and snapshots power both reader isolation and time travel. But the underrated feature is [branches and tags](https://iceberg.apache.org/docs/latest/branching/) — named references to snapshots with their own retention policies:

```sql
CREATE BRANCH prod.db.orders.audit_branch
  AS OF VERSION 1234
  RETAIN 30 DAYS;
```

Run a validation or data-quality workflow against `audit_branch`, then expire it on schedule — no copied tables, no sidecar datasets. Note the schema semantics: writes validate against the *table's* current schema, while querying a tag replays the schema as of that snapshot.

## Maintenance is your job now

Here's the honest part: Iceberg ships the primitives, not the janitor. There is no autovacuum. Every write appends a snapshot — a streaming table committing every five minutes accumulates ~8,600 snapshots a month — and until you clean up, those snapshots pin data files and bloat metadata.

Four operations matter ([official maintenance guide](https://iceberg.apache.org/docs/latest/maintenance/)):

| Operation | Procedure | Why |
| --- | --- | --- |
| Snapshot expiration | `expire_snapshots` | Drops old versions, releases data files |
| Orphan cleanup | `remove_orphan_files` | Deletes unreferenced physical files |
| Compaction | `rewrite_data_files` | Merges small files (binpack/sort/z-order) |
| Manifest rewrite | `rewrite_manifests` | Consolidates fragmented manifests |

**Order matters, and sources disagree.** [LakeOps argues](https://lakeops.dev/blog/automating-iceberg-table-maintenance) for *expire → orphan cleanup → compact → rewrite manifests*: expiring first avoids wasting compute rewriting files that were about to be deleted anyway, and manifests go last so they reflect the final layout. Others prefer compacting before expiring so fresh compaction snapshots aren't immediately dropped. Either school works — the non-negotiable principles are:

- Run all four as **one coordinated pipeline**, not four unrelated cron jobs.
- Use a **7+ day threshold** on `remove_orphan_files` (files from in-flight writes look orphaned; the default 3-day cutoff has bitten people).
- Set `retain_last` on expiration so a quiet table doesn't lose its entire history.
- Restrict streaming-table compaction to cold partitions with a `where` clause so you don't fight active writers.

Cadence depends on workload — [one practical guide](https://iceberglakehouse.com/iceberg/iceberg-maintenance-scheduling/) suggests hourly compaction/daily expiry for streaming tables versus post-load compaction for daily batches. And you don't need to guess: the `.files` and `.snapshots` metadata tables tell you average file sizes and snapshot ages, which makes great alerting queries.

Orchestrating it in Airflow is just sequential Spark procedure calls:

```python
compact >> rewrite_manifests >> expire_snapshots >> remove_orphan_files
# nightly, catchup=False, per table
```

## The opinionated summary

1. Treat the **catalog as production infrastructure**, not a config detail. REST-first.
2. **One writer catalog per dataset**, forever.
3. **Schedule maintenance from day one** — metadata debt compounds silently.
4. Use **branches and tags** for audit/validation workflows instead of cloning tables.
5. Partition on `day(ts)` unless volume truly demands `hour()`; hidden partitioning is one of Iceberg's best tricks.

Iceberg doesn't just make tables portable across engines — it makes the lakehouse *operable*. But only if you treat its metadata lifecycle as first-class engineering work rather than an afterthought.

## Sources

- [Apache Iceberg — Table Spec](https://iceberg.apache.org/spec/)
- [Apache Iceberg — Spark Configuration](https://iceberg.apache.org/docs/latest/spark-configuration/)
- [Apache Iceberg — Spark DDL](https://iceberg.apache.org/docs/latest/spark-ddl/)
- [Apache Iceberg — Branching and Tagging](https://iceberg.apache.org/docs/latest/branching/)
- [Apache Iceberg — Table Maintenance](https://iceberg.apache.org/docs/latest/maintenance/)
- [Apache Iceberg — Catalogs](https://apache.github.io/iceberg/catalog/)
- [Alex Merced — Apache Iceberg Catalogs Explained](https://iceberglakehouse.com/posts/2026-05-22-apache-iceberg-catalogs-explained/)
- [Alex Merced — Iceberg Maintenance Scheduling](https://iceberglakehouse.com/iceberg/iceberg-maintenance-scheduling/)
- [LakeOps — Automating Apache Iceberg Table Maintenance](https://lakeops.dev/blog/automating-iceberg-table-maintenance)
