+++
title = "Databricks FILE Type: Unstructured Data in the Lakehouse"
date = 2026-08-09
taxonomies = { tags = ["data-engineering", "lakehouse", "databricks", "unstructured-data", "multimodal"] }
description = "Databricks FILE type turns documents, images, and video into governed, AI-ready lakehouse columns. How EXTERNAL vs MANAGED work, and why it matters."
link = "https://blog.pvcodes.in/unstructured-data-becomes-a-first-class-citizen-databricks-file-type-and-the-multimodal-lakehouse"
+++


## Unstructured data becomes a first-class citizen: Databricks FILE type and the multimodal lakehouse

Last week Databricks announced the beta of `FILE`, a native column type that brings documents, images, audio, and video into your lakehouse tables with the same governance you already apply to structured data. It is a small syntax change and a large architectural shift: unstructured data stops living in a separate object store bolted on the side, and becomes something you can `SELECT`, secure, delete, and feed to AI functions like any other column.

If you have spent the last few years stitching PDFs and screenshots into ad-hoc pipelines, this is the post for you. We will look at what `FILE` actually stores, the two flavors you can declare, what it unlocks for AI workloads, and where the ecosystem (BigQuery, Lance, Snowflake) is heading in parallel.

## The problem: unstructured data has always been a second-class citizen

Most data estates are mostly unstructured. Contracts, product images, call recordings, and video outweigh metrics and transaction logs by a wide margin — but they never get the same treatment. The old playbook was to keep files in cloud storage and store a URL (or worse, the raw bytes) in a table:

- A `STRING` column holding an object key gives you zero governance. Permissions on the table and permissions on the bucket are two different models that drift apart, and deleting a row rarely deletes the file — leaving orphaned data behind.
- A `BINARY` column inlines the bytes, which works for thumbnails up to ~64 KB but collapses under gigabyte-scale video or full contract scans.

So teams defaulted to the real old playbook: store files in one system, build a separate pipeline to parse them into another system, and manage a third index. The Databricks announcement is a bet that this fragmentation is the bottleneck for AI: AI turns unstructured data into something queryable, but *only if* it is governed and managed alongside everything else.

## What `FILE` actually stores

`FILE` is not a blob type. It stores a **governed reference** to a file plus metadata — the file bytes stay in object storage and are read lazily, only when an AI function or UDF actually processes them. A `FILE` value has five fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `uri` | `STRING` | Location of the file (cannot be null) |
| `offset` | `BIGINT` | Byte offset into the file |
| `size` | `BIGINT` | Size in bytes |
| `content_type` | `STRING` | MIME type |
| `checksum` | `STRING` | Integrity token, e.g. `MD5:<digest>` or `ETAG:"..."` |

Because the value is a reference, metadata queries (`SELECT file.uri, file.size FROM ...`) never touch the actual bytes. That is what keeps query performance intact even when your table references gigabytes of media.

Databricks positions this deliberately against `BINARY`:

- Use `FILE` when you need to manage and process large unstructured files alongside structured data and pass them to AI functions.
- Use `BINARY` for small objects (up to 64 KB by default) where you want the bytes inline — a thumbnail stored with its row, for example.

## FILE EXTERNAL vs FILE MANAGED

You cannot declare a plain `FILE` column — you must choose one of two lifecycle flavors:

**`FILE EXTERNAL`** references files that already exist in a Unity Catalog volume. Nothing is copied, other tools reading the same paths keep working, and permissions are governed by the volume (`READ VOLUME`). You manage the files' lifecycle yourself — deleting a row does not touch the underlying file. This is the choice when you must not move data or disrupt existing readers.

**`FILE MANAGED`** copies files into managed storage (a "FileSpace" volume you declare on the table). Lifecycle is tied to the table: deleting rows makes the referenced files eligible for garbage collection, which is exactly the behavior you want for compliance flows like GDPR right-to-be-forgotten. It costs you a copy, and it is the recommended flavor for ML training and retrieval-augmented generation, where workloads reach the files through the table.

```sql
-- Reference files already in a volume, without moving them
CREATE TABLE attachments (
  id        BIGINT,
  document  FILE EXTERNAL
);

-- Populate from files that already exist in a volume
INSERT INTO attachments
SELECT row_number() OVER (ORDER BY file.uri), file
FROM read_files('/Volumes/my_catalog/my_schema/my_volume/', format => 'file');
```

For rows that need the bytes managed by the table:

```sql
ALTER TABLE reports ADD COLUMN attachment FILE MANAGED;
```

One practical note from the docs: automatic garbage collection is **not** supported in beta — you need a cleanup job for unreferenced managed files. Read the garbage-collection guidance before you assume deletes tidy up after themselves.

## Constructing and consuming FILE values

A handful of functions create `FILE` references: `to_file(path)` and `try_to_file(path)` build a reference from a path (returning NULL if the file is missing), `create_file()` uploads content, and `copy_file()` copies to a target. You can also cast from `VARIANT` or a `STRUCT` with exactly the fields `struct<uri:string, offset:bigint, size:bigint, content_type:string, checksum:string>`:

```sql
SELECT named_struct(
  'uri',          '/Volumes/my_catalog/my_schema/my_volume/report.pdf',
  'size',         CAST(19494 AS BIGINT),
  'content_type', 'application/pdf',
  'checksum',     'ETAG:v1'
)::FILE;
```

On the consumption side, the interesting part is `ai_parse_document(file)`, which extracts structured content from a file, and the fact that `FILE` values flow through SQL, Python, and Scala UDFs. Databricks' driving-clips example shows the shape of the future: each row pairs a `video` FILE column with structured columns (route, scene description, hazard label, embedding) — a single row holding both the media and its derived meaning.

The roadmap doubles down on that: versioning and cloning without copying binaries, streaming datasets directly into PyTorch for GPU-ready tensors, feature engineering (embeddings, classifications) without rewriting tables, and direct search over the table using vector and full-text indexes.

## Governance is the real headline

The most important line in the announcement is not the column type — it is that Unity Catalog row- and column-level access control now extends to raw files, and that lifecycle is unified. Delete a row that references a `FILE MANAGED` value and the file becomes eligible for cleanup, which is the compliance behavior (right-to-be-forgotten) that previously required custom reconciliation jobs between your table and your bucket. FILE type is being developed as an open standard with work underway to integrate it into the Parquet and Delta Lake formats — the explicit goal being portability across the ecosystem rather than lock-in.

## The wider race: the multimodal lakehouse

Databricks is far from alone in diagnosing this problem. The July 2026 landscape piece "The Multimodal Lakehouse" maps the field:

- **Google Cloud** announced (April 2026) a cross-cloud lakehouse built on managed Apache Iceberg plus BigQuery ObjectRefs, merging structured Iceberg tables with unstructured Cloud Storage data for unified multimodal analysis — the same gesture as FILE type, from the warehouse side.
- **LanceDB / Lance** argues vector-plus-metadata deserves its own AI-native file format, positioning Lance as a lakehouse for multimodal data rather than a siloed vector database.
- **Snowflake** is extending its warehouse dominance into lakehouse and multimodal territory; **Microsoft Fabric** is unifying lakehouse, warehouse, BI, and governance under OneLake; **Dremio** is pitching an "agentic lakehouse" with a semantic layer so AI agents query governed data directly.

The convergence is telling: everyone has concluded that unstructured data must be managed *as data*, with the same access control, lifecycle, and governance as tables — not parked in an adjacent service.

## What it means for data teams

If you are on Databricks with Delta Lake on Runtime 18 LTS+, the practical takeaways:

1. **Start with FILE EXTERNAL.** It does not move data or disrupt existing readers, so it is the safe first step for pointing governed references at files that already exist in volumes.
2. **Use FILE MANAGED when the table owns the data.** Compliance workflows and RAG/ML workloads that reach files through the table are where the lifecycle guarantees pay off — but plan a cleanup job, because automatic GC is not here yet.
3. **Don't bother parsing everything up front anymore.** Lazy loading plus `ai_parse_document` means you extract meaning when a query demands it, and Declarative Pipelines can do incremental processing so only new or modified documents hit expensive model API calls.
4. **Watch the open-standard angle.** If FILE lands natively in Parquet and Delta Lake, the "reference + metadata" pattern becomes the shared substrate for unstructured data across engines — the same way Delta/Iceberg became the substrate for tables.

Structured and unstructured data sharing one governed system was the hardest part of making AI production-ready. With FILE type, Databricks has made that a column declaration. Expect the other lakehouses to answer in kind — the multimodal lakehouse is coming, and it is being built out of columns, not bolt-on indexes.

## Sources

- [Introducing FILE type: a native column type for multimodal data](https://www.databricks.com/blog/introducing-file-type-native-column-type-multimodal-data)

- [FILE type](https://docs.databricks.com/aws/en/sql/language-manual/data-types/file-type) and [FILE type and unstructured data](https://docs.databricks.com/aws/en/unstructured/file) on Databricks Documentation.

- [The Multimodal Lakehouse: Data Engineering's Answer to AI's Messiest Problem](https://towardsai.com/p/machine-learning/the-multimodal-lakehouse-data-engineerings-answer-to-ais-messiest-problem)
