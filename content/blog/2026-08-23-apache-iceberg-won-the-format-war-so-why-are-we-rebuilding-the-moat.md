+++
title = "Apache Iceberg Won the Format War — So Why Are We Rebuilding the Moat?"
date = 2026-08-23
taxonomies = { tags = ["spark", "data-engineering", "apache-iceberg", "lakehouse", "airflow", "data-catalog"] }
description = "How Apache Iceberg's snapshots, catalogs, and maintenance routines power a fast multi-engine lakehouse — with Spark config and an Airflow DAG"
link = "https://blog.pvcodes.in/apache-iceberg-the-backbone-of-the-modern-open-lakehouse"
params = { math = true }
+++


For three years, every lakehouse architecture review started with the same tribal question: *Iceberg, Delta Lake, or Hudi?*

In 2026, that question died. And the interesting part is what happened next — because the victory didn't produce the freedom everyone assumed it would. Lock-in didn't disappear. It moved.

## How the war actually ended

By mid-2026, the outcome stopped being debatable. Iceberg v3 went generally available on Snowflake in May, Databricks ran it in public preview, Amazon's S3 Tables signed on, and Delta Lake started converging rather than fighting — its [UniForm](https://www.cloudmagazin.com/en/2026/07/01/iceberg-won-the-format-war-now-the-catalog-counts/) feature exposes Delta tables as Iceberg-readable, so Snowflake, BigQuery, Redshift, and Trino can all read them. When your rival's escape hatch is *becoming you*, the war is over.

The finishing move came last week: on August 21, [AWS shipped Glue 6.0](https://aws.amazon.com/blogs/aws/aws-glue-6-0-now-available-with-30-lower-price-and-full-apache-iceberg-v3-support/) with full Iceberg v3 support — the `VARIANT` type with shredding, native geometry/geography, nanosecond timestamps — on a modernized Spark 4.1 runtime, at 30% lower price. When the largest cloud discounts compute to carry your format, you're not a challenger anymore. You're infrastructure.

Netflix, Apple, Airbnb, LinkedIn already run petabyte-scale estates on it. Fine. Everyone popped champagne. And then something quietly inconvenient happened.

## The moat moved one floor up

Here's the uncomfortable observation making the rounds among platform teams: **an open format does not automatically shield you from lock-in — because the binding point moved from the files to the catalog.**

Strip away the marketing and a catalog does one mechanical thing: it holds the pointer to each table's current metadata and swaps it atomically on commit. But sitting on that pointer means sitting on everything that matters operationally — which tables exist, who may read them, which credentials engines get, where lineage and audit trails live. As [dataarchitect.studio put it](https://dataarchitect.studio/essays/how-to-choose-an-iceberg-catalog/): the data files never move when you switch catalogs, but every engine config, access policy, and audit history does. That switching cost — not the format — is where lock-in lives now.

And the vendors know it. Look at the current lineup: Apache Polaris (donated by Snowflake, now a top-level Apache project as of February 2026), Databricks' Unity Catalog (open-sourced under the Linux Foundation), AWS Glue's REST catalog, Google's BigQuery managed REST interface (preview since April), Snowflake Open Catalog, Nessie. Every major player suddenly discovered religion about metadata. That's not a coincidence; that's positioning.

## Read the receipts: the federation fight

If you want to see how real this is, read the [July blog post from Snowflake](https://www.snowflake.com/content/snowflake-site/global/en/blog/bidirectional-interoperability-snowflake-horizon-databricks) taking direct aim at Unity Catalog. Whatever you think of the source — and you should read it as one side of a vendor dispute — the *technical distinctions* it draws are exactly the ones worth internalizing:

- **Inbound federation**: external engines connect to *your* catalog and read/write your tables. Everyone claims this.
- **Outbound federation**: your platform reaches into *someone else's* catalog through the standard Iceberg REST protocol — respecting their policies, using their credentials. This is where things get selective.

Snowflake's claim: Unity Catalog handles inbound fine, but when accessing external catalogs it bypasses the REST protocol entirely — using a proprietary SDK over JDBC plus preconfigured IAM roles — and leaves those tables read-only. Databricks would frame the same choices differently, no doubt. But notice what the argument is *about*: nobody is fighting over Parquet files anymore. They're fighting over **credential vending** — whether short-lived tokens scoped to one query flow from catalog to engine per the open spec, versus long-lived IAM keys duplicated across two policy systems.

That detail sounds obscure until you realize it determines who enforces row-level security when a query crosses platforms. The format guaranteed the bytes were readable. It says nothing about who's allowed to read them.

## The deeper insight: openness is an architectural property, not a file property

This is the takeaway worth sharing with your team:

> A table format can be perfectly open while your platform remains perfectly closed — because openness doesn't live in the data path. It lives in the control plane: the commit protocol, the credential path, the policy enforcement point.

We learned this exact lesson with Kubernetes (open source ≠ portable) and we're re-learning it here. The question "are we using open technology?" is nearly useless. The operational questions are:

1. Does our catalog implement the **full Iceberg REST spec**, including commit conflict handling?
2. Does it **vend credentials bidirectionally** — consume other catalogs' vended tokens, not just issue its own?
3. Can we leave? Concretely: if we switched catalogs next quarter, what breaks besides connection strings?

Ask those before you ask "which catalog is best." There is no best. Follow your gravity — Databricks estate → Unity, deep-AWS → Glue, deliberately neutral → Polaris — but choose knowing that gravity is the price.

## Meanwhile, underground: even the winner accrues debt

One more thing the victory lap misses. While the vendors fight over catalogs, the Iceberg community is doing the unglamorous work of paying down the format's own debts — and the [August dev-list activity](https://dev.to/alexmercedcoder/apache-data-lakehouse-weekly-august-10-to-august-18-2026-1nf5) is a masterclass in how mature infrastructure actually evolves:

- **Equality deletes are being banned in V4.** A formal vote proposes forbidding new equality-delete writes because they impose a tax on *every read* and block CDC and incremental views. Deletion vectors replace them. If you run streaming pipelines with equality deletes today, you have a migration deadline hiding in a spec vote.
- **V4 manifests may go Parquet-only.** Avro can't do projection reads on manifest stats, so the community is converging on dropping it — meaning tooling built to inspect Avro manifests needs a plan.
- **Upgrades are cheap by design, clean only if you maintain.** V3→V4 will be an O(1) metadata swap — but community regulars are openly worrying that legacy manifests linger unless operators actually run maintenance jobs they historically skip. Sound familiar? Same lesson as snapshot expiry: the spec hands you the lever; discipline pulls it.
- Even Spark still defaults to creating **v2 tables** while Variant/Geo implementations mature — "won" formats still ship with asterisks.

The pattern generalizes: *adopting the winning format is the beginning of maintenance obligations, not the end of decisions.* The teams getting burned in 2027 will be the ones who treated "we're on Iceberg" as a completed project.

## What to actually do

1. **Treat catalog selection like a marriage, not a config choice.** Audit against the REST spec, test cross-engine writes both directions, verify vended credentials end-to-end.
2. **Keep exactly one authoritative catalog per dataset.** Two independent catalogs pointing at the same files don't coordinate commits — that's corruption, not redundancy.
3. **Build the exit ramp early.** Federation (Glue↔Polaris↔Unity style) lets you introduce a neutral catalog incrementally instead of betting the platform on one vendor's goodwill.
4. **Track the V4 votes** if you stream writes — equality-delete deprecation is the rare breaking change you can see coming two releases ahead.

The lakehouse finally has its universal storage format. Whether it has universal *freedom* depends on a component most architectures treat as an implementation detail. Don't.

## Sources

- [Cloudmagazin — Iceberg won the format war. Now the catalog counts.](https://www.cloudmagazin.com/en/2026/07/01/iceberg-won-the-format-war-now-the-catalog-counts/)
- [AWS News Blog — Glue 6.0 with 30% lower price and full Iceberg v3 support](https://aws.amazon.com/blogs/aws/aws-glue-6-0-now-available-with-30-lower-price-and-full-apache-iceberg-v3-support/)
- [Snowflake Blog — Why Bidirectional Iceberg REST Matters](https://www.snowflake.com/content/snowflake-site/global/en/blog/bidirectional-interoperability-snowflake-horizon-databricks)
- [Alex Merced — Choosing the Right Iceberg Control Plane](https://iceberglakehouse.com/posts/2026-05-24-choosing-iceberg-control-plane/)
- [dataarchitect.studio — How to Choose an Iceberg Catalog](https://dataarchitect.studio/essays/how-to-choose-an-iceberg-catalog/)
- [Apache Data Lakehouse Weekly — Aug 10–18, 2026](https://dev.to/alexmercedcoder/apache-data-lakehouse-weekly-august-10-to-august-18-2026-1nf5)
- [Apache Data Lakehouse Weekly — Aug 5–12, 2026](https://amdatalakehouse.substack.com/p/apache-data-lakehouse-weekly-august)
- [Apache Iceberg 1.12.0 release discussion](https://www.mail-archive.com/dev@iceberg.apache.org/msg14746.html)
