"""
Project 2: PySpark Batch Processing Pipeline
Purpose: Transform generated data, add Project 2 extensions, compute analytics

Run from PowerShell:
docker exec -it sparklab bash -lc "python work/batch/batch_pipeline.py --in work/data/batch --out work/data/silver --config work/config/spark_config.json"
"""

import argparse
import json
import os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *


def build_spark(app_name, shuffle_partitions=200, jars_path="work/jars"):
    """
    Initialize Spark session with PostgreSQL JDBC driver if available
    """
    print(f"Initializing Spark session: {app_name}")
    
    builder = (SparkSession.builder
               .appName(app_name)
               .config("spark.sql.shuffle.partitions", shuffle_partitions)
               .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
               .config("spark.sql.adaptive.enabled", "true"))
    
    # Add JDBC driver if available
    jars_abs = f"/home/jovyan/{jars_path}"
    if os.path.isdir(jars_abs):
        jars = [os.path.join(jars_abs, f) for f in os.listdir(jars_abs) if f.endswith(".jar")]
        if jars:
            cp = ":".join(jars)
            builder = (builder
                      .config("spark.driver.extraClassPath", cp)
                      .config("spark.executor.extraClassPath", cp))
            print(f"✓ JDBC driver found: {len(jars)} jar(s)")
        else:
            print("⚠ No JDBC jars found - PostgreSQL publishing will fail")
    
    return builder.getOrCreate()


def read_parquet_safe(spark, path, entity_name):
    """Safely read Parquet with error handling"""
    try:
        df = spark.read.parquet(path)
        count = df.count()
        print(f"✓ Loaded {entity_name}: {count:,} records from {path}")
        return df
    except Exception as e:
        print(f"✗ ERROR reading {entity_name} from {path}")
        print(f"  Error: {e}")
        raise SystemExit(f"Failed to read {entity_name}")


def clean_customers(customers_df):
    """
    Data Quality: Clean customer data
    - Remove nulls in email
    - Deduplicate
    """
    print("\nCleaning customers data...")
    initial_count = customers_df.count()
    
    # Remove records with null emails
    cleaned = customers_df.filter(F.col("email").isNotNull())
    
    # Deduplicate by customer_id
    cleaned = cleaned.dropDuplicates(["customer_id"])
    
    final_count = cleaned.count()
    print(f"  Initial: {initial_count:,} → Final: {final_count:,} ({initial_count - final_count:,} removed)")
    
    return cleaned


def clean_products(products_df):
    """
    Data Quality: Clean product data
    - Remove negative prices
    - Remove duplicates
    """
    print("\nCleaning products data...")
    initial_count = products_df.count()
    
    # Remove negative prices
    cleaned = products_df.filter(F.col("unit_price") > 0)
    
    # Deduplicate
    cleaned = cleaned.dropDuplicates(["product_id"])
    
    final_count = cleaned.count()
    print(f"  Initial: {initial_count:,} → Final: {final_count:,} ({initial_count - final_count:,} removed)")
    
    return cleaned


def clean_orders(orders_df):
    """
    Data Quality: Clean orders data
    - Remove null customer_ids
    - Deduplicate
    """
    print("\nCleaning orders data...")
    initial_count = orders_df.count()
    
    # Remove orders without customers
    cleaned = orders_df.filter(F.col("customer_id").isNotNull())
    
    # Deduplicate
    cleaned = cleaned.dropDuplicates(["order_id"])
    
    final_count = cleaned.count()
    print(f"  Initial: {initial_count:,} → Final: {final_count:,} ({initial_count - final_count:,} removed)")
    
    return cleaned


def compute_customer_analytics(customers_df, orders_df, products_df):
    """
    PROJECT 2 REQUIREMENT: Customer Entity Extensions
    - lifetime_value (calculated field)
    - segment (derived from spending)
    - activity_score (based on order frequency)
    - preferred_category (most purchased category)
    """
    print("\n" + "="*70)
    print("COMPUTING CUSTOMER ANALYTICS (Project 2 Extensions)")
    print("="*70)
    
    # Join orders with products to get categories
    orders_enriched = (orders_df
                      .join(products_df.select("product_id", "category", "unit_price"), 
                            "product_id", "left"))
    
    # Calculate customer metrics
    customer_metrics = (orders_enriched
                       .groupBy("customer_id")
                       .agg(
                           F.sum("total_amount").alias("lifetime_value"),
                           F.count("order_id").alias("total_orders"),
                           F.countDistinct("order_date").alias("active_days"),
                           F.first("category").alias("preferred_category")  # Simplified
                       ))
    
    # Calculate activity score (orders per day active)
    customer_metrics = customer_metrics.withColumn(
        "activity_score",
        F.round(F.col("total_orders") / F.greatest(F.col("active_days"), F.lit(1)), 2)
    )
    
    # Assign segments based on lifetime value
    customer_metrics = customer_metrics.withColumn(
        "segment",
        F.when(F.col("lifetime_value") >= 1000, "Premium")
        .when(F.col("lifetime_value") >= 500, "Gold")
        .when(F.col("lifetime_value") >= 100, "Silver")
        .otherwise("Bronze")
    )
    
    # Join back to customers
    customers_extended = customers_df.join(customer_metrics, "customer_id", "left")
    
    # Fill nulls for customers with no orders
    customers_extended = (customers_extended
                         .fillna({
                             "lifetime_value": 0.0,
                             "total_orders": 0,
                             "active_days": 0,
                             "activity_score": 0.0,
                             "segment": "New",
                             "preferred_category": "Unknown"
                         }))
    
    print(f"✓ Customer analytics computed for {customers_extended.count():,} customers")
    
    # Show segment distribution
    print("\nCustomer Segment Distribution:")
    customers_extended.groupBy("segment").count().orderBy(F.desc("count")).show()
    
    return customers_extended


def compute_product_metrics(products_df, orders_df):
    """
    PROJECT 2 REQUIREMENT: Product Entity Extensions
    - popularity_score (based on order frequency)
    - profit_margin (unit_price vs cost - simulated)
    - inventory_turnover (sales velocity)
    """
    print("\n" + "="*70)
    print("COMPUTING PRODUCT METRICS (Project 2 Extensions)")
    print("="*70)
    
    # Calculate product sales metrics
    product_sales = (orders_df
                    .groupBy("product_id")
                    .agg(
                        F.count("order_id").alias("times_ordered"),
                        F.sum("quantity").alias("total_quantity_sold"),
                        F.sum("total_amount").alias("total_revenue")
                    ))
    
    # Join with products
    products_extended = products_df.join(product_sales, "product_id", "left")
    
    # Fill nulls for products never ordered
    products_extended = products_extended.fillna({
        "times_ordered": 0,
        "total_quantity_sold": 0,
        "total_revenue": 0.0
    })
    
    # Calculate popularity score (normalized)
    max_orders = products_extended.agg(F.max("times_ordered")).collect()[0][0] or 1
    products_extended = products_extended.withColumn(
        "popularity_score",
        F.round((F.col("times_ordered") / max_orders) * 100, 2)
    )
    
    # Simulate profit margin (assuming cost is 60-80% of price)
    products_extended = products_extended.withColumn(
        "estimated_cost",
        F.round(F.col("unit_price") * (0.6 + F.rand() * 0.2), 2)
    )
    
    products_extended = products_extended.withColumn(
        "profit_margin",
        F.round(((F.col("unit_price") - F.col("estimated_cost")) / F.col("unit_price")) * 100, 2)
    )
    
    # Calculate inventory turnover (quantity sold / stock quantity)
    products_extended = products_extended.withColumn(
        "inventory_turnover",
        F.round(F.col("total_quantity_sold") / F.greatest(F.col("stock_quantity"), F.lit(1)), 2)
    )
    
    print(f"✓ Product metrics computed for {products_extended.count():,} products")
    
    # Show top products
    print("\nTop 10 Products by Popularity:")
    (products_extended
     .select("product_id", "product_name", "category", "popularity_score", "times_ordered")
     .orderBy(F.desc("popularity_score"))
     .limit(10)
     .show(truncate=False))
    
    return products_extended


def compute_daily_metrics(orders_df, products_df):
    """
    Compute daily product performance metrics for dashboards
    """
    print("\n" + "="*70)
    print("COMPUTING DAILY PRODUCT METRICS")
    print("="*70)
    
    # Enrich orders with product info
    orders_enriched = orders_df.join(
        products_df.select("product_id", "unit_price", "category"),
        "product_id",
        "left"
    )
    
    # Aggregate by product and date
    daily_metrics = (orders_enriched
                    .groupBy("product_id", "order_date", "category")
                    .agg(
                        F.sum("quantity").alias("quantity_sold"),
                        F.sum("total_amount").alias("revenue"),
                        F.count("order_id").alias("num_orders"),
                        F.round(F.avg("total_amount"), 2).alias("avg_order_value")
                    ))
    
    # Calculate profit (simplified: revenue * 0.3)
    daily_metrics = daily_metrics.withColumn(
        "profit",
        F.round(F.col("revenue") * 0.3, 2)
    )
    
    print(f"✓ Daily metrics computed: {daily_metrics.count():,} product-day combinations")
    
    return daily_metrics


def write_postgres(df, config, table_key, table_name=None):
    """Write DataFrame to PostgreSQL"""
    try:
        pg_config = config["postgres"]
        table = table_name or pg_config.get(table_key)
        
        if not table:
            print(f"⚠ Table key '{table_key}' not found in config, skipping")
            return
        
        print(f"  Writing to PostgreSQL: {table}...")
        
        props = {
            "user": pg_config["user"],
            "password": pg_config["password"],
            "driver": pg_config["driver"]
        }
        
        df.write.mode(pg_config["mode"]).jdbc(
            pg_config["url"],
            table,
            properties=props
        )
        
        print(f"  ✓ Successfully wrote {df.count():,} records to {table}")
        
    except Exception as e:
        print(f"  ✗ PostgreSQL write failed for {table_key}: {e}")
        print("  (Check JDBC driver and database connection)")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="PySpark Batch Processing Pipeline - Project 2")
    parser.add_argument("--in", dest="in_path", default="work/data/batch",
                       help="Input directory with Parquet files")
    parser.add_argument("--out", dest="out_path", default="work/data/silver",
                       help="Output directory for processed data")
    parser.add_argument("--config", default="work/config/spark_config.json",
                       help="Configuration JSON file")
    parser.add_argument("--publish-postgres", default=None,
                       help="Override config: publish to PostgreSQL (true/false)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("PROJECT 2: PYSPARK BATCH PROCESSING PIPELINE")
    print("="*70)
    print(f"Input:  {args.in_path}")
    print(f"Output: {args.out_path}")
    print(f"Config: {args.config}")
    print("="*70)
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Determine if publishing to Postgres
    publish_postgres = config.get("publish_postgres", False)
    if args.publish_postgres is not None:
        publish_postgres = args.publish_postgres.lower() == "true"
    
    print(f"PostgreSQL Publishing: {'ENABLED' if publish_postgres else 'DISABLED'}")
    print("="*70)
    
    # Initialize Spark
    spark = build_spark(
        config.get("app_name", "DATA430_BatchPipeline"),
        config.get("shuffle_partitions", 200)
    )
    
    # Read raw data
    print("\n📥 READING RAW DATA")
    print("-"*70)
    customers_raw = read_parquet_safe(spark, f"{args.in_path}/customers", "Customers")
    products_raw = read_parquet_safe(spark, f"{args.in_path}/products", "Products")
    orders_raw = read_parquet_safe(spark, f"{args.in_path}/orders", "Orders")
    
    # Data Quality Cleaning
    print("\n🧹 DATA QUALITY CLEANING")
    print("-"*70)
    customers_clean = clean_customers(customers_raw)
    products_clean = clean_products(products_raw)
    orders_clean = clean_orders(orders_raw)
    
    # Compute Project 2 Analytics
    customers_final = compute_customer_analytics(customers_clean, orders_clean, products_clean)
    products_final = compute_product_metrics(products_clean, orders_clean)
    daily_metrics = compute_daily_metrics(orders_clean, products_clean)
    
    # Write to Parquet (Silver Layer)
    print("\n💾 WRITING PROCESSED DATA (Parquet)")
    print("-"*70)
    
    print("Writing customers with analytics...")
    customers_final.write.mode("overwrite").parquet(f"{args.out_path}/customers_analytics")
    print(f"✓ Customers written: {customers_final.count():,}")
    
    print("\nWriting products with metrics...")
    products_final.write.mode("overwrite").parquet(f"{args.out_path}/products_metrics")
    print(f"✓ Products written: {products_final.count():,}")
    
    print("\nWriting daily metrics (partitioned by order_date)...")
    daily_metrics.write.mode("overwrite").partitionBy("order_date").parquet(f"{args.out_path}/daily_metrics")
    print(f"✓ Daily metrics written: {daily_metrics.count():,}")
    
    print("\nWriting cleaned orders...")
    orders_clean.write.mode("overwrite").partitionBy("order_date").parquet(f"{args.out_path}/orders_clean")
    print(f"✓ Orders written: {orders_clean.count():,}")
    
    # Publish to PostgreSQL (optional)
    if publish_postgres:
        print("\n📤 PUBLISHING TO POSTGRESQL")
        print("-"*70)
        
        write_postgres(customers_final, config, "table_customers")
        write_postgres(products_final, config, "table_products")
        write_postgres