"""
Project 2: PySpark Data Generator
Purpose: Generate large-scale synthetic datasets matching Project 1 schema
         Scale from 1,000 to 1,000,000 records per entity

Run from PowerShell:
docker exec -it sparklab bash -lc "python work/batch/data_generator_spark.py --rows 1000000 --out work/data/batch"
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DecimalType, DateType, TimestampType, BooleanType
from datetime import datetime, timedelta
import random


def build_spark(app_name: str = "DATA430_DataGenerator"):
    """Initialize Spark session"""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def generate_customers(spark, num_records):
    """
    Generate customers matching Project 1 schema with Project 2 extensions
    
    Project 1 Schema:
    - customer_id, first_name, last_name, email, phone, city, 
      registration_date, customer_type
    
    Project 2 Extensions (calculated later in pipeline):
    - lifetime_value, segment, activity_score, preferred_category
    """
    print(f"Generating {num_records:,} customer records...")
    
    # Cities for realistic data
    cities = ["Columbus", "Cleveland", "Cincinnati", "Toledo", "Akron", 
              "Dayton", "Parma", "Canton", "Youngstown", "Lorain"]
    
    # Customer types
    customer_types = ["Premium", "Standard", "Basic"]
    
    # Generate base dataframe
    df = spark.range(num_records)
    
    # Add all columns matching Project 1 schema
    df = (df
          .withColumn("customer_id", (F.col("id") + 1).cast("long"))
          .withColumn("first_name", F.concat(F.lit("FirstName"), F.col("customer_id")))
          .withColumn("last_name", F.concat(F.lit("LastName"), F.col("customer_id")))
          .withColumn("email", 
                     F.concat(F.lit("customer"), F.col("customer_id"), 
                             F.lit("@example.com")))
          .withColumn("phone", 
                     F.concat(F.lit("614-555-"), 
                             F.lpad((F.rand() * 10000).cast("int"), 4, "0")))
          .withColumn("city", 
                     F.array([F.lit(c) for c in cities])
                     .getItem((F.rand() * len(cities)).cast("int")))
          .withColumn("registration_date", 
                     F.date_sub(F.current_date(), (F.rand() * 1095).cast("int")))  # Last 3 years
          .withColumn("customer_type",
                     F.array([F.lit(ct) for ct in customer_types])
                     .getItem((F.rand() * len(customer_types)).cast("int")))
          .drop("id")
    )
    
    # Add some data quality issues (0.1% nulls in email)
    df = df.withColumn("email", 
                       F.when(F.rand() < 0.001, F.lit(None))
                       .otherwise(F.col("email")))
    
    return df


def generate_products(spark, num_records):
    """
    Generate products matching Project 1 schema with Project 2 extensions
    
    Project 1 Schema:
    - product_id, product_name, category, brand, unit_price, 
      stock_quantity, is_active
    
    Project 2 Extensions (calculated later):
    - popularity_score, profit_margin, inventory_turnover, recommendation_rank
    """
    # Scale products reasonably (not 1M products, but enough variety)
    actual_products = max(1000, num_records // 100)
    print(f"Generating {actual_products:,} product records...")
    
    categories = ["Electronics", "Books", "Clothing", "Home & Garden", 
                  "Toys", "Sports", "Automotive", "Health", "Grocery"]
    
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", 
              "BrandF", "BrandG", "BrandH", "BrandI", "BrandJ"]
    
    df = spark.range(actual_products)
    
    df = (df
          .withColumn("product_id", (F.col("id") + 1).cast("long"))
          .withColumn("product_name", 
                     F.concat(F.lit("Product_"), F.col("product_id")))
          .withColumn("category",
                     F.array([F.lit(c) for c in categories])
                     .getItem((F.rand() * len(categories)).cast("int")))
          .withColumn("brand",
                     F.array([F.lit(b) for b in brands])
                     .getItem((F.rand() * len(brands)).cast("int")))
          .withColumn("unit_price", 
                     F.round(F.rand() * 495 + 5, 2))  # $5 to $500
          .withColumn("stock_quantity",
                     (F.rand() * 1000).cast("int"))
          .withColumn("is_active",
                     F.when(F.rand() < 0.95, F.lit(True))
                     .otherwise(F.lit(False)))
          .drop("id")
    )
    
    # Add some invalid prices (0.05% negative prices for data quality testing)
    df = df.withColumn("unit_price",
                       F.when(F.rand() < 0.0005, F.lit(-10.0))
                       .otherwise(F.col("unit_price")))
    
    return df


def generate_orders(spark, num_records, max_customer_id, max_product_id):
    """
    Generate orders matching Project 1 schema with Project 2 extensions
    
    Project 1 Schema:
    - order_id, customer_id, order_date, order_status, payment_method,
      total_amount, discount_amount, shipping_cost
    
    Project 2 Extensions (calculated later):
    - processing_time, fraud_score, customer_satisfaction, revenue_impact
    """
    print(f"Generating {num_records:,} order records...")
    
    order_statuses = ["Completed", "Pending", "Shipped", "Cancelled", "Processing"]
    payment_methods = ["Credit Card", "Debit Card", "PayPal", "Cash", "Bank Transfer"]
    
    df = spark.range(num_records)
    
    df = (df
          .withColumn("order_id", (F.col("id") + 1).cast("long"))
          .withColumn("customer_id", 
                     (F.floor(F.rand() * max_customer_id) + 1).cast("long"))
          .withColumn("product_id",
                     (F.floor(F.rand() * max_product_id) + 1).cast("long"))
          .withColumn("quantity",
                     (F.floor(F.rand() * 5) + 1).cast("int"))
          .withColumn("order_date",
                     F.date_sub(F.current_date(), (F.rand() * 365).cast("int")))  # Last year
          .withColumn("order_status",
                     F.array([F.lit(s) for s in order_statuses])
                     .getItem((F.rand() * len(order_statuses)).cast("int")))
          .withColumn("payment_method",
                     F.array([F.lit(pm) for pm in payment_methods])
                     .getItem((F.rand() * len(payment_methods)).cast("int")))
          .withColumn("total_amount",
                     F.round(F.rand() * 495 + 10, 2))  # $10 to $505
          .withColumn("discount_amount",
                     F.round(F.rand() * 50, 2))  # $0 to $50
          .withColumn("shipping_cost",
                     F.round(F.rand() * 20 + 5, 2))  # $5 to $25
          .drop("id")
    )
    
    # Add data quality issues
    # 0.05% null customer_ids (orphaned orders)
    df = df.withColumn("customer_id",
                       F.when(F.rand() < 0.0005, F.lit(None))
                       .otherwise(F.col("customer_id")))
    
    # Add some duplicates (0.1% of records)
    duplicates = df.sample(0.001)
    df = df.unionByName(duplicates)
    
    return df


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Generate large-scale test data for PySpark Project 2"
    )
    parser.add_argument(
        "--rows", 
        type=int, 
        default=100000,
        help="Number of order records to generate (default: 100000)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="work/data/batch",
        help="Output directory path (default: work/data/batch)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("PROJECT 2: PYSPARK DATA GENERATOR")
    print("="*70)
    print(f"Target order records: {args.rows:,}")
    print(f"Output directory: {args.out}")
    print("="*70)
    
    # Initialize Spark
    spark = build_spark()
    
    # Calculate scaled record counts
    orders_count = args.rows
    customers_count = max(10000, orders_count // 5)  # 20% of orders
    products_count = max(1000, orders_count // 100)   # 1% of orders
    
    print(f"\nScaled record counts:")
    print(f"  Customers: {customers_count:,}")
    print(f"  Products:  {products_count:,}")
    print(f"  Orders:    {orders_count:,}")
    print()
    
    # Generate datasets
    customers_df = generate_customers(spark, customers_count)
    products_df = generate_products(spark, products_count)
    orders_df = generate_orders(spark, orders_count, customers_count, products_count)
    
    # Write to Parquet (columnar format, efficient for Spark)
    print("\nWriting customers to Parquet...")
    customers_df.write.mode("overwrite").parquet(f"{args.out}/customers")
    print(f"✓ Customers written: {customers_df.count():,} records")
    
    print("\nWriting products to Parquet...")
    products_df.write.mode("overwrite").parquet(f"{args.out}/products")
    print(f"✓ Products written: {products_df.count():,} records")
    
    print("\nWriting orders to Parquet (partitioned by order_date)...")
    orders_df.write.mode("overwrite").partitionBy("order_date").parquet(f"{args.out}/orders")
    print(f"✓ Orders written: {orders_df.count():,} records")
    
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"\nFiles written to: {args.out}/")
    print("  - customers/")
    print("  - products/")
    print("  - orders/ (partitioned by order_date)")
    print("\nNext step: Run batch_pipeline.py to transform this data")
    print("="*70)
    
    spark.stop()


if __name__ == "__main__":
    main()