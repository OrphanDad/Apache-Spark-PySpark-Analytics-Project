from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, StringType
from pathlib import Path
import configparser, sys

def spark():
    return (SparkSession.builder
            .appName("quotes-processing")
            .master("local[*]")
            .config("spark.driver.host","127.0.0.1")
            .config("spark.driver.bindAddress","127.0.0.1")
            .getOrCreate())

def main():
    base = Path(__file__).resolve().parents[1]
    cfg = configparser.ConfigParser(); cfg.read(base / "config.ini")
    input_dir = base / cfg["SCRAPER"]["output_dir"]

    files = sorted(str(p.resolve()).replace("\\","/") for p in input_dir.glob("*.json"))
    if not files:
        print("No JSON files found in:", input_dir); sys.exit(1)

    s = spark()
    df = s.read.option("multiLine", True).json(files)

    df = (df.withColumn("quote_text", F.trim("quote_text"))
            .withColumn("author", F.trim("author"))
            .withColumn("tags", F.col("tags").cast(ArrayType(StringType())))
            .withColumn("tags", F.when(F.col("tags").isNull(), F.array().cast(ArrayType(StringType()))).otherwise(F.col("tags")))
            .dropna(subset=["quote_text","author"])
            .dropDuplicates(["quote_text","author"])
            .withColumn("quote_length", F.length("quote_text"))
            .withColumn("word_count", F.size(F.split(F.col("quote_text"), r"\s+"))))

    authors    = df.select(F.col("author").alias("name")).distinct().orderBy("name")
    tags       = df.select(F.explode_outer("tags").alias("name")).where("name is not null").distinct().orderBy("name")
    quotes     = df.select("quote_text","author","page_number","scraped_at","quote_length","word_count")
    quote_tags = df.select("quote_text","author",F.explode_outer("tags").alias("tag")).where("tag is not null")

    out = base / "data" / "processed"
    authors.coalesce(1).write.mode("overwrite").option("header",True).csv(str(out / "authors"))
    tags.coalesce(1).write.mode("overwrite").option("header",True).csv(str(out / "tags"))
    quotes.coalesce(1).write.mode("overwrite").option("header",True).csv(str(out / "quotes"))
    quote_tags.coalesce(1).write.mode("overwrite").option("header",True).csv(str(out / "quote_tags"))
    df.coalesce(1).write.mode("overwrite").json(str(out / "cleaned"))

    s.stop()
    print("Processing complete →", out)

if __name__ == "__main__":
    main()
