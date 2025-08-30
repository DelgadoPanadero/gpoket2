import datasets
import awswrangler as wr

# Configuration for your Athena source
ATHENA_DATABASE = "your_data_lake_db"
ATHENA_ICEBERG_TABLE = "pokemon_raw_iceberg"
S3_STAGING_DIR = "s3://your-athena-query-results-bucket/staging/"

class BronzePokemon(datasets.GeneratorBasedBuilder):
    """Bronze layer: Ingests raw Pokemon data directly from an Athena/Iceberg table."""

    def _info(self):
        # This schema MUST match the schema of your Iceberg table in Athena.
        return datasets.DatasetInfo(
            description="Raw data of Pokemon, ingested from the enterprise data lake.",
            features=datasets.Features({
                # Note: Adjust these types to match your Iceberg table schema exactly.
                'name': datasets.Value("string"),
                'type_1': datasets.Value("string"),
                'type_2': datasets.Value("string"),
                'hp': datasets.Value("int64"), # Use int64 for big int
                'attack': datasets.Value("int64"),
                'defense': datasets.Value("int64"),
                'is_legendary': datasets.Value("boolean"),
            }),
        )

    def _split_generators(self, dl_manager):
        # We don't need the download manager because our data source is a query.
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _generate_examples(self, **kwargs):
        """
        This method connects to Athena, queries the Iceberg table,
        and yields each row.
        """
        query = f'SELECT * FROM "{ATHENA_DATABASE}"."{ATHENA_ICEBERG_TABLE}"'

        print(f"Executing Athena query: {query}")

        # Use chunked=True to get an iterator of pandas DataFrames.
        # This is highly memory-efficient for very large tables.
        df_iterator = wr.athena.read_sql_query(
            sql=query,
            database=ATHENA_DATABASE,
            s3_output=S3_STAGING_DIR,
            chunked=True
        )

        key = 0
        for chunk_df in df_iterator:
            for _, row in chunk_df.iterrows():
                # Yield a dictionary matching the features defined in _info()
                yield key, row.to_dict()
                key += 1