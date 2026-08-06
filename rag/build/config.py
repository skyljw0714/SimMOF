import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)

RAW_XML_DIR = os.path.join(DATA_DIR, "raw_xml")
PARSED_TEXT_DIR = os.path.join(DATA_DIR, "parsed_fulltext")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_db_fulltext")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

VERBOSE = True
