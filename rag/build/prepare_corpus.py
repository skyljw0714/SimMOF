from parser import parse_all_xml_files, delete_short_txt_files
from embedder import build_vector_store_batched
from utils import ensure_dir
from config import RAW_XML_DIR, PARSED_TEXT_DIR, VECTOR_STORE_DIR

SKIP_PARSE = True
SKIP_EMBEDDING = False

def main():
    ensure_dir(PARSED_TEXT_DIR)
    ensure_dir(VECTOR_STORE_DIR)

    if not SKIP_PARSE:
        print("[1/2] Parsing XML files...")
        count = parse_all_xml_files(RAW_XML_DIR)
        print(f"Parsed {count} files.")
        delete_short_txt_files(PARSED_TEXT_DIR)
    else:
        print("[1/2] Skipped XML parsing.")

    if not SKIP_EMBEDDING:
        print("[2/2] Building vector store...")
        build_vector_store_batched(file_batch_size=200, encode_batch_size=512)
    else:
        print("[2/2] Skipped embedding.")

if __name__ == "__main__":
    main()