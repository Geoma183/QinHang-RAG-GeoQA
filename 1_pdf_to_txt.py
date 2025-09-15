import os
import json
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pdfminer.high_level import extract_text
import pytesseract
from PIL import Image
import logging
import hashlib
from tqdm import tqdm
from langdetect import detect
import argparse


# ========================== Configuration ==========================

TESSDATA_DIR = 'tesseract/tessdata'  # Update as needed
TESSERACT_CMD = 'tesseract'          # 'tesseract.exe' on Windows or just 'tesseract' on Linux/Mac
LANGUAGES = ['chi_sim', 'eng']
MAX_WORKERS = 8
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
OUTPUT_FORMAT = 'json'

# ========================== PDF Processor Class ==========================

class PDFProcessor:
    def __init__(self, tessdata_dir, tesseract_cmd, languages=['chi_sim', 'eng']):
        os.environ['TESSDATA_PREFIX'] = tessdata_dir
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.languages = languages
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_file_hash(self, pdf_path):
        """Generate a hash for the file for consistent output naming."""
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def extract_text_pdfminer(self, pdf_path):
        """Extract text using PDFMiner."""
        try:
            self.logger.info(f"Extracting text via pdfminer: {pdf_path}")
            return extract_text(pdf_path)
        except Exception as e:
            self.logger.warning(f"PDFMiner failed: {e}")
            return ""

    def extract_text_ocr(self, pdf_path):
        """Extract text using OCR as fallback."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang='+'.join(self.languages))
                text += page_text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return ""

    def clean_text(self, text):
        """Clean unwanted characters and excessive spaces."""
        self.logger.info("Cleaning text...")
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def detect_language(self, text):
        """Detect the language of the given text."""
        try:
            return detect(text)
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return "unknown"

    def split_text(self, text, chunk_size=800, chunk_overlap=150):
        """Split text into overlapping chunks with sentence boundary preference."""
        paragraphs = text.split("\n\n")
        chunks = []
        sentence_splitter = re.compile(r'(?<=[.!?；。！？])')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                sentences = sentence_splitter.split(para)
                temp_chunk = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += sentence
                    else:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = temp_chunk[-chunk_overlap:] + sentence
                if temp_chunk:
                    chunks.append(temp_chunk.strip())

        self.logger.info(f"Text split into {len(chunks)} chunks.")
        return chunks

# ========================== Processing Pipeline ==========================

class PDFProcessingPipeline:
    def __init__(self, processor, output_dir):
        self.processor = processor
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_pdf_file(self, pdf_path):
        file_hash = self.processor.get_file_hash(pdf_path)
        self.logger.info(f"Processing: {pdf_path}")

        text = self.processor.extract_text_pdfminer(pdf_path)
        if not text:
            text = self.processor.extract_text_ocr(pdf_path)
        if not text:
            self.logger.warning(f"No text extracted from: {pdf_path}")
            return None

        cleaned_text = self.processor.clean_text(text)
        lang = self.processor.detect_language(cleaned_text)
        chunks = self.processor.split_text(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)

        output_path = os.path.join(self.output_dir, f"{file_hash}.{OUTPUT_FORMAT}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'file_hash': file_hash,
                'language': lang,
                'text_chunks': chunks,
            }, f, ensure_ascii=False, indent=4)

        return file_hash, output_path

# ========================== Batch Runner ==========================

def process_pdf_files(pdf_folder, output_folder):
    processor = PDFProcessor(TESSDATA_DIR, TESSERACT_CMD, LANGUAGES)
    pipeline = PDFProcessingPipeline(processor, output_folder)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for root, _, files in os.walk(pdf_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    futures[executor.submit(pipeline.process_pdf_file, pdf_path)] = pdf_path

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
            try:
                result = future.result()
                if result:
                    file_hash, output_path = result
                    print(f"Processed: {file_hash} → {output_path}")
            except Exception as e:
                print(f"Error: {e}")

# ========================== Entry Point ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="Batch PDF to cleaned, chunked text converter.")
    parser.add_argument('--pdf_folder', type=str, required=True, help='Input folder containing PDF files')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for processed JSON files')
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    process_pdf_files(args.pdf_folder, args.output_folder)
