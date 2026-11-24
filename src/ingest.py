# Import the JSON library to convert Python dictionaries into JSON strings.
import json

# Import the library for reading PDF documents.
from pypdf import PdfReader

# Import tools for handling file paths.
from pathlib import Path

# Define the path to the source PDF.
pdf_path = Path("data/raw/nist-ai-rmf.pdf")

# Load the PDF document.
reader = PdfReader(pdf_path)

# Display the total number of pages in the PDF.
print("Number of pages: ", len(reader.pages))

# Create a list to store extracted text and source info.
pages = []

# Loop through every page in the PDF, starting from page 1.
for page_number, page in enumerate(reader.pages, start=1):
    # Extract text from the current page.
    text = page.extract_text()

    # Store the source filename, page number, and extracted text.
    pages.append({
        "source": pdf_path.name,
        "page": page_number,
        "text": text
    })

    # Display the number of processed pages.
    print("Number of processed pages: ", len(pages))

    # Display the first extracted page to verify the results.
    print(pages[0])

# Define the output file path.
output_path = Path("data/processed/pages.jsonl")

# Create the output directory if it does not exist.
output_path.parent.mkdir(parents=True, exist_ok=True)

# Open the output file in write mode.
with open(output_path, "w", encoding="utf-8") as f:
    # Loop through each extracted text and save them as a JSON line.
    for page in pages:
        f.write(json.dumps(page, ensure_ascii=False) + "\n")

# Confirm where the processed data was saved.
print("Saved extracted pages to:", output_path)

# Identify pages that contain no extracted text.
empty_pages = [
    page["page"] for page in pages
    # Check if the page text is empty after removing surrounding whitespace.
    if not page["text"].strip()
]

# Display the page numbers of any empty pages
print("Empty pages: ", empty_pages)
print("Number of empty pages: ", len(empty_pages))

# Select the 5th PDF page for inspection
sample_page = pages[4]

# Display the source filename.
print("Source: ", sample_page["source"])
# Display the source page number
print("Page: ", sample_page["page"])

# Display the first 1000 characters of the extracted text.
print(sample_page["text"][:1000])