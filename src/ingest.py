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

# Select pages from the beginning, middle, and end of the document.
sample_indices = [0, len(pages) // 2, len(pages) - 1]


# Loop through the selected page indices.
for index in sample_indices:
    # Retrieve the corresponding page.
    sample_page = pages[index]

    # Display a separator between sample pages.
    print("\n" + "=" * 60)

    # Display the PDF page number.
    print("Page: ", sample_page["page"])

    # Display the full extracted text of the selected page.
    print(sample_page["text"])

# Define a section heading to search for in the extracted text.
heading = "Framing Risk"

# Loop through all extracted pages.
for page in pages:
    # Check whether the section heading appears in the page text.
    if heading in page["text"]:
        # Display the page number. 
        print("\nPage: ", page["page"])
        # Display the extracted text.
        print(page["text"])
