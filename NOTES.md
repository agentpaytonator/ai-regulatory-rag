## Version-One Scope

- Input: A question about NIST AI Risk Management Framework (AI RMF 1.0)
- Output: An answer with the supporting source passage
- Interface: Command line
- Not included: User accounts, cloud deployment, elaborate UI, or a large document corpus


## Inspect Before Indexing

Inspection Performed
- The pages have been inspected from the beginning, middle, and end of the document.
- Made sure there are no broken words, repeated headers, and blank pages.
- Made sure the text is readable and in order.
- Searched for section headings to evaluate section boundaries.

Known Extraction Defects
- No broken words, repeated headers and footers, blank pages, nor lost section boundaries.

Current Limitations
- Page numbers correpsond to the physical PDF pages rather than matching the printed page numbers.
- Section headings are not stored as separate metadata fields.
- Text extraction quality may be differ across pages.
- No automated text-cleaning process has been implemeneted yet.