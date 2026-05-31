# Document Memory Model

**Status:** Design note. Not yet implemented.

## Starting Point

The uploaded material already points toward a chapter-aware and revision-friendly memory model. The thesis has an explicit structure with Introduction, Theory and Related Work, Methodology, Results, Discussion, Conclusion, Appendix, and Bibliography. The methodology also describes a local pipeline that stores results as CSV files, SQLite databases, and visualizations.

This means the document memory should preserve structure, not only support full-text search. Every chapter, section, table, and figure should have a stable identity and its own lineage so later revisions can target the right part of the thesis without losing context.

## First-Class Quality Tags

The assessment material divides quality work into five fixed aspects:

- textual presentation
- understanding
- solution
- discussion
- overall impression

These should be first-class tags in the memory system. They create a direct bridge between stored document fragments and later chapter-level improvement work.

## PDF Ingestion Requirement

The image-based external assessment PDF shows why the ingestion layer must distinguish between born-digital PDFs and scanned documents. Born-digital PDFs can be parsed directly before semantic indexing. Scanned or image-based PDFs should be routed through OCR first, then indexed after text extraction.

## Practical Storage Shape

Each memory item should store:

- stable ID
- document ID
- chapter and section path
- content type, such as paragraph, table, figure, rubric note, or appendix item
- source file and page range
- quality tags
- revision lineage
- extracted text
- embedding metadata

The goal is to make retrieval useful for thesis revision, not only for answering ad hoc search queries.
