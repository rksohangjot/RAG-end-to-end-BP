# Medical test set - sources

Small demo set for testing ingestion and retrieval. Not for production. **Demo only - not medical, legal or insurance advice.**
Medicare figures change over time; questions are pinned to the documents as downloaded.

| File | Type | Origin |
|---|---|---|
| `medicare_claims_appeals.pdf` | Public | CMS, *Medicare Appeals* (Product No. 11525), https://www.medicare.gov/publications/11525-medicare-appeals.pdf - downloaded 2025-10-02. **Trimmed to pages 1-18** (cover, table of contents, Sections 1-3: filing an appeal and Original Medicare appeals). The table of contents still lists the removed sections. US government publication. |
| `medicare_how_to_file_claim.html` | Public | Medicare.gov, "Filing a claim", https://www.medicare.gov/providers-services/claims-appeals-complaints/claims - saved raw (with navigation and boilerplate, on purpose) 2025-10-02. US government content. |
| `plan_claims_refund_policy.docx` | Synthetic | Written for this project. Fictional insurer "Summit Health Supplement Plan". Includes a plans/costs table. |
| `scanned_claim_form.docx` | Synthetic | The claim form below as a picture inside a Word file. No text layer - needs OCR. |
| `scanned_claim_form.pdf` | Synthetic | The same picture as an image-only PDF. No text layer - needs OCR. Fallback in case OCR does not run on images embedded in DOCX. |

All names, IDs and amounts in the synthetic files are fictional.
