# DentalSelfie First-Phase Progress Report

This folder contains the English thesis progress report, its editable Markdown source, figures, Mermaid diagrams and build scripts. It follows the academic progress-report layout in the user-provided `gen_report` directory while using DentalSelfie project evidence for the content.

## Files

- `output/DentalSelfie_Progress_Report_01.docx` — final 21-page Word report for supervisor review; TOC, lists, numbered tables and page fields refreshed in Microsoft Word and visually inspected after PDF rendering.
- `report.md` — editable English narrative and technical content.
- `TEMPLATE_SPEC.md` — distilled layout and fidelity notes from the reference report.
- `report_config.json` — cover metadata. Confirm student and supervisor fields before institutional submission.
- `figure/` — charts and selected project-image demonstrations embedded in the report.
- `mermaid/` — editable Mermaid source and rendered diagrams.
- `references/` — report-local logo and bibliographic notes.
- `script/build_figures.py` — regenerate report-specific diagrams and charts from evidence.
- `script/build_report.py` — create the Word document from the Markdown and report assets.

## Rebuild

From the repository root, with the project environment active:

```powershell
.venv\Scripts\python.exe progress_report_01\script\build_figures.py
C:\Users\huynh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe progress_report_01\script\build_report.py
```

Mermaid diagrams are rendered through Mermaid Ink by the figure script; other figures use Matplotlib and checked local assets. A network connection is needed only to refresh Mermaid renderings. The DOCX builder copies the supplied Word template, then uses installed Microsoft Word to refresh its table of contents and page fields. The submitted DOCX includes all rendered images.

The report distinguishes complete work from next-phase proposals. Model results are validation-set evidence; no clinical-use, patient-independent or ordinary-phone performance claim is made. The test split remains reserved. The DOCX was rebuilt from the editable Markdown and figures on 25 October 2026; its 21-page Word rendering was inspected. The temporary QA PDF and page images are kept under the ignored `.venv/work/progress_report_01/` folder, not in the deliverable package.
