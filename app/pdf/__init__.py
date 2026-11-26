"""Reusable PDF rendering for downloadable learning materials (study sheets, exam papers).

``theme`` holds the look (colours, fonts, sizes), ``components`` the reusable
building blocks (text styles, section blocks, header/footer), and each document
type has its own renderer module (``study_sheet``, ``exam_paper``). Renderers
take structured content and never call an LLM.
"""
