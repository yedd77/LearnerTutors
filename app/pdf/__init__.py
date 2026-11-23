"""Reusable PDF rendering for downloadable learning materials (study sheets, later guides/exams).

``theme`` holds the look (colours, fonts, sizes), ``components`` the reusable
building blocks (icons, cards, header/footer), and each document type has its
own renderer module (``study_sheet``). Renderers take structured content and
never call an LLM.
"""
