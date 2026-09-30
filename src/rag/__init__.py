"""Офлайн-пайплайн индексации локального корпуса документов (data/source/pdf/,
data/source/txt/) — разбор метаданных из имени файла, извлечение и очистка текста,
две независимые стратегии чанкинга и хранение индекса FAISS+SQLite по каждой из них.

Точка входа — `python -m rag.cli` (см. `make index`). Не связан с Telegram: команды
бота, читающие построенный этим пайплайном индекс (/research_chunking_stats,
/research_chunking_compare), находятся в research/chunking_stats.py и
research/chunking_compare.py, а не здесь — см. design.md изменения
add-rag-indexing-pipeline, раздел «Структура кода».
"""
