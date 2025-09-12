import asyncio
import logging
import sys
import re
from os import getenv
from pathlib import Path
from datetime import datetime

from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    ReplyKeyboardRemove,
    Document,
)
from aiogram.exceptions import TelegramBadRequest

from dotenv import load_dotenv, find_dotenv

from backend.indexer import index_file
from backend.rag_qa import answer_with_top_docs, answer_rag_with_llm


env_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=env_path, override=False)

BOT_TOKEN = (getenv("BOT_TOKEN") or "").strip()
if not re.match(r"^\d{6,}:[A-Za-z0-9_-]{35,}$", BOT_TOKEN):
    raise RuntimeError(
        "BOT_TOKEN не задан или выглядит некорректно. Проверь .env или переменные окружения.")

ALLOWED_EXT = tuple(
    x.strip().lower().lstrip(".")
    for x in (getenv("ALLOWED_EXT") or "pdf,txt,md,docx,html").split(",")
    if x.strip()
)
MAX_FILE_MB = int(getenv("MAX_FILE_MB") or "25")
TOP_DOCS = int(getenv("TOP_DOCS") or "5")
CHUNKS_PER_DOC = int(getenv("CHUNKS_PER_DOC") or "3")
LOG_LEVEL = (getenv("LOG_LEVEL") or "INFO").upper()
RAG_USE_LLM = (getenv("RAG_USE_LLM") or "true").strip().lower() in {
    "1", "true", "t", "yes", "y"}

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("rag-tg-bot")

start_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Загрузить документ",
                              callback_data="upload")],
        [InlineKeyboardButton(text="Помощь", callback_data="help")],
    ]
)

dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    """Приветствие и базовая инструкция."""
    await message.answer(
        f"Привет, {html.bold(message.from_user.full_name)}!\n"
        f"Этот бот помогает отвечать на вопросы по твоим документам.\n\n"
        f"Нажми «Загрузить документ», затем прикрепи файл как документ.",
        reply_markup=start_keyboard,
    )


@dp.message(Command("help"))
async def on_help_cmd(message: Message) -> None:
    """Справка по использованию бота."""
    exts = ", ".join(f".{x}" for x in ALLOWED_EXT)
    await message.answer(
        "Как пользоваться:\н"
        "• Нажми «Загрузить документ» и пришли файл как документ.\н"
        "• После индексации задай вопрос — бот вернет релевантные фрагменты или итоговый ответ.\н\n"
        f"Поддерживаемые форматы: {exts}\н"
        f"Максимальный размер файла: {MAX_FILE_MB} MB\n",
        reply_markup=start_keyboard,
    )


@dp.callback_query(F.data == "help")
async def on_help(callback: CallbackQuery) -> None:
    """Подсказка по загрузке документа."""
    await callback.message.answer(
        "Как загрузить документ:\n"
        "1) Нажми «Загрузить документ»\n"
        "2) Нажми скрепку → «Файл/Документ»\n"
        "3) Выбери файл (PDF/DOCX/TXT/MD/HTML)\n",
        reply_markup=ReplyKeyboardRemove(),
    )
    await callback.answer()


@dp.callback_query(F.data == "upload")
async def on_upload_click(callback: CallbackQuery) -> None:
    """Инструкция перед отправкой файла."""
    await callback.message.answer(
        "Пришли документ как файл. Для отмены ничего не отправляй или введи /start заново."
    )
    await callback.answer()


def _ext(filename: str) -> str:
    """Вернуть расширение файла без точки, в нижнем регистре."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@dp.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    """Принять документ, сохранить на диск и запустить индексацию в Milvus-lite."""
    doc: Document = message.document
    filename = doc.file_name or "file"
    ext = _ext(filename)

    if ext not in ALLOWED_EXT:
        await message.answer(
            f"Формат не поддерживается: <code>{filename}</code>\n"
            "Разрешено: " + ", ".join(f".{x}" for x in ALLOWED_EXT),
            parse_mode=ParseMode.HTML,
        )
        return

    if (doc.file_size or 0) > MAX_FILE_MB * 1024 * 1024:
        await message.answer(
            f"Файл слишком большой: {doc.file_size} байт. Лимит: {MAX_FILE_MB} MB."
        )
        return

    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"{timestamp}__{filename}"
    dest_path = upload_dir / safe_name

    try:
        try:
            await bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_DOCUMENT)
        except TelegramBadRequest:
            pass
        await bot.download(doc, destination=dest_path)
    except Exception as e:
        logger.exception("download failed")
        await message.answer("Ошибка скачивания файла:\n" + html.quote(str(e)))
        return

    await message.answer("Файл получен. Начинаю индексацию.")

    loop = asyncio.get_running_loop()

    def _run_index():
        return index_file(str(dest_path))

    try:
        doc_id, chunks = await loop.run_in_executor(None, _run_index)
        await message.answer(f"Индексация завершена.\nДокумент: {doc_id}\nЧанков: {chunks}")
    except Exception as e:
        logger.exception("indexing failed")
        await message.answer("Ошибка индексации:\n" + html.quote(str(e)))


@dp.message(F.text & ~F.via_bot)
async def handle_question(message: Message) -> None:
    """Ответить на вопрос: либо списком фрагментов, либо финальным ответом через GigaChat."""
    query = (message.text or "").strip()
    if not query:
        return
    try:
        if RAG_USE_LLM and (getenv("GIGACHAT_CREDENTIALS") or "").strip():
            answer = answer_rag_with_llm(
                query, top_docs=TOP_DOCS, chunks_per_doc=CHUNKS_PER_DOC)
        else:
            answer = answer_with_top_docs(
                query, top_docs=TOP_DOCS, chunks_per_doc=CHUNKS_PER_DOC)
        await message.answer(answer, parse_mode=None)
    except Exception as e:
        await message.answer(f"Ошибка обработки запроса:\n{e}", parse_mode=None)


async def main() -> None:
    """Инициализация и запуск поллинга."""
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(
        parse_mode=ParseMode.HTML))
    logging.info("starting polling…")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
