import logging


logging.basicConfig(level=logging.INFO)


def log_request(update):
    """Log all fields of an incoming update for debugging purposes.
    Expected update object provides attributes: message.from_user.id, username, first_name; message.text; message.chat.id.
    """
    try:
        user = update.message.from_user
        chat = update.message.chat
        logging.info(
            f"User id: {user.id}",
            f" username: {user.username}",
            f" first_name: {user.first_name}",
            f" message text: {update.message.text}",
            f" chat_id: {chat.id}"
        )
    except AttributeError:
        logging.info(f"Received update with missing expected fields: {update}")

