import email
from email import message
from imaplib import IMAP4
from typing import Any, Union

from core.logger import email_logger


class EmailParserManager:

    def _fetch_emails_in_chunks(
        self,
        mail: IMAP4,
        email_ids: list[Union[bytes, str]],
        chunk_size: int = 500
    ) -> list[tuple[bytes, Any]]:
        """
        Получает письма чанками, чтобы не перегружать IMAP сервер.

        Args:
            mail (IMAP4): активное IMAP4 соединение
            email_ids (list): список ID писем (bytes или str)
            chunk_size (int): размер чанка для FETCH

        Returns:
            list: список сообщений в сыром виде от imaplib
        """
        normalized_ids = [
            id_.decode() if isinstance(id_, bytes) else str(id_)
            for id_ in email_ids
        ]

        total = len(normalized_ids)
        all_messages = []

        for i in range(0, total, chunk_size):
            chunk = normalized_ids[i:i + chunk_size]
            id_range = ','.join(chunk)

            try:
                status, messages = mail.fetch(id_range, '(RFC822)')
            except KeyboardInterrupt:
                raise
            except (IMAP4.abort, ConnectionResetError, OSError):
                continue
            except Exception as e:
                email_logger.error(
                    'Ошибка при FETCH (ids=%s): %s', id_range, str(e)
                )
                continue

            if status != 'OK':
                email_logger.warning(
                    f'Ошибка при получении писем (status={status})',
                )
                continue

            all_messages.extend(messages)

        return all_messages

    def _parse_raw_messages(
        self, messages: list
    ) -> list[tuple[message.Message, bytes]]:
        """
        Преобразует сырые байты из IMAP-ответа в объекты email.message.Message.
        Возвращает список кортежей (объект_письма, исходные_байты).
        """
        parsed_messages: list[tuple[message.Message, bytes]] = []

        for part in messages:
            if isinstance(part, tuple) and len(part) == 2:
                msg_bytes = part[1]

                if not msg_bytes:
                    continue

                msg = email.message_from_bytes(msg_bytes)
                parsed_messages.append((msg, msg_bytes))

        return parsed_messages
