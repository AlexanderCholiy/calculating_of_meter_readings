import os
from email import header, message
from email.header import decode_header
from email.utils import parseaddr

from .constants import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_PREFIXES,
    EMAIL_DIR,
    MAX_ATTACHMENT_SIZE
)
from .exceptions import ValidationError


class EmailValidator:

    def prepare_msg_id(self, msg_id: str) -> str:
        msg_id = msg_id.strip()
        return self.prepare_text_from_encode(msg_id).split(' ')[-1]

    def prepare_text_from_encode(self, original_text: str) -> str:
        decoded_words = header.decode_header(original_text)
        email_filename = ''.join(
            str(
                word, encoding if encoding else 'utf-8', errors='replace'
            ) if isinstance(word, bytes) else word
            for word, encoding in decoded_words
        )
        return email_filename

    def prepare_email_from(self, email_from_original: header.Header) -> str:
        raw_value = str(email_from_original)

        raw_value = ' '.join(raw_value.splitlines())

        decoded_from = self._decode_mime_header(raw_value)

        _, addr = parseaddr(decoded_from)

        if addr:
            return addr.strip().lower()

        raise ValueError(
            f'Не удалось извлечь email из From: {decoded_from}'
        )

    def _decode_mime_header(self, value: str) -> str:
        """Декодирует MIME-заголовки вроде =?utf-8?B?...?="""
        if not value:
            return ''

        decoded_parts = decode_header(value)

        decoded = ''.join(
            part.decode(encoding or 'utf-8') if isinstance(
                part, bytes
            ) else part
            for part, encoding in decoded_parts
        )

        return ' '.join(decoded.replace('\r', '').split('\n'))

    def save_email_attachments(self, filename: str, part: message.Message):
        """
        Сохранение вложений из почты, с проверкой типов файлов.

        Raises:
            ValidationError: не допустимый тип файла.
        """
        content_type = part.get_content_type()

        if content_type == 'message/rfc822':
            inner = part.get_payload()[0] if isinstance(
                part.get_payload(), list
            ) else part.get_payload()
            payload = inner.as_bytes()
        else:
            payload = part.get_payload(decode=True)

        if payload is None:
            raise ValidationError(
                f'Не удалось извлечь содержимое файла {filename} '
                f'({content_type})'
            )

        file_size = len(payload)
        ext = os.path.splitext(filename)[1].lower()

        if not any(
            content_type.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES
        ) and ext not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                f'Недопустимый тип файла {filename} ({content_type})'
            )

        if file_size > MAX_ATTACHMENT_SIZE:
            raise ValidationError(
                f'Файл {filename} превышает max размер '
                f'{MAX_ATTACHMENT_SIZE / (1024 * 1024):.1f} MB'
            )

        filepath = os.path.join(EMAIL_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(payload)
