import os
from datetime import datetime, timedelta
from email import header
from imaplib import IMAP4_SSL
from typing import Optional

from dotenv import load_dotenv

from core.config import Config
from core.logger import email_logger
from core.pretty_print import PrettyPrint

from .constants import FILENAME_DATETIME_PREFIX
from .exceptions import EmptyEmailSelect
from .utils import EmailParserManager
from .validators import EmailValidator

load_dotenv(override=True)

email_config = {
    'EMAIL_HOST': os.getenv('EMAIL_HOST'),
    'EMAIL_USER': os.getenv('EMAIL_USER'),
    'EMAIL_PSWD': os.getenv('EMAIL_PSWD'),
    'EMAIL_PORT': int(os.getenv('EMAIL_PORT', 993)),
    'EMAIL_SUBJECT': os.getenv('EMAIL_SUBJECT', 'profiledata').strip(),
    'EMAIL_DAYS_BEFORE': int(os.getenv('EMAIL_DAYS_BEFORE', 33)),
    'EMAIL_SENDER': os.getenv('EMAIL_SENDER', '').strip(),
}

Config.validate_env_variables(email_config)


class EmailParser(EmailParserManager, EmailValidator):

    def __init__(
        self,
        email_login: str,
        email_pswd: str,
        email_server: str,
        email_port: str | int,
        mailbox: str = 'INBOX'
    ):
        self.email_login = email_login
        self.email_pswd = email_pswd
        self.email_server = email_server
        self.email_port = int(email_port)
        self.mailbox = mailbox

    def find_msg_by_template(self, mail: IMAP4_SSL, search_query: str):
        status, messages = mail.search(None, search_query)
        if status != 'OK' or not messages[0]:
            return []

        return messages[0].split()

    def parser(self, subject: str, days_before: int, sender_email: str):
        date_cutoff = (
            datetime.now() - timedelta(days=days_before)
        ).strftime('%d-%b-%Y')
        search_query = (
            f'(SUBJECT "{subject}" '
            f'SINCE {date_cutoff} '
            f'FROM "{sender_email}")'
        )

        with IMAP4_SSL(self.email_server, self.email_port) as mail:
            mail.login(self.email_login, self.email_pswd)
            mail.select(self.mailbox, readonly=True)

            msg_ids = self.find_msg_by_template(mail, search_query)

            # Разворачиваем список: самые свежие ID теперь в начале
            msg_ids.reverse()
            messages = self._fetch_emails_in_chunks(mail, msg_ids)
            parsed_messages = self._parse_raw_messages(messages)

            total = len(parsed_messages)

            if not total:
                raise EmptyEmailSelect(search_query)

            for index, (msg, _) in enumerate(parsed_messages):
                PrettyPrint.progress_bar_warning(
                    index, total,
                    'Поиск письма с файлом для дорасчёта профилей мощности:'
                )

                email_msg_id: str = self.prepare_msg_id(msg['Message-ID'])
                sbj_header = msg.get('Subject')
                if sbj_header is not None:
                    subject, _ = header.decode_header(sbj_header)[0]
                else:
                    subject, _ = None, None

                cleaned_date_string = msg.get('Date').split(' (')[0]
                try:
                    email_date: datetime = datetime.strptime(
                        cleaned_date_string, '%a, %d %b %Y %H:%M:%S %z'
                    )
                except ValueError:
                    email_date: datetime = datetime.strptime(
                        cleaned_date_string, '%d %b %Y %H:%M:%S %z'
                    )

                if not msg.is_multipart():
                    email_logger.warning(
                        f'За {email_date} найдено письмо ({email_msg_id}) '
                        f'соответствующее шаблону {search_query}, '
                        'но без вложений.'
                    )

                for part in msg.walk():
                    unique_filename_part: str = (
                        email_date.strftime(FILENAME_DATETIME_PREFIX)
                    )
                    content_type: Optional[str] = (
                        part.get_content_type()
                    )
                    content_disposition: Optional[str] = (
                        part.get_content_disposition()
                    )
                    if (
                        (
                            content_disposition
                            and content_disposition == 'attachment'
                        )
                        or (
                            content_type
                            and content_type == 'message/rfc822'
                        )
                    ):
                        original_file_name: Optional[str] = part.get_filename()

                        if (
                            not original_file_name
                            and content_type
                            and content_type == 'message/rfc822'
                        ):
                            original_file_name = 'msg.eml'

                        if not original_file_name:
                            continue

                        email_filename: str = (
                            self.prepare_text_from_encode(
                                original_file_name
                            )
                        )
                        filename = (
                            f'{unique_filename_part}__{email_filename}'
                        )
                        self.save_email_attachments(filename, part)


email_parser = EmailParser(
    email_login=email_config['EMAIL_USER'],
    email_pswd=email_config['EMAIL_PSWD'],
    email_server=email_config['EMAIL_HOST'],
    email_port=email_config['EMAIL_PORT'],
)
