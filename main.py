

from ai_dev import AIDev
from dona_dev import DonaDev

if __name__ == "__main__":
    ai_dev = AIDev()
    dona_dev = DonaDev(
        token='YOUR_TELEGRAM_BOT_TOKEN',
        chat_id='YOUR_CHAT_ID',
        ai_dev=ai_dev,
        data_dir='data_dir'
    )
    dona_dev.main_loop()
