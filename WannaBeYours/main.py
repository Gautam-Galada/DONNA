from ai_dev import AIDev
from dona_dev import DonaDev

if __name__ == "__main__":
    ai_dev = AIDev()
    dona_dev = DonaDev(
        token='7166875544:AAGBi7azOdXowUbrZKXHpm7p152pG-mDxkA',
        chat_id='5255840420',
        ai_dev=ai_dev,
        data_dir='data_dir'
    )
    dona_dev.main_loop()
