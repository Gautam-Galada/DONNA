

from ai_dev import AIDev
from dona_dev import DonaDev

if __name__ == "__main__":
    ai_dev = AIDev()
    dona_dev = DonaDev(
        token='BOT TOKEN',
        chat_id='CHAT ID',
        ai_dev=ai_dev,
        data_dir='data_dir',
        git_token='YOUR_GITHUB_PAT',
        ollama_host='localhost', 
        ollama_port=11434,
        custom_commands={
            'shutdown': 'shutdown -s -f -t 0',
            'lock pc': 'Rundll32.exe user32.dll,LockWorkStation',
            'hello world': 'echo Hello friend'
        }
    )
    dona_dev.main_loop1()
