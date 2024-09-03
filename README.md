DonaDev Project Documentation
=============================

<div align="center">
  <img src="https://github.com/user-attachments/assets/4ca88502-4ff5-4578-ad4b-c394b24287ad" width="200" />
  <img src="https://github.com/user-attachments/assets/c9b1df0b-802f-4b5b-93fa-838b05c4fd6e" width="200" />
  <img src="https://github.com/user-attachments/assets/4daca6dd-f449-4670-b15f-a4d467a046f7" width="200" />
  <img src="https://github.com/user-attachments/assets/f3ddfba4-3cab-4270-ae4c-11da9f8e8ebd" width="200" />
</div>


Introduction
------------
DonaDev is a project that integrates a neural network training framework with Telegram, allowing users to control and interact with the training process through a Telegram bot. It supports common datasets like MNIST and CIFAR-10, as well as custom datasets provided by the user. It also provides on-fly data collector. 

Features
--------
- Control the training process via a Telegram bot.
- Set hyperparameters such as learning rate, batch size, and number of epochs.
- Collect and label data directly through Telegram.
- Train models on standard datasets or custom datasets.
- Check CUDA status for GPU availability.
- Log random seed values to ensure reproducibility.
- Automatically push updates to GitHub.

Installation
------------
1. Clone the repository:
   
```bash
git clone https://github.com/yourusername/donadev.git 
```

2. Install the required dependencies
   
```bash
pip install -r requirements.txt
```

How to Use Dona Bot
-------------------
1. **Set Up the Telegram Bot:**
- Create a new Telegram bot by chatting with the BotFather on Telegram.
- Obtain the bot token and your chat ID.

2. **Run the DonaDev Main Script:**
- Open the `main.py` file and replace `YOUR_TELEGRAM_BOT_TOKEN` and `YOUR_CHAT_ID` with your actual bot token and chat ID.
- Run the script:
  
  ```
  python main.py
  ```

3. **Interacting with the Bot:**
- Start a conversation with your bot on Telegram.
- Follow the instructions sent by the bot to:
  - Set hyperparameters (learning rate, batch size, epochs, etc.).
  - Train a model on a dataset (MNIST, CIFAR-10, or a custom dataset).
  - Collect and label data for custom datasets.
  - Check CUDA status to ensure GPU availability.
  - Review random seed logs and training results.
  - Push code updates to a GitHub repository.

4. **Training a Model:**
- The bot will ask if you want to input custom hyperparameters or use defaults.
- Choose the dataset type (package-based like MNIST or CIFAR-10, or custom).
- If using a custom dataset, you can upload images directly through Telegram.

5. **GitHub Integration:**
- The bot will ask if you want to push updates to a GitHub repository after training.
- If yes, provide the repository URL and indicate whether it is a private repository.

6. **Ending the Session:**
- You can stop the training process at any time by sending the "stop" command.
- You can rerun the training with different parameters by sending the "rerun" command.

Dependencies
------------
The following Python packages are required to run DonaDev:
- torch
- torchvision
- requests
- matplotlib
- Pillow
