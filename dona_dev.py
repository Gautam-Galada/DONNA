import os
import time
import requests
import subprocess
from git import Repo

class DonaDev:
    def __init__(self,token,chat_id,ai_dev,data_dir):
        self.TOKEN=token
        self.chat_id=chat_id
        self.ai_dev=ai_dev
        self.data_dir=data_dir
        self.last_update_id=None

    def send_telegram_message(self,message):
        url=f"https://api.telegram.org/bot{self.TOKEN}/sendMessage?chat_id={self.chat_id}&text={message}"
        response=requests.get(url)
        print(f"Sent message: {message}, Response: {response.json()}")

    def send_telegram_image(self,image_path):
        url = f"https://api.telegram.org/bot{self.TOKEN}/sendPhoto"
        with open(image_path, 'rb') as image_file:
            files={'photo':image_file}
            data={'chat_id':self.chat_id}
            response=requests.post(url,files=files,data=data)
        print(f"Sent image: {image_path}, Response: {response.json()}")

    def get_updates(self,offset=None):
        url=f"https://api.telegram.org/bot{self.TOKEN}/getUpdates"
        params={'offset': offset,'timeout': 30}
        response=requests.get(url,params=params).json()
        return response.get('result',[])

    def clear_previous_messages(self):
        updates=self.get_updates()
        if updates:self.last_update_id=updates[-1]['update_id']+1
        print(f"Cleared previous messages. Last update ID: {self.last_update_id}")

    def get_cuda_options(self):
        p=subprocess.run(['nvidia-smi','-q'],capture_output=True,text=True)
        pret=p.returncode
        if pret != 0:
            print("Device doesn't have GPU or CUDA drivers error")
            self.send_telegram_message("Device doesn't have GPU or CUDA drivers error")
            return
        pop=p.stdout
        for popline in pop.splitlines():self.send_telegram_message(popline)

    def request_user_input(self, prompt):
        self.send_telegram_message(prompt)
        start_time=time.time()
        timeout=300
        while time.time()-start_time<timeout:
            updates=self.get_updates(self.last_update_id)
            for update in updates:
                self.last_update_id = update['update_id'] + 1
                if 'message' in update and 'text' in update['message']:return update['message']['text'].lower()
                if 'message' in update and 'photo' in update['message']:
                    image_id=update['message']['photo'][0]['file_id']
                    file_info=requests.get(f"https://api.telegram.org/bot{self.TOKEN}/getFile?file_id={image_id}").json()
                    file_path=file_info['result']['file_path']
                    image_bytes=requests.get(f"https://api.telegram.org/file/bot{self.TOKEN}/{file_path}").content
                    return image_bytes
            time.sleep(1)
        self.send_telegram_message("No input received within 5 minutes. Using default value.")
        return None

    def get_hyperparameters(self,dataset_choice):
        if dataset_choice=='mnist':
            default_lr=0.001
            default_bs=64
            default_epochs=2
            default_hidden_size=512
            default_image_size=(28,28)
            default_input_channels =1
        elif dataset_choice == 'cifar10':
            default_lr=0.001
            default_bs=64
            default_epochs=2
            default_hidden_size=512
            default_image_size=(32,32)
            default_input_channels=3
        else:
            default_lr=0.001
            default_bs=64
            default_epochs=2
            default_hidden_size=512
            default_image_size=(28,28)
            default_input_channels=3

        choice=self.request_user_input("Do you want to input hyperparameters? (yes/no)")
        if choice=='yes':
            learning_rate=float(self.request_user_input(f"Enter learning rate (e.g., {default_lr}):") or default_lr)
            batch_size=int(self.request_user_input(f"Enter batch size (e.g., {default_bs}):") or default_bs)
            num_epochs=int(self.request_user_input(f"Enter number of epochs (e.g., {default_epochs}):") or default_epochs)
            hidden_size=int(self.request_user_input(f"Enter hidden layer size (e.g., {default_hidden_size}):") or default_hidden_size)
            image_size=tuple(map(int, self.request_user_input(f"Enter image size (e.g., {default_image_size[0]}, {default_image_size[1]}):").split(',')) or default_image_size)
            input_channels=int(self.request_user_input(f"Enter input channels (e.g., {default_input_channels}):") or default_input_channels)
        else:
            learning_rate =default_lr
            batch_size =default_bs
            num_epochs= default_epochs
            hidden_size= default_hidden_size
            image_size= default_image_size
            input_channels= default_input_channels
        self.ai_dev.learning_rate= learning_rate
        self.ai_dev.batch_size= batch_size
        self.ai_dev.num_epochs = num_epochs
        self.ai_dev.hidden_size = hidden_size
        self.ai_dev.image_size = image_size
        self.ai_dev.input_channels = input_channels
        self.send_telegram_message(f"Using hyperparameters: learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}, hidden_size={hidden_size}, image_size={image_size}, input_channels={input_channels}")

    def check_file_sizes(self):
        max_file_size=100*1024*1024
        acceptable_files=[]
        for root, _, files in os.walk('.'):
            for file in files:
                file_path = os.path.join(root,file)
                if os.path.getsize(file_path)<=max_file_size:acceptable_files.append(file_path)
        return acceptable_files

    def push_to_github(self, repo_url, is_private):
        try:
            if os.path.exists('.git'):repo=Repo('.')
            else:
                repo=Repo.init('.')
                repo.create_remote('origin',url=repo_url)
            repo.git.add(A=True)
            repo.index.commit("Update from Telegram bot")
            origin=repo.remote('origin')
            origin.push()
            self.send_telegram_message("Successfully pushed to GitHub repository.")
        except Exception as e:self.send_telegram_message(f"Error pushing to GitHub: {str(e)}")

    def collect_data(self):
        while True:
            label_choice=self.request_user_input("Does the label folder already exist? (yes/no)")
            if label_choice == "yes":
                existing_labels=os.listdir(self.data_dir) 
                while True:
                    self.send_telegram_message(f"Existing labels: {', '.join(existing_labels)}")
                    label=self.request_user_input("Enter the label folder name:")
                    if label not in existing_labels:self.send_telegram_message("Error: Label folder does not exist.")
                    else:break
            else:
                label=self.request_user_input("Enter the new label folder name:")
                os.makedirs(os.path.join(self.data_dir, label), exist_ok=True)
            while True:
                image_bytes=self.request_user_input(f"Upload an image for the label '{label}':")
                if isinstance(image_bytes, bytes):
                    image_path=os.path.join(self.data_dir, label, f"{int(time.time())}.png")
                    with open(image_path,'wb') as f:f.write(image_bytes)
                    self.send_telegram_message(f"Image saved to {image_path}")
                else:break
            next_action=self.request_user_input("Type 'stop' to finish collecting data, or 'continue' to add more images:")
            if next_action=="stop":return

    def show_random_seed_logs(self):
        logs = self.ai_dev.show_logs()
        if not logs:
            self.send_telegram_message("No logs found.")
            return
        message = "Random Seed Logs:\n"
        for log in logs:
            message += (f"Random Seed: {log['random_seed']}, Loss: {log['final_loss']}, "
                        f"Hyperparameters: {log['hyperparameters']}\n")
        self.send_telegram_message(message)

    def main_loop(self):
        self.clear_previous_messages()
        data_path_entered = self.request_user_input("Have you already entered the data path? (yes/no)")
        if data_path_entered == "stop":
            self.send_telegram_message("Process stopped by user command.")
            return
        if data_path_entered == "yes":
            dataset_type = "custom"
            self.get_hyperparameters(dataset_type)
            self.ai_dev.modelinit(dataset_type, self.data_dir)
            self.send_telegram_message("Model training has started.")
            epoch_losses = self.ai_dev.train()
            final_loss = epoch_losses[-1]
            hyperparameters = {
                "learning_rate": self.ai_dev.learning_rate,
                "batch_size": self.ai_dev.batch_size,
                "num_epochs": self.ai_dev.num_epochs,
                "hidden_size": self.ai_dev.hidden_size,
                "image_size": self.ai_dev.image_size,
                "input_channels": self.ai_dev.input_channels
            }
            self.ai_dev.save_log(hyperparameters, final_loss, self.ai_dev.random_seed)
            self.send_telegram_message(f"Training complete. Final loss: {final_loss:.4f}")
            self.ai_dev.plot_losses(epoch_losses)
            self.send_telegram_image('training_loss_plot.png')

            use_github = self.request_user_input("Do you want to use GitHub? (yes/no)")
            while use_github not in ["yes", "no"]:
                self.send_telegram_message("Invalid choice. Please enter 'yes' or 'no'.")
                use_github = self.request_user_input("Do you want to use GitHub? (yes/no)")
            if use_github == 'yes':
                acceptable_files = self.check_file_sizes()
                self.send_telegram_message(f"Files acceptable for GitHub push: {', '.join(acceptable_files)}")
                push_decision = self.request_user_input("Do you want to push these files to GitHub? (yes/no)")
                while push_decision not in ["yes", "no"]:
                    self.send_telegram_message("Invalid choice. Please enter 'yes' or 'no'.")
                    push_decision = self.request_user_input("Do you want to push these files to GitHub? (yes/no)")
                if push_decision == 'yes':
                    repo_url = self.request_user_input("Enter your GitHub repository URL:")
                    is_private = self.request_user_input("Is this a private repository? (yes/no)") == 'yes'
                    self.push_to_github(repo_url, is_private)

            while True:
                command = self.request_user_input("Enter 'rerun' to train again with new parameters, or 'stop' to end the program, 'cuda' to get CUDA status and stop, or an image to test model:")
                if isinstance(command, bytes):
                    self.ai_dev.test(command)
                else:
                    if command not in ["rerun", "stop", "cuda"]:self.send_telegram_message("Invalid choice. Please enter 'rerun', 'stop', 'cuda', or send image.")
                    if command == "stop":
                        self.send_telegram_message("Training stopped by user command.")
                        break
                    elif command == "rerun":
                        self.send_telegram_message("Rerunning the training with new parameters.")
                        continue
                    elif command == "cuda":self.get_cuda_options()

        else:
            use_onthefly_dataloader = self.request_user_input("Do you want to use an on-the-fly dataloader? (yes/no)")
            
            if use_onthefly_dataloader == "stop":
                self.send_telegram_message("Process stopped by user command.")
                return
            if use_onthefly_dataloader == "yes":
                dataset_choice = self.request_user_input("Do you want to use package-based data or custom data? (package/custom)")
                if dataset_choice == "stop":
                    self.send_telegram_message("Process stopped by user command.")
                    return
                while dataset_choice not in ["package", "custom"]:
                    self.send_telegram_message("Invalid choice. Please enter 'package' or 'custom'.")
                    dataset_choice = self.request_user_input("Do you want to use package-based data or custom data? (package/custom)")
                    if dataset_choice == "stop":
                        self.send_telegram_message("Process stopped by user command.")
                        return
                if dataset_choice == "package":
                    dataset_type = self.request_user_input("Choose a dataset (mnist/cifar10):")
                    while dataset_type not in ["mnist", "cifar10"]:
                        self.send_telegram_message("Invalid dataset choice. Please choose 'mnist' or 'cifar10'.")
                        dataset_type = self.request_user_input("Choose a dataset (mnist/cifar10):")
                        if dataset_type == "stop":
                            self.send_telegram_message("Process stopped by user command.")
                            return
                    next_action = "train"
                else:
                    dataset_type = "custom"
                    self.collect_data()
                    next_action = self.request_user_input("Data collection complete. Do you want to train or stop? (train/stop)")
                    while next_action not in ["train", "stop"]:
                        self.send_telegram_message("Invalid choice. Please enter 'train' or 'stop'.")
                        next_action = self.request_user_input("Data collection complete. Do you want to train or stop? (train/stop)")
                    if next_action == "stop":
                        self.send_telegram_message("Process stopped by user command.")
                        return
            else:
                dataset_type = self.request_user_input("Choose a dataset to use for training (mnist/cifar10/custom):")
                while dataset_type not in ["mnist", "cifar10", "custom"]:
                    self.send_telegram_message("Invalid dataset choice. Please choose 'mnist', 'cifar10', or 'custom'.")
                    dataset_type = self.request_user_input("Choose a dataset to use for training (mnist/cifar10/custom):")
                next_action = "train"
            if next_action == "train":
                self.show_random_seed_logs()
                seed_choice = self.request_user_input("Do you want to set a custom random seed? (yes/no)")
                if seed_choice == "yes":
                    random_seed = int(self.request_user_input("Enter the random seed value:"))
                    self.ai_dev.set_random_seed(random_seed)
                self.get_hyperparameters(dataset_type)
                self.ai_dev.modelinit(dataset_type, self.data_dir if dataset_type == "custom" else None)
                self.send_telegram_message("Model training has started.")
                epoch_losses = self.ai_dev.train()
                final_loss = epoch_losses[-1]
                hyperparameters = {
                    "learning_rate": self.ai_dev.learning_rate,
                    "batch_size": self.ai_dev.batch_size,
                    "num_epochs": self.ai_dev.num_epochs,
                    "hidden_size": self.ai_dev.hidden_size,
                    "image_size": self.ai_dev.image_size,
                    "input_channels": self.ai_dev.input_channels
                }
                self.ai_dev.save_log(hyperparameters, final_loss, self.ai_dev.random_seed)
                self.send_telegram_message(f"Training complete. Final loss: {final_loss:.4f}")
                self.ai_dev.plot_losses(epoch_losses)
                self.send_telegram_image('training_loss_plot.png')

                use_github = self.request_user_input("Do you want to use GitHub? (yes/no)")
                while use_github not in ["yes", "no"]:
                    self.send_telegram_message("Invalid choice. Please enter 'yes' or 'no'.")
                    use_github = self.request_user_input("Do you want to use GitHub? (yes/no)")
                if use_github == 'yes':
                    acceptable_files = self.check_file_sizes()
                    self.send_telegram_message(f"Files acceptable for GitHub push: {', '.join(acceptable_files)}")
                    push_decision = self.request_user_input("Do you want to push these files to GitHub? (yes/no)")
                    while push_decision not in ["yes", "no"]:
                        self.send_telegram_message("Invalid choice. Please enter 'yes' or 'no'.")
                        push_decision = self.request_user_input("Do you want to push these files to GitHub? (yes/no)")
                    if push_decision == 'yes':
                        repo_url = self.request_user_input("Enter your GitHub repository URL:")
                        is_private = self.request_user_input("Is this a private repository? (yes/no)") == 'yes'
                        self.push_to_github(repo_url, is_private)
                while True:
                    command = self.request_user_input("Enter 'rerun' to train again with new parameters, or 'stop' to end the program, 'cuda' to get CUDA status and stop, or an image to test model:")
                    if isinstance(command, bytes):
                        self.ai_dev.test(command)
                    else:
                        if command not in ["rerun", "stop", "cuda"]:
                            self.send_telegram_message("Invalid choice. Please enter 'rerun', 'stop', 'cuda', or send image.")
                        if command == "stop":
                            self.send_telegram_message("Training stopped by user command.")
                            break
                        elif command == "rerun":
                            self.send_telegram_message("Rerunning the training with new parameters.")
                            continue
                        elif command == "cuda":
                            self.get_cuda_options()
