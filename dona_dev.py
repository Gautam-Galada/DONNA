import os
import time
import json
import requests
import subprocess
import datetime

from git import Repo
from ollama import Client

from run_store import RunStore


class DonaDev:
    def __init__(
        self,
        token,
        chat_id,
        ai_dev,
        data_dir,
        git_token=None,
        ollama_host="localhost",
        ollama_port=11434,
        custom_commands=None,
        db_path=None,
    ):
        """
        Orchestrator for Donna 2.0.

        - Talks to Telegram
        - Uses Ollama to collect config as JSON
        - Delegates training to ai_dev
        - Logs runs into RunStore using dataset_id + seed_id
        """
        if custom_commands is None:
            custom_commands = {}

        self.TOKEN = token
        self.chat_id = chat_id
        self.ai_dev = ai_dev
        self.data_dir = data_dir
        self.last_update_id = None
        self.git_token = git_token

        # Lightweight run store (SQLite)
        self.store = RunStore(db_path) if db_path is not None else RunStore()

        # LLM wrapper
        self.llm = OllamaOperation(
            ollama_host=ollama_host,
            ollama_port=ollama_port,
            userin=self.request_user_input_noprompt,
            userop=self.send_telegram_message,
        )

        # Custom commands (lower-cased keys)
        self.commands = {k.lower(): v for k, v in custom_commands.items()}

    # ------------------------------------------------------------------
    # Telegram helpers
    # ------------------------------------------------------------------

    def get_commands_list_text(self):
        if not self.commands:
            return ""
        return ", ".join(self.commands.keys())

    def send_telegram_message(self, message: str):
        url = (
            f"https://api.telegram.org/bot{self.TOKEN}/sendMessage"
            f"?chat_id={self.chat_id}&text={requests.utils.quote(str(message))}"
        )
        response = requests.get(url)
        try:
            resp_json = response.json()
        except Exception:
            resp_json = {"ok": False, "raw": response.text}
        print(f"Sent message: {message}, Response: {resp_json}")

    def send_telegram_image(self, image_path: str):
        url = f"https://api.telegram.org/bot{self.TOKEN}/sendPhoto"
        with open(image_path, "rb") as image_file:
            files = {"photo": image_file}
            data = {"chat_id": self.chat_id}
            response = requests.post(url, files=files, data=data)
        try:
            resp_json = response.json()
        except Exception:
            resp_json = {"ok": False, "raw": response.text}
        print(f"Sent image: {image_path}, Response: {resp_json}")

    def get_updates(self, offset=None):
        url = f"https://api.telegram.org/bot{self.TOKEN}/getUpdates"
        params = {"offset": offset, "timeout": 30}
        response = requests.get(url, params=params).json()
        return response.get("result", [])

    def clear_previous_messages(self):
        updates = self.get_updates()
        if updates:
            self.last_update_id = updates[-1]["update_id"] + 1
        print(f"Cleared previous messages. Last update ID: {self.last_update_id}")

    # ------------------------------------------------------------------
    # System helpers
    # ------------------------------------------------------------------

    def get_cuda_options(self):
        try:
            p = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True)
            if p.returncode != 0:
                msg = "Device doesn't have GPU or CUDA drivers (nvidia-smi error)."
                print(msg)
                self.send_telegram_message(msg)
                return
            for line in p.stdout.splitlines():
                self.send_telegram_message(line)
        except Exception:
            msg = "Device doesn't have GPU or CUDA drivers (exception)."
            print(msg)
            self.send_telegram_message(msg)

    # ------------------------------------------------------------------
    # User input
    # ------------------------------------------------------------------

    def request_user_input_noprompt(self):
        return self.request_user_input("")

    def request_user_input(self, prompt: str):
        """
        Ask the user for text or image via Telegram.
        Returns:
            - str (lowercased) if text
            - bytes if image
            - None if timeout
        """
        if prompt:
            self.send_telegram_message(prompt)

        start_time = time.time()
        timeout = 300  # 5 minutes

        while time.time() - start_time < timeout:
            updates = self.get_updates(self.last_update_id)
            for update in updates:
                self.last_update_id = update["update_id"] + 1

                if "message" in update and "text" in update["message"]:
                    return update["message"]["text"].lower()

                if "message" in update and "photo" in update["message"]:
                    image_id = update["message"]["photo"][0]["file_id"]
                    file_info = requests.get(
                        f"https://api.telegram.org/bot{self.TOKEN}/getFile?file_id={image_id}"
                    ).json()
                    file_path = file_info["result"]["file_path"]
                    image_bytes = requests.get(
                        f"https://api.telegram.org/file/bot{self.TOKEN}/{file_path}"
                    ).content
                    return image_bytes
            time.sleep(1)

        self.send_telegram_message(
            "No input received within 5 minutes. Using default value."
        )
        return None

    # ------------------------------------------------------------------
    # Default hyperparameters
    # ------------------------------------------------------------------

    def get_hyperparam(self):
        datamnist = {
            "learning_rate": 0.01,
            "batch_size": 64,
            "num_epochs": 2,
            "hidden_size": 512,
            "image_size": {"length": 28, "width": 28},
            "input_channels": 1,
        }
        datacifr = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 2,
            "hidden_size": 512,
            "image_size": {"length": 32, "width": 32},
            "input_channels": 3,
        }
        datacustom = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 2,
            "hidden_size": 512,
            "image_size": {"length": 28, "width": 28},
            "input_channels": 3,
        }
        # NOTE: key for CIFAR stays "cifr10" for backwards compatibility
        data = {"mnist": datamnist, "cifr10": datacifr, "custom": datacustom}
        return data

    # ------------------------------------------------------------------
    # GitHub / filesystem helpers
    # ------------------------------------------------------------------

    def check_file_sizes(self):
        max_file_size = 100 * 1024 * 1024
        acceptable_files = []
        for root, _, files in os.walk("."):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) <= max_file_size:
                    acceptable_files.append(file_path)
        return acceptable_files

    def push_to_github(self, repo_url, is_private):
        try:
            if os.path.exists(".git"):
                repo = Repo(".")
            else:
                repo = Repo.init(".")
                repo.create_remote("origin", url=repo_url)

            repo.git.add(A=True)
            repo.index.commit("Update from Telegram bot")
            origin = repo.remote("origin")
            origin.push()
            self.send_telegram_message("Successfully pushed to GitHub repository.")
        except Exception as e:
            self.send_telegram_message(f"Error pushing to GitHub: {str(e)}")

    # ------------------------------------------------------------------
    # Data collection for custom dataset
    # ------------------------------------------------------------------

    def collect_data(self):
        while True:
            label_choice = self.request_user_input(
                "Does the label folder already exist? (yes/no)"
            )
            if label_choice == "yes":
                existing_labels = os.listdir(self.data_dir)
                while True:
                    self.send_telegram_message(
                        f"Existing labels: {', '.join(existing_labels)}"
                    )
                    label = self.request_user_input("Enter the label folder name:")
                    if label not in existing_labels:
                        self.send_telegram_message("Error: Label folder does not exist.")
                    else:
                        break
            else:
                label = self.request_user_input("Enter the new label folder name:")
                os.makedirs(os.path.join(self.data_dir, label), exist_ok=True)

            while True:
                image_bytes = self.request_user_input(
                    f"Upload an image for the label '{label}':"
                )
                if isinstance(image_bytes, bytes):
                    image_path = os.path.join(
                        self.data_dir, label, f"{int(time.time())}.png"
                    )
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    self.send_telegram_message(f"Image saved to {image_path}")
                else:
                    break

            next_action = self.request_user_input(
                "Type 'stop' to finish collecting data, or 'continue' to add more images:"
            )
            if next_action == "stop":
                return

    # ------------------------------------------------------------------
    # Logs (legacy ai_dev logs)
    # ------------------------------------------------------------------

    def show_random_seed_logs(self):
        logs = self.ai_dev.show_logs()
        if not logs:
            self.send_telegram_message("No logs found.")
            return
        message = "Random Seed Logs:\n"
        for log in logs:
            message += (
                f"Random Seed: {log['random_seed']}, "
                f"Loss: {log['final_loss']}, "
                f"Hyperparameters: {log['hyperparameters']}\n"
            )
        self.send_telegram_message(message)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def process_repo_url(self, repo_url):
        if self.git_token:
            repo_url = repo_url.replace(
                "github.com", f"{self.git_token}@github.com"
            )
        return repo_url

    def run_command(self, command):
        try:
            p = subprocess.run(command, capture_output=True, text=True, shell=True)
            if p.stdout:
                self.send_telegram_message(p.stdout)
            if p.stderr:
                self.send_telegram_message(p.stderr)
        except Exception:
            self.send_telegram_message("Some error has occurred while running command.")

    # ------------------------------------------------------------------
    # Main orchestration loop
    # ------------------------------------------------------------------

    def main_loop1(self):
        """
        Single training session:
        - wipe old Telegram updates
        - get config JSON from Ollama
        - hydrate defaults
        - set up dataset_id & seed_id for RunStore
        - run training and log artifacts
        """
        self.clear_previous_messages()
        defaults = self.get_hyperparam()

        # 1) Get config JSON from LLM
        inp = self.llm.ollama_runner()
        config = json.loads(inp)

        default_checker = [
            "learning_rate",
            "batch_size",
            "num_epochs",
            "hidden_size",
            "input_channels",
        ]
        dataset = "custom"

        if isinstance(config.get("data_path_entered"), dict) and "no" in config[
            "data_path_entered"
        ]:
            dataset = config["data_path_entered"]["no"]["dataset"]

        # Fill scalar defaults
        for key in default_checker:
            if config.get(key) == "default":
                config[key] = defaults[dataset][key]

        # Fill image size defaults
        default_checker2 = ["length", "width"]
        for key in default_checker2:
            if config["image_size"][key] == "default":
                config["image_size"][key] = defaults[dataset]["image_size"][key]

        print("Final config from LLM:", config)

        # 2) Apply config to ai_dev
        self.ai_dev.learning_rate = config["learning_rate"]
        self.ai_dev.batch_size = config["batch_size"]
        self.ai_dev.num_epochs = config["num_epochs"]
        self.ai_dev.hidden_size = config["hidden_size"]
        self.ai_dev.image_size = (
            config["image_size"]["length"],
            config["image_size"]["width"],
        )
        self.ai_dev.input_channels = config["input_channels"]

        # 3) Data collection if custom and user said "no" to data_path_entered
        if isinstance(config.get("data_path_entered"), dict) and "no" in config[
            "data_path_entered"
        ] and dataset == "custom":
            self.collect_data()

        dataset_type = dataset

        # 4) Seed handling (we still ask user for random_seed; internally we map to seed_id)
        user_seed_val = config.get("random_seed", "default")
        seed_value = None

        # Set seed if user provided numeric
        if user_seed_val != "default":
            try:
                seed_value = int(user_seed_val)
                self.ai_dev.set_random_seed(seed_value)
            except Exception:
                seed_value = None  # fallback to ai_dev internal value

        # If still None, try to read from ai_dev; if missing, set a default
        if seed_value is None:
            seed_value = getattr(self.ai_dev, "random_seed", None)

        if seed_value is None:
            # Ensure we have a reproducible seed
            seed_value = 42
            try:
                self.ai_dev.set_random_seed(seed_value)
            except Exception:
                pass

        # 5) CUDA info if requested
        if config.get("view_cuda_details") == "yes":
            self.get_cuda_options()

        # 6) Initialize model
        self.ai_dev.modelinit(
            dataset_type, self.data_dir if dataset_type == "custom" else None
        )

        # 7) Create dataset_id + seed_id and insert run row (RunStore)
        dataset_id = self.store.get_or_create_dataset(
            name=dataset_type,
            version=None,
            split_hash="default",
            note=None,
        )
        seed_id = self.store.get_or_create_seed(seed_value)

        run_id = f"run-{int(time.time())}"
        created_at = datetime.datetime.utcnow().isoformat()

        run_config = {
            "model_name": "SimpleNet",
            "learning_rate": self.ai_dev.learning_rate,
            "batch_size": self.ai_dev.batch_size,
            "num_epochs": self.ai_dev.num_epochs,
            "hidden_size": self.ai_dev.hidden_size,
        }

        self.store.create_run(
            run_id=run_id,
            created_at=created_at,
            dataset_id=dataset_id,
            seed_id=seed_id,
            config=run_config,
            device="auto",  # or detect from ai_dev if you prefer
            status="created",
            notes=None,
        )

        # 8) Train
        self.send_telegram_message("Model training has started.")
        epoch_losses = self.ai_dev.train()
        final_loss = epoch_losses[-1]

        hyperparameters = {
            "learning_rate": self.ai_dev.learning_rate,
            "batch_size": self.ai_dev.batch_size,
            "num_epochs": self.ai_dev.num_epochs,
            "hidden_size": self.ai_dev.hidden_size,
            "image_size": self.ai_dev.image_size,
            "input_channels": self.ai_dev.input_channels,
        }

        # Legacy file-based log for backwards compatibility
        self.ai_dev.save_log(hyperparameters, final_loss, seed_value)

        # Update run status in DB
        self.store.update_run_status(
            run_id=run_id,
            status="done",
            best_val_acc=None,  # if ai_dev exposes accuracy you can set it here
            train_samples=None,
            val_samples=None,
        )

        self.send_telegram_message(f"Training complete. Final loss: {final_loss:.4f}")

        # 9) Plot + artifact registration
        self.ai_dev.plot_losses(epoch_losses)
        plot_path = "training_loss_plot.png"
        self.send_telegram_image(plot_path)
        self.store.add_artifact(run_id=run_id, type_="plot", path=plot_path)

        # 10) Post-training interactive loop
        next_action = ""
        keep_running = True

        while keep_running:
            command = self.request_user_input(
                "Enter 'rerun' to train again with new parameters, or 'stop' to end the "
                "program, or send an image to test the model, or run a custom command: "
                f"{self.get_commands_list_text()}, or start a shell command with 'shell:'."
            )

            if isinstance(command, bytes):
                # Image: test model
                self.send_telegram_message(self.ai_dev.test(command))
                continue

            # Text command
            if (
                command not in ["rerun", "stop"]
                and command not in self.commands
                and not str(command).startswith("shell:")
            ):
                self.send_telegram_message(
                    "Invalid choice. Please enter 'rerun', 'stop', send an image, "
                    f"a command from {self.get_commands_list_text()}, "
                    "or start a shell command with 'shell:'."
                )
                continue

            if command == "stop":
                self.send_telegram_message("Training stopped by user command.")
                keep_running = False
                break

            if command == "rerun":
                self.send_telegram_message(
                    "Rerunning the training with new parameters."
                )
                next_action = "rerun"
                keep_running = False
                break

            if command in self.commands:
                cmd = self.commands[command]
                self.run_command(cmd)
                continue

            if str(command).startswith("shell:"):
                self.run_command(str(command)[6:])
                continue

        if next_action == "rerun":
            self.main_loop1()


# ======================================================================
# Ollama Operation
# ======================================================================


class OllamaOperation:
    def __init__(self, ollama_host="localhost", ollama_port=11434, userin=input, userop=print):
        self.ollama = Client(host=f"{ollama_host}:{ollama_port}")
        self.chat = self.ollama.chat
        self.userin = userin
        self.userop = userop

        # NOTE: JSON still uses "random_seed" (user chooses seed value).
        # Internally we convert to seed_id via RunStore.
        self.initprompt = """
You are an assistant called DONNA. You need to interact with the user to get information to fill the below JSON. Each key in the JSON will have a unique value which is not a list. In the given JSON the lists show the possible choices. You dont have to input the key jsoncomplete, just have it in the output. The word 'jsonbegin' should prefix the json and 'jsonend' should be after it.

"value" is not a possible choice. "value" should be replaced by a numeric value from the user. When you have received all the information to fill the JSON, Output the JSON without any other text or explanation.

{
    "data_path_entered": [
        "yes",
        {
            "no": {
                "dataset": [
                    "mnist",
                    "cifr10",
                    "custom"
                ]
            }
        }
    ],
    "learning_rate": [
        "value",
        "default"
    ],
    "batch_size": [
        "value",
        "default"
    ],
    "num_epochs": [
        "value",
        "default"
    ],
    "hidden_size": [
        "value",
        "default"
    ],
    "image_size": {
        "length": [
            "value",
            "default"
        ],
        "width": [
            "value",
            "default"
        ]
    },
    "input_channels": [
        "value",
        "default"
    ],
    "random_seed": [
        "value",
        "default"
    ],
    "view_cuda_details": [
        "yes",
        "no"
    ],
    "jsoncomplete": "jsonyes"
}

For example, a filled JSON would be of the form
jsonbegin
{
    "data_path_entered":{
        "no": {
            "dataset": "mnist"
        }
    },
    "learning_rate": "default",
    "batch_size": "default",
    "num_epochs": 10,
    "hidden_size": "default",
    "image_size": {
        "length": 32,
        "width":  32
    },
    "input_channels": "default",
    "random_seed": 15,
    "view_cuda_details": "no",
    "jsoncomplete": "jsonyes"
}
jsonend

Remember, your last message should be only the filled json without any other text or exaplanation or message.
"""

    def get_ollama(self, messages, userop=print):
        res = ""
        for part in self.chat("llama3", messages=messages, stream=True):
            x = part["message"]["content"]
            print(x, end="", flush=True)
            res += x
        print()

        msg = {"role": "assistant", "content": res}
        if "jsoncomplete" in msg["content"] and "jsonyes" in msg["content"]:
            # JSON is complete; no need to echo separately
            pass
        else:
            userop(res)
        return msg

    def get_user(self, userin=input):
        inp = userin()
        print(inp)
        msg = {"role": "user", "content": inp}
        return msg

    def ollama_runner(self):
        messages = []
        first = {"role": "user", "content": self.initprompt}
        messages.append(first)

        while True:
            ollamaresp = self.get_ollama(messages, self.userop)
            if "jsoncomplete" in ollamaresp["content"] and "jsonyes" in ollamaresp[
                "content"
            ]:
                cont = ollamaresp["content"]
                a = cont.find("jsonbegin") + len("jsonbegin")
                b = cont.find("jsonend")
                print("Complete")
                return cont[a:b]
            messages.append(ollamaresp)
            userresp = self.get_user(self.userin)
            messages.append(userresp)
