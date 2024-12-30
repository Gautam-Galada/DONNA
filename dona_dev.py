import os
import time
import requests
import subprocess
from git import Repo

import json

class DonaDev:
    def __init__(self,token,chat_id,ai_dev,data_dir, git_token=None, ollama_host='localhost', ollama_port=11434, custom_commands={}):
        self.TOKEN=token
        self.chat_id=chat_id
        self.ai_dev=ai_dev
        self.data_dir=data_dir
        self.last_update_id=None
        self.git_token=git_token
        self.llm= OllamaOperation(ollama_host=ollama_host, ollama_port=ollama_port, userin=self.request_user_input_noprompt, userop=self.send_telegram_message)
        self.commands={}
        for x in custom_commands.keys():
             self.commands[x.lower()]=custom_commands[x]

    def get_commands_list_text(self):
        ret=''
        for x in self.commands.keys():
             ret=ret+x+' ,'
        ret=ret[:-1]
        return ret

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
        try:
            p=subprocess.run(['nvidia-smi','-q'],capture_output=True,text=True)
            pret=p.returncode
            if pret != 0:
                print("Device doesn't have GPU or CUDA drivers error")
                self.send_telegram_message("Device doesn't have GPU or CUDA drivers error")
                return
            pop=p.stdout
            for popline in pop.splitlines():self.send_telegram_message(popline)
        except:
            print("Device doesn't have GPU or CUDA drivers error")
            self.send_telegram_message("Device doesn't have GPU or CUDA drivers error")
        
    def request_user_input_noprompt(self):
        return self.request_user_input('')


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

    def get_hyperparam(self):
        datamnist={
            "learning_rate": 0.01,
            "batch_size": 64,
            "num_epochs": 2,
            "hidden_size": 512,
            "image_size": {
                "length": 28,
                "width":  28
            },
            "input_channels": 1,
        }
        datacifr={
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 2,
            "hidden_size": 512,
            "image_size": {
                "length": 32,
                "width":  32
            },
            "input_channels": 3,
        }
        datacustom={
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 2,
            "hidden_size": 512,
            "image_size": {
                "length": 28,
                "width":  28
            },
            "input_channels": 3,
        }
        data={"mnist":datamnist, "cifr10":datacifr, "custom": datacustom}
        return data
        
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

    def process_repo_url(self,repo_url):
        repo_url=repo_url.replace('github.com', self.git_token+'@github.com')
        return repo_url
    
    def run_command(self, command):
        try:
            p = subprocess.run(command, capture_output=True, text=True)
            self.send_telegram_message(p.stdout)
            self.send_telegram_message(p.stderr)
        except:
            self.send_telegram_message('Some error has occurred')


    def main_loop1(self):
        self.clear_previous_messages()
        defaults=self.get_hyperparam()
        inp=self.llm.ollama_runner()
        config=json.loads(inp)
        default_checker=['learning_rate', 'batch_size', 'num_epochs', 'hidden_size', 'input_channels']
        dataset='custom'
        if 'no' in config['data_path_entered']:
            dataset=config['data_path_entered']['no']['dataset']

        for x in default_checker:
            if config[x]=='default':
                config[x]=defaults[dataset][x]

        default_checker2=['length', 'width']
        for x in default_checker2:
            if config['image_size'][x]=='default':
                config['image_size'][x]=defaults[dataset]['image_size'][x]
        print(config)

        self.ai_dev.learning_rate=config['learning_rate']
        self.ai_dev.batch_size=config['batch_size']
        self.ai_dev.num_epochs=config['num_epochs']
        self.ai_dev.hidden_size=config['hidden_size']
        self.ai_dev.image_size=(config['image_size']['length'], config['image_size']['width'])
        self.ai_dev.input_channels=config['input_channels']

        if 'no' in config['data_path_entered'] and dataset=='custom':
            self.collect_data()

        dataset_type=dataset
        
        try:
            if config['random_seed'] != 'default':
                self.ai_dev.set_random_seed(config['random_seed'])
        except:
            pass

        if config['view_cuda_details']=='yes':
            self.get_cuda_options()

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
        next=''
        kk= True
        while kk:
            command = self.request_user_input("Enter 'rerun' to train again with new parameters, or 'stop' to end the program, or an image to test model, or run a custom command: "+self.get_commands_list_text()+", or start writing custom command with shell:")
            if isinstance(command, bytes):
                self.send_telegram_message(self.ai_dev.test(command))
            else:
                if command not in ["rerun", "stop"] and command not in self.commands and not command.startswith('shell:'):
                    self.send_telegram_message("Invalid choice. Please enter 'rerun', 'stop', or send image, or a command from "+self.get_commands_list_text()+", or start writing custom command with shell:")
                if command == "stop":
                        self.send_telegram_message("Training stopped by user command.")
                        kk=False
                        break
                elif command == "rerun":
                        self.send_telegram_message("Rerunning the training with new parameters.")
                        next='rerun'
                        kk=False
                        break
                elif command in self.commands:
                        cmd=self.commands[command]
                        self.run_command(cmd)
                elif command.startswith('shell:'):
                        self.run_command(command[6:])
                

        if next=='rerun':
            self.main_loop1()
                        
from ollama import *

class OllamaOperation:
	def __init__(self, ollama_host='localhost', ollama_port=11434, userin=input, userop=print):
		self.ollama=Client(host=ollama_host+':'+str(ollama_port))
		self.chat=self.ollama.chat
		self.userin=userin
		self.userop=userop
		self.initprompt='''
You are an assistant called DONNA. You need to interact with the user to get information to fill the below JSON. Each key in the JSON will have an unique value which is not a list. In the given JSON the lists show the possible choices. You dont have to input the key jsoncomplete, just have it in the output. The word 'jsonbegin' should prefix the json and 'jsonend' should be after it.

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
	'''

	def get_ollama(self, messages, userop=print):
		res=''
		for part in self.chat('llama3', messages=messages, stream=True):
			x=part['message']['content']
			print(x, end='', flush=True)
			res=res+x
		print()
		msg={}
		msg['role']='assistant'
		msg['content']=res
		if 'jsoncomplete' in msg['content'] and 'jsonyes' in msg['content']:
			pass
		else:
			userop(res)
		return msg
	
	def get_user(self, userin=input):
		inp=userin()
		print(inp)
		msg={}
		msg['role']='user'
		msg['content']=inp
		return msg
	
	def ollama_runner(self):
		messages=[]
		msg={}
		msg['role']='user'
		msg['content']=self.initprompt

		messages.append(msg)
		while True:
			ollamaresp=self.get_ollama(messages, self.userop)
			if 'jsoncomplete' in ollamaresp['content'] and 'jsonyes' in ollamaresp['content']:
				cont=ollamaresp['content']
				a=cont.find('jsonbegin')+len('jsonbegin')
				b=cont.find('jsonend')
				print("Complete")
				return cont[a:b]
			messages.append(ollamaresp)
			userresp=self.get_user(self.userin)
			messages.append(userresp)
		

