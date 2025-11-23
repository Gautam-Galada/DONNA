# DonaDev 2.0 – Project Documentation

A Lightweight Telegram-Controlled ML Training & Experiment Tracking System

![image](https://github.com/user-attachments/assets/96119324-f3dd-4cf2-9194-68454e60302c)

---

## 📌 Introduction

**DonaDev 2.0** is a next-generation upgrade to the original Donna training bot.
It combines:

* A **Telegram Bot Interface**
* An **LLM-driven configuration system** powered by **Ollama (Llama 3)**
* A **lightweight training engine** (MNIST, CIFAR-10, and Custom Datasets)
* A **SQLite-backed experiment tracker** with dataset IDs, seed IDs, metrics, and artifacts
* Automatic plot generation, logging, and model testing
* A **zero-UI ML training experience** fully inside Telegram

DonaDev 2.0 allows any user—even beginners—to configure training, collect data, train models, test predictions, and log experiments **entirely from Telegram**, without writing a single line of code.

<div align="center">
  <img src="https://github.com/user-attachments/assets/4ca88502-4ff5-4578-ad4b-c394b24287ad" width="200" />
  <img src="https://github.com/user-attachments/assets/c9b1df0b-802f-4b5b-93fa-838b05c4fd6e" width="200" />
  <img src="https://github.com/user-attachments/assets/4daca6dd-f449-4670-b15f-a4d467a046f7" width="200" />
  <img src="https://github.com/user-attachments/assets/f3ddfba4-3cab-4270-ae4c-11da9f8e8ebd" width="200" />
</div>

---

## 🚀 What’s New in Version 2?

DonnaDev 2.0 introduces a redesigned architecture with:

### **1. SQLite-based Lightweight Experiment Tracking**

* `runs` table uses:

  * `dataset_id` instead of raw dataset names
  * `seed_id` instead of raw seed values
* Per-run configs stored cleanly
* Artifacts (plots/checkpoints) recorded in DB
* Seeds and datasets normalized to avoid duplication

### **2. LLM-Driven Hyperparameter Collection**

Ollama (Llama-3) dynamically interacts with the user to generate a **complete JSON config**, including:

* learning_rate
* batch_size
* num_epochs
* hidden_size
* image_size (length/width)
* input_channels
* random_seed
* dataset selection (MNIST, CIFAR-10, custom)

### **3. Custom Dataset Collection Inside Telegram**

Upload labeled images directly through Telegram → bot stores them → uses them for training.

### **4. Improved Modularity**

* `run_store.py`: experiment tracking
* `dona_dev.py`: Telegram + instruction flow
* `ai_dev.py`: ML implementation
* `main.py`: launch point

### **5. Plots & Artifacts Logging**

After training, Donna automatically:

* Generates training loss plot
* Sends plot to Telegram
* Logs plot in SQLite as an artifact

---

# ✨ Features (V2)

### ✔️ Control AI training entirely through Telegram

### ✔️ LLM-powered configuration (Llama 3 via Ollama)

### ✔️ Lightweight MNIST / CIFAR-10 / custom dataset trainer

### ✔️ On-device / remote / cloud Ollama inference

### ✔️ Real-time custom dataset collection from Telegram

### ✔️ SQLite Experiment Tracker (RunStore)

* runs
* datasets
* seeds
* metrics
* artifacts

### ✔️ Plot generation & Telegram image sending

### ✔️ Auto GitHub integration (optional)

### ✔️ Seed tracking for reproducibility

### ✔️ GPU check (CUDA/nvidia-smi)

---

# 📁 Project Structure (V2)

```
DONNA-main/
│
├── dona_dev.py         # Orchestrator: Telegram + Ollama + training logic
├── ai_dev.py           # Model code, datasets, training & plotting
├── run_store.py        # NEW: SQLite experiment tracking
├── main.py             # Start the DonnaDev bot
├── __init__.py
│
├── runs/               # All run logs + artifacts
│   ├── donna2.db       # SQLite database
│   ├── checkpoints/    # model weights
│   └── plots/          # training plots
│
└── requirements.txt
```

---

# 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/donadev.git
cd donadev
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

DonnaDev 2.0 requires:

* `torch`
* `torchvision`
* `matplotlib`
* `Pillow`
* `requests`
* `gitpython`
* `ollama`
* `sqlite3` (built-in)

---

# 🤖 Setting Up Your Telegram Bot

1. Open **BotFather** → create bot → get bot token
2. Get your **chat ID**
3. Update `main.py`:

```python
token = "YOUR_TELEGRAM_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"
data_dir = "data"       # folder for datasets
```

---

# 🧠 Setting Up the LLM Server (Ollama)

1. Install Ollama → [https://ollama.com](https://ollama.com)
2. Pull the Llama-3 model:

```bash
ollama pull llama3
```

3. Start server:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

4. Ensure port **11434** is open
5. In `main.py`, edit:

```python
ollama_host = "your-ollama-ip"
ollama_port = 11434
```

---

# ▶️ Running DonnaDev

```bash
python main.py
```

---

# 💬 Using the Bot

Once started, open Telegram and chat with your bot.

DonnaDev will:

* Ask configuration questions through LLM dialog
* Generate JSON configuration automatically
* Train the model
* Send training loss plots
* Accept new commands:

  * **send image → model predicts**
  * **rerun → new config**
  * **stop → end session**
  * **shell:<cmd> → run a system command**
  * **custom commands → defined in main.py**

---

# 🧪 Training a Model

### 1. Choose dataset

* MNIST
* CIFAR-10
* Custom (upload images)

### 2. LLM collects parameters

Example:

```
learning_rate: 0.001
batch_size: 64
num_epochs: 5
image_size: 28×28
random_seed: 42
```

### 3. DonnaDev trains & logs internally

* SQLite entry created with:

  * dataset_id
  * seed_id
  * hyperparameters
  * run_id
  * timestamps
* Training loss plotted
* Result sent to Telegram
* Plot saved as an artifact

---

# 🗄️ Experiment Tracking (V2)

`run_store.py` manages everything:

### **Tables**

* `datasets`
* `seeds`
* `runs`
* `metrics`
* `artifacts`

### **Logged automatically**

* Training hyperparameters
* Dataset information
* Random seed
* Epoch-by-epoch metrics (optional future extension)
* Loss plot artifact

---

# 🌐 GitHub Integration (Optional)

DonnaDev can:

* Commit all files
* Push to any repository

(Enable in settings; requires GitHub PAT)

---

# 🛑 Ending a Session

* Send **"stop"**
* Or send **"rerun"** to start a new configuration
* Or send an image to test prediction

---

# 📦 Dependencies

```
torch
torchvision
requests
matplotlib
Pillow
ollama
gitpython
```

---

# 🟢 Status of Version 2

✔️ New architecture implemented
✔️ SQLite experiment tracking (dataset_id + seed_id)
✔️ LLM-driven config system works
✔️ Telegram control loop completed
✔️ Artifact logging (plots)
✔️ Custom dataset collector integrated
✔️ Training pipeline stable

🔄 Coming in a future update:

* Per-epoch metrics logging
* Accuracy/best_val_acc
* Formal command router (/new, /train, /status)
* Better artifact folder structure
* Model zoo extensions

