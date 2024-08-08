every ai dev needs a DONNA, to stepout once in a while and get some sunshine and caffeine.

![image](https://github.com/user-attachments/assets/97e06283-abc8-4934-a118-21590f170c5c) ![image](https://github.com/user-attachments/assets/49eed0bd-9763-492e-b0fa-f378106bda4b)

## Now with testing on the fly
![image](https://github.com/user-attachments/assets/7e944a19-5a8f-4b4f-aa72-dcd76b43c03a) ![image](https://github.com/user-attachments/assets/d43e9de4-e6bf-411d-a497-326ca12bf55d)

## Project Workflow

1. **User Credentials and Setup**
   - **Telegram Integration**: User provides Telegram credentials, bot ID, and chat ID.
   - **Dataloader Option**: Query if the developer has a dataloader or needs to create one on-the-fly. Choices:
     - Use an existing dataloader (if user has predefined path).
     - Create an on-the-fly dataloader with predefined or dynamically generated labels.
     - Utilize a pre-built package.

2. **Seed Management**
   - **Seed Selection**: Prompt for changing the seed with existing comparisons or using the same seed for reproducibility.

3. **Training Process**
   - **Checkpoints**: Determine whether to use existing checkpoints or start training from scratch.
   - **Hyperparameters**: Ask the user if they want to use custom hyper parameters or current, with current being displayed. (scalable)
   - **Training Start**: Display a message indicating that training has begun. Display the training plots (scalable). 
   - **User Choices**: Provide options to retrain, stop, or upload files to GitHub, or display cuda information.
   - **GitHub Upload**:
     - pre-verified local authentication should be done.
     - Ask for repository creation or check for existing repositories by asking the repo name.
     - Push all possible files and folders, display their names.
   - **Final Steps**: Prompt to rerun the process or stop.

